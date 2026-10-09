# 통합 배치 스케줄러 설계 (unified batch scheduler)

> 목적: `multitask_test/online.cc`(통짜 라우팅)와 `split_profiler/jackson_online.cc`(1컷 분할 + Jackson)를
> **하나의 오프라인 배치 플래너**로 합친다. N개의 task가 한 번에 주어졌을 때,
> **CPU·GPU 이기종 자원을 모두 활용하면서 전체 makespan을 최단으로** 끝내도록
> 각 task의 실행 모드(통짜 GPU / 통짜 CPU / 1컷 분할)와 스케줄을 오프라인에 확정한다.
>
> 환경 기준: RTX 3060 Laptop(6GB, OpenCL) + i5-11400H(6C) / TFLite fp32.
> 상태: **설계(합의 대기)**. 구현 전 이 문서의 "8. 미해결" 확정 필요.

---

## 1. 목표와 범위

- **입력**: task 집합 `{(model, count)}` — 도착시각 없음, 전부 t0에 가용(배치).
- **자원**: GPU 1개(직렬 커맨드 스트림) + CPU P코어(병렬).
- **출력**: task별 실행 모드 + 실행 순서(plan) → 그대로 실행해 makespan 실측.
- **목적함수**: makespan 최소. 부차적으로 두 레인(GPU/CPU)을 균형 맞춰 **놀리는 자원 없이**.
- **범위 밖(지금 아님)**: 온라인 도착/큐잉, 실시간 마감, 배칭(같은 모델 batch invoke). → "9. 로드맵".

### 현실적 천장 (정직)
- fp32 + discrete GPU에서 **CPU가 4~100배 느림**. 이론 하한 `LB=max(GPUside,CPUside)`에서
  대부분 워크로드는 `GPUside ≫ CPUside`(불균형 84~92%, [scheduler_visualization.md](scheduler_visualization.md) Fig B).
- 따라서 **makespan 하한 ≈ GPU 총작업량**이고, CPU로 옮겨 얻는 이득은 "GPU가 병목이고 CPU에 여유가
  있을 때 GPU 일부를 덜어내는" 딱 그만큼. 큰 이득(≫1.1x)은 **int8/CPU 경쟁력**이 붙어야 나온다.
- → 이 플래너의 정직한 성공 기준: **"주어진 자원에서 이론 하한 LB에 최대한 근접"**(과대주장 금지).

---

## 2. 통합 실행 모델

### 2.1 실행 단위(ExecUnit) 추상화
두 도구의 차이를 하나의 단위로 흡수한다. task 인스턴스 i는 다음 중 하나:

| 모드 | GPU 기여 | CPU 기여 | GPU 컨텍스트 | 근거 도구 |
|---|---|---|---|---|
| `WHOLE_GPU` | `Gwhole` | 0 | **모델당 공유 1개** | online.cc |
| `WHOLE_CPU` | 0 | `Cwhole` | 없음 | online.cc |
| `SPLIT(dir,k)` | `gpu_stage` | `cpu_stage` | **인스턴스당 1개** | jackson_online.cc |

- `SPLIT`은 두 stage가 **순차 의존**(flow shop): CPU-first면 CPU→(경계전송)→GPU, GPU-first면 반대.
- **컨텍스트 비용의 비대칭이 핵심 제약**: WHOLE_GPU는 모델당 컨텍스트 1개(50~120개도 정상,
  [scheduler_visualization.md](scheduler_visualization.md) §7)지만 SPLIT은 인스턴스마다 컨텍스트라
  ~15개 초과 시 VRAM 폭발(§5, 30개서 2000배 느림). → **SPLIT은 예산제**로 제한한다.

### 2.2 두 레인 모델 (makespan 추정)
```
GPUlane = Σ_i (GPU 기여)          # GPU는 직렬 → 단순 합
CPUlane = CpuMakespan({CPU 기여}, P)   # P코어 LPT 분배 후 최대 부하 (online.cc:109 재사용)
makespan ≈ max(GPUlane, CPUlane) + Θ_coupling
```
- `Θ_coupling` = flow-shop 겹침 손실 + 경계 전송(PCIe) + per-instance GPU 디스패치 오버헤드.
  이 항 때문에 실측이 `LB` 위 +13~28%로 뜬다([figA](figures/figA_optimality_gap.svg)). 플래너는
  `max(GPUlane, CPUlane)`를 최소화하되 `Θ_coupling`을 **패널티로 근사 반영**(3.3).
- **CPU 경합 보정** `α≈1.5`: CPUlane 예측에 곱해 단독측정 C의 낙관성을 보정
  (online.cc:137 규칙 계승, 단 값은 4.1에서 재교정).

---

## 3. 오프라인 플래너 알고리즘 (핵심)

### 3.1 발상: "전부 GPU"에서 시작해 병목 레인을 덜어낸다
GPU가 거의 항상 병목이므로, **GPU→CPU로 일을 흘려 두 레인을 균형점까지** 맞춘다.
흘리는 방법 두 가지의 marginal 효율을 비교한다:

- **통짜 offload** (task i를 WHOLE_CPU로): GPUlane `−Gwhole_i`, CPUlane `+α·Cwhole_i`, 컨텍스트 비용 0.
- **분할** (task i를 SPLIT로): GPUlane `−(Gwhole_i − gpu_stage_i)`, CPUlane `+α·cpu_stage_i`,
  + 경계 전송 패널티, **컨텍스트 예산 1 소비**.

### 3.2 절차
```
입력: 인스턴스 목록, P, 컨텍스트예산 B_split, α
1. 모든 인스턴스 = WHOLE_GPU.  GPUlane=ΣGwhole, CPUlane=0.
2. cur = max(GPUlane, CPUlane).
3. 반복:
     후보 = { 각 인스턴스에 대해 적용 가능한 move(통짜offload | split) }
     각 move의 Δmakespan = max(GPUlane', CPUlane') + penalty − cur  계산
       (split은 B_split>0 일 때만; penalty=경계전송+디스패치)
     best = Δ 최소인 move
     ★가드: CPUlane' ≤ GPUlane' 유지(과잉 offload 방지, online.cc:161 규칙 계승)
     if best.Δ < 0 and 가드 통과:
        적용. GPU/CPUlane 갱신. split이면 B_split--.
     else: 중단
4. 순서결정:
     - SPLIT 인스턴스: 방향별 Johnson 정렬 후 인터리브 (jackson_online.cc:92,207 재사용)
     - WHOLE_GPU: GPU 레인 뒤에 이어붙임(직렬 합이라 순서 무관)
     - WHOLE_CPU: CPU 워커풀에 LPT
5. plan 출력 (인스턴스별 모드/방향/k + 디스패치 순서).
```

### 3.3 왜 이게 "이기종 최대 활용 + 최단"인가
- **균형 가드**가 CPUlane을 GPUlane 아래로 유지 → makespan≈GPUlane, 그리고 GPUlane은 흘린 만큼 감소.
  즉 CPU가 놀지 않는 선에서 GPU 부담을 최대로 던다 → `max` 최소화.
- **marginal 효율 순 선택**은 "GPU 1ms 덜 때 CPU에 얼마 얹히나"가 가장 싼 것부터 → 같은 CPU 예산으로
  GPU를 최대한 깎음. (통짜 vs 분할을 같은 척도로 비교 → 상황맞는 입도 자동 선택)
- **컨텍스트 예산 B_split**이 SPLIT 남발을 막아 §5 폭발을 원천 차단. 부하 큰 배치일수록 자연히
  통짜 위주가 됨 = CLAUDE.md "입도는 부하에 따라 뒤집힌다"가 알고리즘으로 구현됨.

> ⚠️ 이건 **휴리스틱**(2-자원 heterogeneous scheduling은 NP-hard). N이 작으면(≤~20) 모드 조합
> 전수/DP도 가능. 우선 그리디로 만들고, LB 대비 gap을 [figA](figures/figA_optimality_gap.svg)로 검증.

---

## 4. 측정·비용표 통합

### 4.1 통합 프로파일
두 CSV를 하나로 합친다(모델별 1행 그룹):
```
profile.csv       : model, C_whole, G_whole, gpu_ok         (online.cc:278)
split_profile.csv : model, direction, k, cpu_stage, gpu_stage, boundary_kb  (jackson_online.cc:159)
→ unified_profile.csv : model, C_whole, G_whole, gpu_ok,
                        best_dir, best_k, cpu_stage, gpu_stage, boundary_kb
```
- **α 재교정(필수)**: `online.cc`의 α=1.5는 근거 없는 매직넘버. **P개 동시 실행 벤치**로
  실효 C(경합 포함)/단독 C 비율을 모델별 실측해 채운다. → `docs/measurements/`.
- **G_whole 재확인**: [multitask_experiments.md](multitask_experiments.md) §3에서 프로파일 G가
  연속실행 실측보다 부풀려져 오배정 유발. 연속 실행 실측으로 갱신.

---

## 5. 두 코드베이스 병합 지점 (구체)

| 구성요소 | online.cc | jackson_online.cc | 통합 방침 |
|---|---|---|---|
| Resolver | `BuiltinOpResolver`(XNNPACK on) | `...WithoutDefaultDelegates`(off) | **통일 필요**(공정비교). 기본 off로 맞추고 측정 |
| GPU 컨텍스트 | 모델당 공유(online.cc:187) | 인스턴스당(jackson:109) | **모드별 분기**: WHOLE_GPU 공유 / SPLIT 인스턴스별 |
| 분할 지정 | 없음 | `first/last_delegate_node_index`(jackson:81) | SPLIT 빌더에 그대로 사용 |
| 실행 | GPU=메인직렬 + CPU워커풀(online:210,221) | 단일 워커풀(jackson:118) | **GPU 직렬 보장** 필요 → online 방식 채택 |
| 순서 | C/G 정렬 offload(online:150) | Johnson+인터리브(jackson:205) | 플래너가 둘 다 산출(3.2 step4) |
| 타이밍 하네스 | build/warmup 타이머 밖 | 동일 | 공통 유틸로 추출 |

- **공통 추출 대상**: `BaseName`, `FillInputsZero`, warmup 루프, CSV 로더, makespan 타이머 →
  `common.h`로 분리. 두 빌더(`BuildWhole`, `BuildSplit`)는 ExecUnit 팩토리로 통합.
- **GPU 직렬성**: SPLIT의 gpu_stage와 WHOLE_GPU의 Gwhole가 **같은 물리 GPU를 공유**하므로,
  실행기는 "GPU를 쓰는 모든 작업"이 직렬화됨을 보장해야 한다(세마포어 M=1 또는 GPU 전용 스레드 1개).
  현재 jackson의 워커풀은 이걸 보장 안 함 → **통합 실행기는 GPU 워커 1 + CPU 워커 P** 구조.

---

## 6. 안전장치 (측정이 명령한 것)

1. **컨텍스트 예산** `B_split`(기본 ~10) — SPLIT 인스턴스 총수 상한. §5 폭발 방지.
2. **균형 가드** — CPUlane이 GPUlane 넘으면 offload/split 취소·중단. 통짜 참사(0.43x) 방지.
3. **α 경합 보정** — 실측 재교정값. 과잉 offload 방지.
4. **GPU 직렬 보장** — GPU 동시 잡 상한(세마포어). 컨텍스트 스위치 thrashing 방지.

---

## 7. 인터페이스(안)

```
unified_planner --profile unified_profile.csv <model:count> ... \
                --cpu-cores P --split-budget B [--alpha A] [--emit-plan plan.json]
unified_runner  --plan plan.json <model:count> ...     # plan대로 실측
```
- 산출 로그: `[PLAN] per-mode 개수`, `GPUlane/CPUlane 예측`, `[RESULT] makespan 실측 vs LB(gap%)`.
- 기존 fig 파이프라인과 **동일 워크로드 이름**으로 맞춰 [figA](figures/figA_optimality_gap.svg) 재사용.

---

## 8. 미해결 (구현 전 확정)

1. **N 규모**: 대상 배치 크기? ≤20이면 그리디 대신 DP/전수로 최적 보장 가능.
2. **Resolver 통일**: XNNPACK on/off 어느 쪽 기준? (CPU 경쟁력에 직결 — off면 CPU가 더 느려짐)
3. **B_split 기본값**: 측정으로 폭발 문턱 재확인(현재 "~15"는 30개 실측 1점 기반).
4. **α 재교정 측정** 먼저 할지, 임시 1.5로 골격부터 만들지.

---

## 9. 구현 로드맵 (phase)

- **P1 통합 프로파일** — 두 CSV 병합 + α/G_whole 재측정 → `unified_profile.csv`.
- **P2 공통부 추출** — `common.h`(로더/warmup/타이머/ExecUnit 팩토리). 두 기존 도구를 이걸로 재작성해 회귀 확인.
- **P3 플래너** — §3 그리디 + 균형가드 + 컨텍스트예산. plan.json 출력.
- **P4 통합 실행기** — GPU워커1+CPU워커P, plan 실행, makespan 실측.
- **P5 검증** — LB 대비 gap([figA]) + 통짜/스플릿 단독 대비 우열. 워크로드 스윕.
- **P6(후속)** — 온라인 큐잉, 같은모델 배칭, int8 경로.

---

## 10. 정직한 기대치

- **성립하는 이득**: GPU가 병목이고 CPU에 여유가 있는 중간 부하에서 GPU 일부를 CPU로 덜어 makespan↓.
  [scheduler_visualization.md](scheduler_visualization.md) §8 수정후처럼 배분이 옳으면 N=120에서 1.17x까지 관측됨.
- **성립 안 하는 경우**: GPU 압도 모델(mobilenet 등)만의 버스트 — 흘릴 여지 없음 → 통짜 GPU가 최선(≈1.0x).
- **핵심 가치**: "어느 task가 와도" 플래너가 **통짜/분할/CPU를 같은 척도로 저울질**해
  그 배치에서 **LB에 가장 근접한 배분**을 자동 선택. 이득의 크기는 자원(특히 CPU 속도)이 정하지만,
  **주어진 자원을 낭비 없이 쓰는 것**은 이 플래너가 보장.
```
