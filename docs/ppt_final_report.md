# TFLite CPU/GPU 분할·스케줄링 — 최종 정리 (PPT용)

> 환경: **RTX 3060 Laptop GPU(6GB, OpenCL) + i5-11400H(6코어)** / TFLite **fp32** / Windows
> 도구: `profile`·`online`(통짜) · `split_profiler`·`jackson_online`(1컷 split) · `visualize`/SVG
> 그림 파일: `docs/figures/*.svg` (VSCode 미리보기·브라우저 렌더). 이 문서 = 슬라이드 대본.

---

## 슬라이드 1 — 문제 정의

- TFLite GPU delegate는 **"지원되면 최대한 GPU로 뭉쳐 보냄"** — op별 "어디가 빠른가(효율)"는 **안 봄.**
- 다중 task면 GPU로 몰려 **직렬 대기(pile-up)**, CPU 코어는 논다.
- 목표: **모델을 CPU/GPU로 나눠** 두 자원을 같이 쓰고, **다중 task를 효율적으로 스케줄.**

## 슬라이드 2 — 두 가지 접근

| 접근 | 방식 |
|---|---|
| **통짜 라우팅 (multitask)** | task를 **안 쪼개고** 통째로 CPU 또는 GPU에 배정 |
| **1컷 분할 (split, jackson)** | 각 task를 **1번 잘라** 앞뒤를 CPU/GPU로 (파이프라인) |

- 핵심 통찰: **입도는 부하에 따라 뒤집힌다.** 한가하면 쪼개서 겹치고, 몰리면 통짜가 유리.
- discrete GPU(별도 VRAM) → 쪼개면 **경계 전송(PCIe)** 비용 발생.

## 슬라이드 3 — 구축한 파이프라인(도구)

```
[오프라인]  profile / split_profiler  → 모델별 비용표(C,G / 방향별 stage,cut,경계)
[온라인]    online(통짜) / jackson_online(split) → 실제 동시 실행 + makespan 측정
[분석]      SVG 시각화 12종
```
- **측정 규범**: warmup 후 다회, XNNPACK off(분할 노드 보존), GPU는 비동기라 **벽시계−CPUstage**로 GPU 시간 산출.

---

# A. 1컷 분할(split) 접근

## 슬라이드 4 — 어디서 잘랐나 (절단 지점)

![Fig1](figures/fig1_cuts.svg)

- 각 모델 두 방향(CPU-first/GPU-first)의 **균형 cut**. 파랑=CPU노드, 빨강=GPU노드.
- **대부분 GPU 쪽이 큼** — CPU가 4~100배 느려 균형점이 "CPU엔 조금만".

## 슬라이드 5 — 방향별 stage 시간

![Fig2](figures/fig2_stages.svg)

- 각 (모델×방향)의 cpu_stage / gpu_stage 실측 = 스케줄러(Jackson) 입력.
- CPU-stage가 0에 가깝거나 GPU-stage가 큼 → CPU 몫이 작다.

## 슬라이드 6 — 스케줄링 = Jackson's rule

- 각 task = **2단계(CPU-part, GPU-part)** → 2기계 job shop.
- **Johnson's rule** ×2로 두 큐 정렬 (논문: Johnson 1954, Jackson 1956):
  - CPU 큐 = CPU서 시작(CG) 먼저 → GPU서 넘어온 것(GC) 나중
  - GPU 큐 = GC 먼저 → CG 나중
- **각 기계는 "자기서 시작하는 것" 먼저** → 의존성 stall 최소화.

![Fig3](figures/fig3_gantt30.svg)

- 30-task 예측 타임라인. **GPU 레인 꽉 참(병목), CPU 레인 거의 빔** — CPU 느려 못 채움.

## 슬라이드 7 — split 성능 (실측)

![Fig4](figures/fig4_speedup.svg)

- **혼합(방향 섞임): 1.01~1.12x**(소폭 이득), **단일모델 버스트: 0.90~0.95x**(손해).
- 방향 섞기가 겹침의 열쇠. 다만 대부분 노이즈(±5~7%) 수준.

## 슬라이드 8 — split 스케일 한계

![Fig5](figures/fig5_ineff.svg)

- **인스턴스당 GPU 컨텍스트 1개** → 30개서 **VRAM 초과 + 컨텍스트 폭발** → ~2000배 느림(무효).
- split은 **~15개까지만** 정상.

---

# B. 통짜 라우팅(multitask) 접근

## 슬라이드 9 — 50-task 통짜 배분·성능

![Fig7](figures/fig7_mt50_assign.svg)

![Fig8](figures/fig8_mt50_ms.svg)

- GPU 인터프리터를 **모델당 1개(공유)** → **컨텍스트 폭발 없이 50~120개 정상 실행.**
- 가벼운 mobilenet·yolov4만 CPU로, 무거운 건 GPU. (초기엔) 0.96x.

## 슬라이드 10 — 통짜 vs split 비교

![Fig6](figures/fig6_compare.svg)

- split이 더 안정적(참사 없음), 통짜는 mob6서 **0.43x 참사**. 둘 다 이득 작음.
- **통짜=스케일 O/배분실수 위험, split=안정 O/스케일 X.**

---

# C. "task 많으면 이득?" — 스케일링 실험 & 수정

## 슬라이드 11 — N=50~120: 많아질수록 손해?!

![Fig9](figures/fig9_scaling_ms.svg)

![Fig10](figures/fig10_scaling_speedup.svg)

- 직관과 반대로 **N 커질수록 speedup 하락 → 손해**(1.07x → 0.86x).

## 슬라이드 12 — 원인 분석: GPU 경합 아님, **CPU측 과부하**

![Fig11](figures/fig11_sides.svg)

- makespan = **max(GPU측, CPU측)**. 분해해보니:
  - **GPU측은 offload할수록 줄어듦** → GPU 경합이 원인이 아님.
  - **CPU측이 병목** — 스케줄러가 CPU로 **과잉 오프로딩**(비싼 yolov4까지) + CPU 동시실행 경합.
- 근본원인: 비용표가 **1스레드 isolated C**라 경합을 못 잡아 "CPU 여유"로 착각.

## 슬라이드 13 — 수정: "CPU측이 GPU측 넘으면 offload 취소"

- 규칙: CPU에 싼 것부터 offload하되 **예측 CPU측 > GPU측이면 취소·중단** + CPU 경합계수 α=1.5.

![Fig11b](figures/fig11b_sides_fixed.svg)

![Fig13](figures/fig13_ms_fixed.svg)

![Fig12](figures/fig12_fix_before_after.svg)

| N | 수정 전 | 수정 후 |
|---|---|---|
| 50 | 1.07x | **1.35x** |
| 90 | 1.00x | 1.03x |
| 120 | **0.86x** | **1.17x** |

- **CPU측이 항상 GPU측 아래** → CPU 병목 소멸 → makespan=GPU측≤ALL-GPU.
- **손해 완전 소멸(전부 ≥1.0x), N 커질수록 이득↑** → "task 많으면 이득" 직관이 (제대로 된 배분으로) 실제 성립.

---

## 슬라이드 14 — 종합 결론

| 관점 | 결론 |
|---|---|
| split 절단(Fig1,2) | CPU 느려 균형 cut이 GPU로 치우침, CPU 몫 작음 |
| split 스케줄(Fig3) | Jackson 정상 작동하나 CPU 레인 비어 겹침 적음 |
| split 성능/스케일(Fig4,5) | 혼합 1.0~1.12x, ~15개서 컨텍스트 폭발 |
| 통짜(Fig6,7,8) | 스케일 O, 초기 배분은 참사 위험 |
| 스케일링(Fig9,10) | (수정 전) 많을수록 손해 |
| 원인(Fig11) | GPU 경합 X, **CPU측 과부하** |
| **수정(Fig11b,12,13)** | **offload 취소 규칙 → 손해 소멸, N 커질수록 이득(최대 1.35x)** |

**핵심 메시지 3줄:**
1. fp32 discrete GPU에선 CPU가 4~100배 느려 이득 한계(`1+P/R`)가 낮다.
2. 순진한 스케줄러는 **CPU에 과잉 오프로딩**해서 오히려 손해 → **"CPU측이 GPU측 넘으면 취소"** 규칙으로 손해를 제거하고 이득으로 전환.
3. **통짜(모델당 컨텍스트 공유)가 스케일에 유리**, split은 겹침 이점은 있으나 컨텍스트 폭발로 소규모 한정.

## 슬라이드 15 — 향후

- **CPU 경쟁력**: int8 양자화로 C↓ → 이득 한계 상승(`1+P/R`↑). (원본 모델 필요)
- **split 스케일**: 동시 GPU 컨텍스트 세마포어 상한 → split도 대규모 실행.
- **경합-인지 비용표**: 동시실행 상태의 C 측정으로 α 대체.
- **온라인 재측정**: 열/터보 변동 반영(실측 run간 편차 큼).

---

### 부록 — 그림 인덱스
- Fig1 절단지점 · Fig2 stage시간 · Fig3 Jackson스케줄 · Fig4 split성능 · Fig5 split스케일한계
- Fig6 통짜vs split · Fig7 50통짜배분 · Fig8 50통짜makespan
- Fig9 makespan vs N · Fig10 speedup vs N · Fig11 GPU측vsCPU측(전) · Fig11b(후) · Fig12 수정전후 · Fig13 makespan(후)
