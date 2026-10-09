# 오프라인 프로파일러 (`multitask_test/profile.cc`) — 설명 문서

> 파이프라인 3단계 중 **1단계(오프라인)**. 각 모델을 CPU/GPU로 재서 "비용표(C, G)"를 만든다.
> 이 문서는 코드를 부위별로 설명한다. 직접 보완/수정할 수 있게 "바꾸는 법"도 함께 적었다.
>
> 파이프라인: **profile.cc (오프라인, 여기) → online.cc (온라인 실행, 예정)**

---

## 0. 이 도구가 하는 일 (한 줄)

모델 .tflite 여러 개를 받아, 각각을 **전부 CPU / 전부 GPU**로 여러 번 돌려 평균 실행시간
`C`(CPU), `G`(GPU)를 재고, 화면 표 + `profile.csv`로 저장한다. 온라인 스케줄러가 이 CSV를 읽는다.

핵심 전제:
- **모델 구분은 데이터가 아니라 "어떤 .tflite를 로드했나"로 결정된다.** 더미 입력(0)은 시간 측정용
  채움일 뿐. 모델마다 인터프리터를 따로 만들어 잰다.
- **추론 시간은 입력 픽셀값과 무관**(연산 횟수 동일)하므로 더미 0으로도 시간은 정확하다.

---

## 1. TFLite 추론 6단계 (이 도구의 뼈대)

한 번의 추론은 항상 이 순서다. 프로파일러의 측정 함수들이 이걸 그대로 따른다.

```
① 모델 로드    FlatBufferModel::BuildFromFile("x.tflite")   // 그래프 + 가중치
② 인터프리터   InterpreterBuilder(model, resolver)(&interp) // 실행 준비
③ 텐서 할당    interp->AllocateTensors()                    // 입력/출력/중간 메모리(arena)
④ 입력 채우기  typed_input_tensor<float>(0) 에 데이터 복사   // 여기서 '사진'이 들어감(우린 0)
⑤ 실행        interp->Invoke()                             // 실제 계산 (시간 재는 대상)
⑥ 출력 읽기    typed_output_tensor<float>(0)                // 결과 (우린 안 씀)
```

> 자주 하는 오해: "Invoke가 사진을 받는다" ❌ → **④에서 미리 채운 입력을 ⑤가 계산**한다.
> device(CPU/GPU) 선택도 Invoke가 아니라 **②~③ 사이에서 GPU delegate를 붙이느냐로** 결정된다.

---

## 2. 코드 부위별 설명

### (a) 더미 입력 — `FillInputsZero`
```cpp
static void FillInputsZero(tflite::Interpreter* interp) {
  for (int idx : interp->inputs()) {          // 입력이 여러 개일 수 있어 전부 순회
    TfLiteTensor* t = interp->tensor(idx);
    if (t && t->data.raw) std::memset(t->data.raw, 0, t->bytes);  // 0으로 채움
  }
}
```
- 위 ④단계. 시간만 재므로 실제 이미지 대신 0.
- **바꾸는 법**: 실제 데이터로 재고 싶으면 여기서 이미지 바이트를 복사. (시간은 어차피 거의 같음)

### (b) 측정 루프 — `TimedRun` (warmup + 평균 + 표준편차)
```cpp
for (int i = 0; i < warmup; ++i) interp->Invoke();     // (가열) 첫 느린 회 버림
for (int i = 0; i < runs; ++i) {                       // (본 측정)
  t0 = now(); interp->Invoke(); t1 = now();            // ⑤ 실제 실행 = 측정 대상
  s.push_back(ms(t1 - t0));
}
mean, stddev 계산
```
- **warmup(가열)**: 첫 몇 회는 GPU 셰이더 컴파일·지연 초기화·콜드 캐시로 느리다. 이걸 버리고
  안정 상태만 잰다. GPU는 셰이더 컴파일 탓에 warmup을 더 크게(10) 준다.
- **바꾸는 법**: 측정 안정성이 필요하면 `runs`를 늘려라(`--runs 100`). 노트북은 열/터보로
  값이 흔들리므로(실측에서 CPU가 run마다 2~3배 차이 남) 표준편차를 꼭 같이 본다.

### (c) 전부 CPU — `MeasureCPU`
```cpp
InterpreterBuilder(...).SetNumThreads(1);   // ★1스레드 = 단일코어 기준
builder(&interp);
interp->AllocateTensors();                  // ③
FillInputsZero(interp.get());               // ④
return TimedRun(interp, 5, runs, ...);      // ⑤ 반복
```
- delegate를 안 붙였으므로 CPU에서 실행된다.
- **SetNumThreads(1)의 의미**: 지금 `C`는 **단일 코어** 시간. 온라인에서 CPU를 P코어 병렬로 쓸 때
  "task당 1코어" 모델과 맞다. **바꾸는 법**: task당 멀티스레드를 쓸 거면 여기 스레드 수를 올려
  `C`를 다시 재라(그러면 `C`가 몇 배 빨라져 배분 이득이 커진다).
- 참고: 기본 `BuiltinOpResolver`라 **XNNPACK이 자동 적용**된다(로그의 "XNNPACK delegate for CPU").
  순수 참조 커널로 재고 싶으면 `BuiltinOpResolverWithoutDefaultDelegates`로 바꾼다.

### (d) 전부 GPU — `MeasureGPU`
```cpp
opts = TfLiteGpuDelegateOptionsV2Default();
opts.inference_preference = ...SUSTAINED_SPEED;
gpu = TfLiteGpuDelegateV2Create(&opts);            // (3) delegate 생성 = 도구 준비
if (interp->ModifyGraphWithDelegate(gpu) != kTfLiteOk)   // (4) 그래프에 적용 = GPU에 올림
    { ...delete; return INF; }                     //     실패 시 CPU 전용 처리
interp->AllocateTensors();                         // ③
FillInputsZero(...); TimedRun(...,10,...);          // ④⑤
interp.reset();                                    // ★인터프리터 먼저 파괴
TfLiteGpuDelegateV2Delete(gpu);                    // (5) 그 다음 delegate 해제
```
- **(4) `ModifyGraphWithDelegate`가 핵심**: 이 줄이 실제로 지원 op를 GPU 커널로 치환한다.
  이게 없으면 delegate를 만들어도 안 붙어서 CPU로 돈다.
- **파괴 순서 중요**: 인터프리터 → delegate 순. 어기면 크래시.
- 일부 op(예: `ARG_MAX`)는 GPU 미지원 → 그 op만 CPU로 빠지고 나머지는 GPU(로그에 표시됨).
  완전 실패면 `gpu_ok=false`로 기록(온라인에서 CPU 전용으로 처리).

### (e) 지휘부 — main의 모델 순회
```cpp
for (const auto& path : models) {
  auto model = FlatBufferModel::BuildFromFile(path);   // (6) ① 로드
  C = MeasureCPU(*model, ...);                          // 이 모델의 CPU 시간
  G = MeasureGPU(*model, ...);                          // 이 모델의 GPU 시간
  profs.push_back(...)                                  // 수집
}
// → 표 출력 + CSV 저장
```
- 측정 함수는 "모델 하나"를 재고, 이 순회가 **5개 모델 전부에** 적용한다.

### (f) 출력 — CSV 형식 (온라인 입력)
```
model_path,C_ms,G_ms,gpu_ok,C_over_G
..\models\BiseNet.tflite,2395.7876,23.7906,1,100.7032
...
```
- `online.cc`가 이 파일을 읽어 각 인스턴스의 (C, G)를 조회한다.
- **바꾸는 법**: 열을 추가하고 싶으면(예: 표준편차, 경계 텐서 크기) 여기 `fprintf`에 컬럼 추가.

---

## 3. 실측 결과 (참고, 값은 run마다 변동)

| id | 모델 | C (ms) | G (ms) | C/G |
|---|---|---|---|---|
| 0 | mobilenet_v2 | 9.56 | 3.74 | 2.6 |
| 1 | lanenet | 352.3 | 9.07 | 38.8 |
| 2 | BiseNet | 2395.8 | 23.79 | 100.7 |
| 3 | yolov4-tiny | 249.2 | 8.84 | 28.2 |
| 4 | WideResNet50 | 865.2 | 10.28 | 84.2 |

- GPU가 CPU보다 **2.6 ~ 100배** 빠름 → 단일 task 통짜 CPU 오프로딩의 이득이 작다.
- **CPU 값이 run마다 2~3배 흔들림**(열/터보) → "오프라인 박제"의 한계. 온라인 실시간 재측정 필요성의 근거.

---

## 4. 빌드 & 실행

```cmd
:: "x64 Native Tools Command Prompt for VS 2022" 에서
cd /d c:\Users\chan\tflite_research\multitask_test
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   :: CMakeLists 바뀔 때만
cmake --build build -j                                    :: .cc 고칠 때마다 (바뀐 것만 재컴파일)

build\profile.exe ..\models\mobilenet_v2_full.tflite ..\models\BiseNet.tflite ^
  "..\models\yolov4-tiny-416_ieie (1).tflite" --out profile.csv --runs 30
```

---

## 5. 바꿔볼 만한 것 (실험 아이디어)

- `--runs`를 늘려 변동 줄이기 / 표준편차 확인.
- `MeasureCPU`의 `SetNumThreads`를 올려 멀티스레드 CPU 시간 재기 → 배분 이득이 커지는지.
- `BuiltinOpResolverWithoutDefaultDelegates`로 XNNPACK 끄고 순수 CPU 커널 시간 비교.
- int8 양자화 모델을 넣어 `C`가 얼마나 내려가는지(LaLaRAND식 CPU 경쟁력 확보).
