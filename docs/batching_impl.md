# 배칭 구현 방식 (PPT용)

> 도구: `split_profiler/batch_test.cc` / 환경: RTX 3060 Laptop + TFLite GPU delegate

---

## 1. 핵심 아이디어

같은 모델 입력 N개를 **batch 차원으로 쌓아** Invoke를 1번만 호출.

```
입력 텐서 shape:  [1, H, W, C]   →   [B, H, W, C]
Invoke 호출 수:    N번(하나씩)    →   ceil(N/B)번(묶음)
```

- batch 원소는 서로 **독립** → GPU엔 "출력이 B배 많은 한 커널" → 노는 코어를 채움
- 동시 Invoke(여러 커널 경쟁)와 다름: 배칭은 **1 커널 + 데이터 B배** (충돌 없음)

---

## 2. 구현 단계 (`batch_test.cc`)

```cpp
// (1) 인터프리터 생성
InterpreterBuilder(model, resolver)(&it); it->SetNumThreads(1);

// (2) 입력 텐서의 batch 차원(dim[0])을 B로 교체  ★핵심
int in = it->inputs()[0];
std::vector<int> dims(t->dims->data, t->dims->data + t->dims->size);
dims[0] = B;                              // [1,H,W,C] → [B,H,W,C]
it->ResizeInputTensor(in, dims);          // ← 여기서 배치 적용

// (3) GPU delegate 적용 + 텐서 할당
TfLiteGpuDelegateV2Create(&opt);
it->ModifyGraphWithDelegate(gpu);
it->AllocateTensors();                    // B 반영해 버퍼 재할당

// (4) 더미 입력 + warmup(셰이더 컴파일)
FillInputsZero(it); for(i<3) it->Invoke();

// (5) 측정: Invoke 1번 = B개 처리
t0 = now(); it->Invoke(); total = now()-t0;
per_task = total / B;                     // task 1개당 실효 시간
```

**순서 중요**: `ResizeInputTensor` → `ModifyGraphWithDelegate` → `AllocateTensors`.
(배치 리사이즈를 delegate/할당 전에 해야 B가 그래프에 반영됨)

---

## 3. 실행 흐름 도식

```
   입력 N개 (같은 모델)
        │
        ▼
   ┌─────────────────────────┐
   │ B개씩 묶기 (stack on dim0)│   [B,H,W,C] 텐서 1개
   └───────────┬─────────────┘
               ▼
        Invoke() 1번  ──►  GPU: 커널 1개가 B개 동시 계산 (노는 코어 채움)
               │
               ▼
        출력 B개 분리
        (ceil(N/B)번 반복)
```

---

## 4. 측정으로 확인한 것

- 배치 **되는지**: `ResizeInputTensor` + `AllocateTensors` + `Invoke` 성공 여부로 판정
  (실패 모델 = "미지원"; 무거운 모델은 큰 B에서 **VRAM 초과(OOM)**)
- 배치 **효과**: `per-task = total/B`를 B=1과 비교 → speedup
  - mobilenet 4.26x, yolov4 2.07x, WideResNet 1.99x, lanenet 1.35x, BiseNet 1.08x

---

## 5. 스케줄러 통합 (현재/다음)

**현재**: 배칭은 **독립 측정 도구**(`batch_test.cc`)로 구현·검증 완료. 통합 플래너엔 아직 미반영.

**통합 시 들어갈 곳** (2군데):

| 부분 | 변경 |
|---|---|
| **실행부 (스케줄러)** | GPU 레인에서 **같은 모델 task를 묶어** batched Invoke (`ceil(N/B)`번) ← 구현 핵심 |
| **플래너** | `g_eff`를 **batched per-task**(B 반영)로 사용 → GPU가 빨라진 걸 배정에 반영 |

```
[지금 실행]  GPU 레인: for task in gpu_list: it->Invoke()        (하나씩)
[배칭 실행]  GPU 레인: for (model, tasks) in group_by_model:
                         묶어서 ceil(len/B)번 batched Invoke       (모아서)
```

---

## 6. 한 줄 메시지

> **배칭 구현 = 입력을 `ResizeInputTensor`로 batch 차원에 쌓아 Invoke 1번에 B개 처리.**
> 스케줄러 실행부에서 "같은 모델 GPU task를 묶어 보내는" 그룹핑이 핵심이고,
> 플래너는 batched per-task 값만 반영하면 된다.
