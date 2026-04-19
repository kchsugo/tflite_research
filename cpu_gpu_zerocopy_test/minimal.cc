#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <gpu_part.tflite> <cpu_part.tflite>\n", argv[0]);
        return 1;
    }

    // ========================================
    // 1. 모델 로드
    // ========================================
    auto gpu_model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    auto cpu_model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
    TFLITE_MINIMAL_CHECK(gpu_model && cpu_model);
    printf("[LOAD] GPU model: %s\n", argv[1]);
    printf("[LOAD] CPU model: %s\n", argv[2]);

    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

    // ========================================
    // 2. GPU Interpreter (첫 번째 분할 모델)
    // ========================================
    std::unique_ptr<tflite::Interpreter> gpu_interp;
    tflite::InterpreterBuilder(*gpu_model, resolver)(&gpu_interp);
    TFLITE_MINIMAL_CHECK(gpu_interp);

    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);

    TFLITE_MINIMAL_CHECK(
        gpu_interp->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk);
    TFLITE_MINIMAL_CHECK(gpu_interp->AllocateTensors() == kTfLiteOk);

    // GPU 입력에 더미 데이터
    float* gpu_input = gpu_interp->typed_input_tensor<float>(0);
    if (gpu_input) {
        std::memset(gpu_input, 0,
            gpu_interp->tensor(gpu_interp->inputs()[0])->bytes);
    }

    // ★ 반드시 한 번 Invoke() 해야 output buffer 주소가 확정됨
    //   (delegate가 내부적으로 GPU 실행 → GPU→CPU 복사 → output tensor에 결과 저장)
    TFLITE_MINIMAL_CHECK(gpu_interp->Invoke() == kTfLiteOk);

    // GPU output 텐서 정보
    int gpu_out_idx = gpu_interp->outputs()[0];
    TfLiteTensor* gpu_out_tensor = gpu_interp->tensor(gpu_out_idx);
    void* gpu_out_ptr = gpu_out_tensor->data.raw;
    size_t gpu_out_bytes = gpu_out_tensor->bytes;

    printf("\n[GPU OUT] tensor index : %d\n", gpu_out_idx);
    printf("[GPU OUT] bytes        : %zu\n", gpu_out_bytes);
    printf("[GPU OUT] data ptr     : %p\n", gpu_out_ptr);
    TFLITE_MINIMAL_CHECK(gpu_out_ptr);

    // ========================================
    // 3. CPU Interpreter (두 번째 분할 모델)
    //    ★ 핵심: input을 GPU output 포인터에 직접 매핑
    // ========================================
    std::unique_ptr<tflite::Interpreter> cpu_interp;
    tflite::InterpreterBuilder(*cpu_model, resolver)(&cpu_interp);
    TFLITE_MINIMAL_CHECK(cpu_interp);
    cpu_interp->SetNumThreads(1);

    int cpu_in_idx = cpu_interp->inputs()[0];

    // ★ SetCustomAllocationForTensor: AllocateTensors 전에 호출해야 함
    //   이렇게 하면 CPU interpreter가 자체 메모리 할당하지 않고
    //   gpu_out_ptr을 input으로 직접 사용
    TfLiteCustomAllocation alloc;
    alloc.data = gpu_out_ptr;
    alloc.bytes = gpu_out_bytes;

    TFLITE_MINIMAL_CHECK(
        cpu_interp->SetCustomAllocationForTensor(cpu_in_idx, alloc) == kTfLiteOk);
    TFLITE_MINIMAL_CHECK(cpu_interp->AllocateTensors() == kTfLiteOk);

    // ★ Zero-copy 검증
    void* cpu_in_ptr = cpu_interp->tensor(cpu_in_idx)->data.raw;
    size_t cpu_in_bytes = cpu_interp->tensor(cpu_in_idx)->bytes;

    printf("\n[CPU IN]  tensor index : %d\n", cpu_in_idx);
    printf("[CPU IN]  bytes        : %zu\n", cpu_in_bytes);
    printf("[CPU IN]  data ptr     : %p\n", cpu_in_ptr);

    printf("\n========================================\n");
    if (cpu_in_ptr == gpu_out_ptr) {
        printf("  ✅ ZERO-COPY 성공!\n");
        printf("  CPU input이 GPU output 버퍼를 직접 참조합니다.\n");
        printf("  memcpy 완전 제거됨.\n");
    } else {
        printf("  ⚠️  ZERO-COPY 실패 — 포인터 불일치\n");
        printf("  GPU out: %p, CPU in: %p\n", gpu_out_ptr, cpu_in_ptr);
        printf("  fallback memcpy를 사용합니다.\n");
    }
    printf("========================================\n");

    bool zero_copy = (cpu_in_ptr == gpu_out_ptr);

    // ========================================
    // 4. Warmup
    // ========================================
    for (int w = 0; w < 5; ++w) {
        gpu_interp->Invoke();
        if (!zero_copy)
            std::memcpy(cpu_in_ptr, gpu_out_ptr,
                gpu_out_bytes < cpu_in_bytes ? gpu_out_bytes : cpu_in_bytes);
        cpu_interp->Invoke();
    }

    // ========================================
    // 5. Profiled Pipeline
    // ========================================
    //
    //  [GPU model Invoke()]
    //       ↓  (delegate 내부: GPU VRAM → CPU RAM 복사 — 이건 불가피)
    //  gpu_out_tensor->data.raw 에 결과 존재
    //       ↓  (zero-copy: 포인터 공유, memcpy 없음!)
    //  [CPU model Invoke()]
    //       ↓
    //  최종 결과
    //
    const int RUNS = 20;
    double sum_gpu = 0, sum_cpu = 0, sum_pipe = 0;

    for (int r = 0; r < RUNS; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Step 1: GPU 분할 모델 실행
        TFLITE_MINIMAL_CHECK(gpu_interp->Invoke() == kTfLiteOk);
        // → Invoke() 리턴 시점에 gpu_out_tensor->data.raw에 결과 있음
        //   (delegate가 내부적으로 clFinish + clEnqueueReadBuffer 완료)

        auto t1 = std::chrono::high_resolution_clock::now();

        // Step 2: 데이터 전달
        if (!zero_copy) {
            // fallback: 포인터가 다르면 수동 복사
            std::memcpy(cpu_in_ptr, gpu_out_ptr,
                gpu_out_bytes < cpu_in_bytes ? gpu_out_bytes : cpu_in_bytes);
        }
        // zero_copy 성공 시: 아무것도 안 함 — 같은 메모리!

        // Step 3: CPU 분할 모델 실행
        TFLITE_MINIMAL_CHECK(cpu_interp->Invoke() == kTfLiteOk);

        auto t2 = std::chrono::high_resolution_clock::now();

        double gpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double cpu_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
        sum_gpu += gpu_us;
        sum_cpu += cpu_us;
        sum_pipe += gpu_us + cpu_us;
    }

    // ========================================
    // 6. 결과 출력
    // ========================================
    printf("\n========================================\n");
    printf("  Pipeline Benchmark (%d runs avg)\n", RUNS);
    printf("========================================\n");
    printf("  GPU part:   %10.2f us (%6.2f ms)\n",
           sum_gpu / RUNS, sum_gpu / RUNS / 1000.0);
    printf("  CPU part:   %10.2f us (%6.2f ms)\n",
           sum_cpu / RUNS, sum_cpu / RUNS / 1000.0);
    printf("  Pipeline:   %10.2f us (%6.2f ms)\n",
           sum_pipe / RUNS, sum_pipe / RUNS / 1000.0);
    printf("  Transfer:   %s\n",
           zero_copy ? "ZERO-COPY (0 us)" : "memcpy (included in CPU time)");
    printf("========================================\n");

    // 최종 출력 샘플
    float* output = cpu_interp->typed_output_tensor<float>(0);
    if (output) {
        int n = cpu_interp->tensor(cpu_interp->outputs()[0])->bytes / sizeof(float);
        printf("\nFinal output[0..4]: ");
        for (int i = 0; i < 5 && i < n; ++i) printf("%.6f ", output[i]);
        printf("\n");
    }

    // ========================================
    // 7. 정리
    // ========================================
    gpu_interp.reset();
    cpu_interp.reset();
    TfLiteGpuDelegateV2Delete(gpu_delegate);
    printf("\n=== Done ===\n");
    return 0;
}
