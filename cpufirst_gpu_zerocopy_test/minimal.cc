#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define TFLITE_MINIMAL_CHECK(x)                                  \
  if (!(x)) {                                                    \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <cpu_part.tflite> <gpu_part.tflite>\n", argv[0]);
        return 1;
    }

    // 1. 모델 로드 (순서 변경: CPU -> GPU)
    std::unique_ptr<tflite::FlatBufferModel> cpu_model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    std::unique_ptr<tflite::FlatBufferModel> gpu_model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
    TFLITE_MINIMAL_CHECK(cpu_model && gpu_model);

    printf("[LOAD] 1st Part (CPU): %s\n", argv[1]);
    printf("[LOAD] 2nd Part (GPU): %s\n", argv[2]);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    

    std::unique_ptr<tflite::Interpreter> cpu_interpreter;
    tflite::InterpreterBuilder(*cpu_model, resolver)(&cpu_interpreter);
    TFLITE_MINIMAL_CHECK(cpu_interpreter);
    cpu_interpreter->SetNumThreads(1);
    TFLITE_MINIMAL_CHECK(cpu_interpreter->AllocateTensors() == kTfLiteOk);

    // [데이터 입력] 검증을 위해 1, 2, 3... 패턴 입력
    float* cpu_in = cpu_interpreter->typed_input_tensor<float>(0);
    int cpu_in_elems = cpu_interpreter->tensor(cpu_interpreter->inputs()[0])->bytes / sizeof(float);
    for (int i = 0; i < cpu_in_elems; i++) cpu_in[i] = (float)i + 1.0f;

    // CPU 실행 (먼저 실행하여 출력 버퍼 확정)
    TFLITE_MINIMAL_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);

    // CPU 출력 정보 획득
    int cpu_out_idx = cpu_interpreter->outputs()[0];
    TfLiteTensor* cpu_out_tensor = cpu_interpreter->tensor(cpu_out_idx);
    void* cpu_out_ptr = cpu_out_tensor->data.raw;
    size_t cpu_out_bytes = cpu_out_tensor->bytes;

    printf("\n[CPU OUT] index: %d, ptr: %p, size: %zu bytes\n", cpu_out_idx, cpu_out_ptr, cpu_out_bytes);

    // ==============================
    std::unique_ptr<tflite::Interpreter> gpu_interpreter;
    tflite::InterpreterBuilder(*gpu_model, resolver)(&gpu_interpreter);
    TFLITE_MINIMAL_CHECK(gpu_interpreter);

    // GPU Delegate 설정
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
    TFLITE_MINIMAL_CHECK(gpu_interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk);

  
    // [핵심] GPU 입력에 CPU 출력 주소 매핑
    int gpu_in_idx = gpu_interpreter->inputs()[0];
    TfLiteCustomAllocation gpu_input_alloc;
    gpu_input_alloc.data = cpu_out_ptr;
    gpu_input_alloc.bytes = cpu_out_bytes;

    TFLITE_MINIMAL_CHECK(
        gpu_interpreter->SetCustomAllocationForTensor(gpu_in_idx, gpu_input_alloc) == kTfLiteOk);
    TFLITE_MINIMAL_CHECK(gpu_interpreter->AllocateTensors() == kTfLiteOk);

    // Zero-copy 검증
    void* gpu_in_ptr = gpu_interpreter->tensor(gpu_in_idx)->data.raw;
    bool is_zero_copy = (gpu_in_ptr == cpu_out_ptr);
    printf("[GPU IN ] index: %d, ptr: %p\n", gpu_in_idx, gpu_in_ptr);
    printf("========================================\n");
    printf("  결과: %s\n", is_zero_copy ? "✅ ZERO-COPY 성공!" : "⚠️ ZERO-COPY 실패 (자동 복사 모드)");
    printf("========================================\n");

 
    const int RUNS = 10;
    double sum_total = 0;

    for (int r = 0; r < RUNS; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Step 1: CPU 연산
        TFLITE_MINIMAL_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);
        
        // Step 2: 만약 Zero-copy 실패 시 수동 복사 (fallback)
        if (!is_zero_copy) {
            std::memcpy(gpu_in_ptr, cpu_out_ptr, cpu_out_bytes);
        }

        // Step 3: GPU 연산
        TFLITE_MINIMAL_CHECK(gpu_interpreter->Invoke() == kTfLiteOk);

        auto t1 = std::chrono::high_resolution_clock::now();
        sum_total += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    // 최종 결과 출력 (데이터가 잘 전달되었는지 확인)
    float* final_out = gpu_interpreter->typed_output_tensor<float>(0);
    if (final_out) {
        printf("\n[FINAL RESULT] 상위 5개 데이터:\n");
        for (int i = 0; i < 5; i++) printf("%.4f ", final_out[i]);
        printf("\n");
    }

    printf("\nAvg Pipeline Latency: %.2f ms\n", (sum_total / RUNS) / 1000.0);

    // 5. 자원 해제
    gpu_interpreter.reset();
    cpu_interpreter.reset();
    TfLiteGpuDelegateV2Delete(gpu_delegate);
    printf("\n=== Done ===\n");

    return 0;
}
