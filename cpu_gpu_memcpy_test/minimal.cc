#include <cstdio>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <cstring>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#define TFLITE_MINIMAL_CHECK(x)                               \
  if (!(x)) {                                                 \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                  \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite CPU model> <tflite GPU model>\n");
    return 1;
  }

  // 1. 모델 로드 (첫 번째: CPU용, 두 번째: GPU용)
  std::unique_ptr<tflite::FlatBufferModel> cpu_model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  std::unique_ptr<tflite::FlatBufferModel> gpu_model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
  TFLITE_MINIMAL_CHECK(cpu_model != nullptr && gpu_model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  
  // 2. 인터프리터 빌드
  std::unique_ptr<tflite::Interpreter> cpu_interpreter;
  std::unique_ptr<tflite::Interpreter> gpu_interpreter;
  
  tflite::InterpreterBuilder(*cpu_model, resolver)(&cpu_interpreter);
  tflite::InterpreterBuilder(*gpu_model, resolver)(&gpu_interpreter);
  TFLITE_MINIMAL_CHECK(cpu_interpreter != nullptr && gpu_interpreter != nullptr);

  // 3. GPU Delegate 설정 (두 번째 인터프리터에 적용)
  TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
  gpu_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_options);
  
  // GPU 모델에 Delegate 적용
  TFLITE_MINIMAL_CHECK(gpu_interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk);

  // 4. 텐서 할당
  TFLITE_MINIMAL_CHECK(cpu_interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_MINIMAL_CHECK(gpu_interpreter->AllocateTensors() == kTfLiteOk);

  printf("\n--- Hybrid Pipeline (CPU -> GPU, Memcpy Mode) ---\n");

  // 5. 실행 및 성능 측정
  auto start_time = std::chrono::high_resolution_clock::now();

  // Step 1: CPU 모델 실행
  TFLITE_MINIMAL_CHECK(cpu_interpreter->Invoke() == kTfLiteOk);

  auto mid_time = std::chrono::high_resolution_clock::now();

  // Step 2: 데이터 복사 (CPU 출력 -> GPU 입력)
  // Zero-copy를 사용하지 않으므로 직접 memcpy 수행
  float* cpu_out = cpu_interpreter->typed_output_tensor<float>(0);
  float* gpu_in = gpu_interpreter->typed_input_tensor<float>(0);
  size_t bytes_to_copy = cpu_interpreter->tensor(cpu_interpreter->outputs()[0])->bytes;
  
  std::memcpy(gpu_in, cpu_out, bytes_to_copy);

  // Step 3: GPU 모델 실행
  TFLITE_MINIMAL_CHECK(gpu_interpreter->Invoke() == kTfLiteOk);

  auto end_time = std::chrono::high_resolution_clock::now();

  // 6. 결과 출력
  auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  auto copy_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time).count(); // 복사 + GPU 실행 포함 (또는 별도 측정 가능)

  printf("Execution Summary:\n");
  printf(" - Total Latency: %ld us (%.2f ms)\n", total_us, total_us / 1000.0);
  printf(" - Transfer (Memcpy): %zu bytes copied\n", bytes_to_copy);

  // 자원 해제
  TfLiteGpuDelegateV2Delete(gpu_delegate);

  return 0;
}