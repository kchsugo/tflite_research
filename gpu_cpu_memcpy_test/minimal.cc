/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <cstdlib>
#include <memory>

////////////////////////////////////////////////////////////////////////////
#include <chrono>   //추가
#include <cstring>  //추가


#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"
////////////////////////////////////////////////////////////////////////////
#include "tensorflow/lite/delegates/gpu/delegate.h" // GPU delegate 추가


// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    ////////////////////////////////////////////////////////////////////////////
    fprintf(stderr, "minimal <tflite first model> <tflite second model>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* second_filename = argv[2];
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  std::unique_ptr<tflite::FlatBufferModel> second_model =
      tflite::FlatBufferModel::BuildFromFile(second_filename);

  TFLITE_MINIMAL_CHECK(model != nullptr && second_model != nullptr);
  
  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  tflite::InterpreterBuilder second_builder(*second_model, resolver);
  std::unique_ptr<tflite::Interpreter> first_interpreter;
  std::unique_ptr<tflite::Interpreter> second_interpreter;
  builder(&first_interpreter);
  second_builder(&second_interpreter);

  TFLITE_MINIMAL_CHECK(first_interpreter != nullptr && second_interpreter != nullptr);


  ///////////////////////////////////////////////////////////////////////////
  //first interpreter에 GPU delegate 적용
  TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
  gpu_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  gpu_options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  gpu_options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  gpu_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
  gpu_options.max_delegated_partitions = 1; // GPU delegate가 최대 1개의 partition을 위임하도록 설정
  TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_options);
  TFLITE_MINIMAL_CHECK(first_interpreter->ModifyGraphWithDelegate(gpu_delegate) == kTfLiteOk);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK( first_interpreter->AllocateTensors() == kTfLiteOk);
  TFLITE_MINIMAL_CHECK( second_interpreter->AllocateTensors() == kTfLiteOk);
  printf("\n--- Hybrid Pipeline Initialized (RTX 5060 & CPU) ---\n");
  //정밀한 성능 측정을 위해 chrono 라이브러리를 사용하여 GPU delegate가 적용된 첫 번째 인터프리터의 성능을 측정
  auto total_time = std::chrono::high_resolution_clock::now();
  TFLITE_MINIMAL_CHECK(first_interpreter->Invoke() == kTfLiteOk);
  auto copy_time = std::chrono::high_resolution_clock::now();

  float* out_ptt = first_interpreter->typed_output_tensor<float>(0);
  float* in_ptt = second_interpreter->typed_input_tensor<float>(0);
  size_t num_elements = first_interpreter->tensor(first_interpreter->outputs()[0])->bytes;
  std::memcpy(in_ptt, out_ptt, num_elements);
  TFLITE_MINIMAL_CHECK(second_interpreter->Invoke() == kTfLiteOk);
  auto total_time_end = std::chrono::high_resolution_clock::now();
  auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_time_end - total_time).count();
  auto copy_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_time - total_time).count();
  //printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  printf("Execution Summary:\n");
  printf(" - Total Latency: %ld us (%.2f ms)\n", total_us, total_us / 1000.0);
  printf(" - Memcpy Latency: %ld us (%.2f ms)\n", copy_us, copy_us / 1000.0);
  printf(" - Data Copy Ratio: %.2f%%\n", (double)copy_us / total_us * 100.0);
  printf("=== GPU Interpreter State ===\n");
  tflite::PrintInterpreterState(first_interpreter.get());

  printf("=== CPU Interpreter State ===\n");
  tflite::PrintInterpreterState(second_interpreter.get());

  TfLiteGpuDelegateV2Delete(gpu_delegate); // GPU delegate 해제
  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  return 0;
}
