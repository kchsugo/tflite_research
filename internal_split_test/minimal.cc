// 내부 분할 (Internal Split) 실험 도구
// ------------------------------------------------------------
// 목적: 원본 .tflite 하나를 단일 인터프리터에서 CPU/GPU로 내부 분할.
//   - 사전 파일 분리 없음, 인터프리터 1개, memcpy/zero-copy 코드 없음.
//   - 절단점 k = first_delegate_node_index (노드 0..k-1 CPU, k.. GPU).
//   - k를 밖에서 지정(단일 모드) 하거나, 여러 k를 훑어(sweep 모드) 가장 빠른 k를 고른다.
//
// 사용법:
//   minimal <model.tflite>            → sweep: k를 훑어 최적 절단점 리포트
//   minimal <model.tflite> <k>        → 단일: k에서 배정표 + 경계 + 지연
//   minimal <model.tflite> sweep <s>  → sweep, 간격 s (기본 1)

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#ifndef TFLITE_DEBUG_DELEGATE
#error "TFLITE_DEBUG_DELEGATE not defined - check CMake add_compile_definitions"
#endif

#define CHECK(x)                                             \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// TfLiteRegistration -> op 이름 문자열 (layer_gpu_test 패턴 재사용)
static const char* OpName(const TfLiteRegistration& reg) {
  if (reg.builtin_code == tflite::BuiltinOperator_CUSTOM)
    return reg.custom_name ? reg.custom_name : "CUSTOM";
  const char* n = tflite::EnumNameBuiltinOperator(
      static_cast<tflite::BuiltinOperator>(reg.builtin_code));
  return n ? n : "Unknown";
}

// 절단점 하나(k)의 측정 결과
struct CutResult {
  int    k            = 0;
  int    cpu_nodes    = 0;   // CPU에 남은 노드 수 (execution_plan 기준)
  int    gpu_kernels  = 0;   // GPU delegate 커널 노드 수 (보통 0 또는 1)
  size_t boundary_bytes = 0; // CPU->GPU 경계 활성 텐서 합계
  double latency_us   = 0;   // 추론 지연 평균
  double stddev_us    = 0;   // 표준편차
  bool   ok           = false;
};

// delegate 없이 인터프리터를 만들어 '원본 노드 개수'를 구한다 (sweep 범위 산정용).
static int CountNodes(const tflite::FlatBufferModel& model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&interp) != kTfLiteOk || !interp) return -1;
  return interp->nodes_size();
}

// 절단점 k에서: 새 인터프리터 → GPU delegate(first_delegate_node_index=k)
// → 배정/경계/지연 측정. verbose면 노드별 CPU/GPU 표를 출력.
static CutResult RunAtCut(const tflite::FlatBufferModel& model, int k, bool verbose) {
  CutResult r;
  r.k = k;

  // (1) 인터프리터를 매번 새로 만든다. delegation은 되돌릴 수 없으므로
  //     k마다 깨끗한 인터프리터가 필요하다.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&interp) != kTfLiteOk || !interp) return r;

  // (2) GPU delegate 옵션: 노드 k 이전은 GPU 후보에서 제외 → CPU 유지.
  TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
  opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  opts.first_delegate_node_index = k;             // ★ 핵심 레버
  // last_delegate_node_index 는 기본값(INT_MAX) = 끝까지 GPU 후보
  TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);

  // (3) delegate 적용 = 이 순간 TFLite가 내부에서 [CPU|GPU] 파티션을 만든다.
  if (interp->ModifyGraphWithDelegate(gpu) != kTfLiteOk) {
    TfLiteGpuDelegateV2Delete(gpu);
    return r;
  }
  if (interp->AllocateTensors() != kTfLiteOk) {
    TfLiteGpuDelegateV2Delete(gpu);
    return r;
  }

  // (4) 실제 실행 플랜을 순회하며 노드별 배정을 읽는다.
  //     execution_plan() = 진짜 실행되는 노드들(흡수된 원본 op은 빠지고
  //     delegate 커널 노드로 치환됨). node.delegate != nullptr 이면 GPU.
  if (verbose) {
    printf("\n%-6s | %-28s | %-6s\n", "NodeId", "Op", "Device");
    printf("-------------------------------------------------------\n");
  }
  std::vector<int> delegate_nodes;
  for (int node_id : interp->execution_plan()) {
    const auto* nr = interp->node_and_registration(node_id);
    const TfLiteNode& node = nr->first;
    const TfLiteRegistration& reg = nr->second;
    bool on_gpu = (node.delegate != nullptr);
    if (on_gpu) { r.gpu_kernels++; delegate_nodes.push_back(node_id); }
    else        { r.cpu_nodes++; }
    if (verbose)
      printf("%-6d | %-28s | %-6s\n", node_id, OpName(reg), on_gpu ? "GPU" : "CPU");
  }
  if (verbose)
    printf("-------------------------------------------------------\n");

  // (5) 경계 텐서 크기: delegate 커널의 입력 중 '상수(가중치)가 아닌' 텐서 =
  //     앞 CPU 파티션에서 흘러온 활성값 = CPU->GPU 경계. 그 bytes 합.
  for (int did : delegate_nodes) {
    const TfLiteNode& dn = interp->node_and_registration(did)->first;
    for (int i = 0; i < dn.inputs->size; ++i) {
      int ti = dn.inputs->data[i];
      if (ti < 0) continue;
      const TfLiteTensor* t = interp->tensor(ti);
      if (t->allocation_type != kTfLiteMmapRo)  // Mmap상수(가중치) 제외
        r.boundary_bytes += t->bytes;
    }
  }

  // (6) 입력을 0으로 채운다 (지연 측정용 더미).
  float* in = interp->typed_input_tensor<float>(0);
  if (in) std::memset(in, 0, interp->tensor(interp->inputs()[0])->bytes);

  // (7) warmup 후 다회 측정 (GPU는 첫 몇 회 셰이더 컴파일로 느림 → 반드시 warmup).
  const int WARMUP = 10, RUNS = 30;
  for (int i = 0; i < WARMUP; ++i)
    if (interp->Invoke() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return r; }

  std::vector<double> samples;
  samples.reserve(RUNS);
  for (int i = 0; i < RUNS; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    interp->Invoke();
    auto t1 = std::chrono::high_resolution_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  double mean = 0;
  for (double s : samples) mean += s;
  mean /= RUNS;
  double var = 0;
  for (double s : samples) var += (s - mean) * (s - mean);
  r.latency_us = mean;
  r.stddev_us  = std::sqrt(var / RUNS);
  r.ok = true;

  // (8) 정리: 인터프리터 먼저 파괴 후 delegate 삭제 (순서 중요).
  interp.reset();
  TfLiteGpuDelegateV2Delete(gpu);
  return r;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <model.tflite>            (sweep all cut points)\n"
            "  %s <model.tflite> <k>        (single cut at k)\n"
            "  %s <model.tflite> sweep <s>  (sweep with step s)\n",
            argv[0], argv[0], argv[0]);
    return 1;
  }
  const char* model_path = argv[1];

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  CHECK(model != nullptr);

  int total_nodes = CountNodes(*model);
  CHECK(total_nodes > 0);
  printf("[MODEL] %s\n", model_path);
  printf("[MODEL] total nodes = %d\n", total_nodes);

  // ---- 단일 모드: minimal model.tflite <k> ----
  if (argc >= 3 && std::strcmp(argv[2], "sweep") != 0) {
    int k = std::atoi(argv[2]);
    printf("[MODE ] single cut, k=%d\n", k);
    CutResult r = RunAtCut(*model, k, /*verbose=*/true);
    CHECK(r.ok);
    printf("\n== Result (k=%d) ==\n", r.k);
    printf("  CPU nodes      : %d\n", r.cpu_nodes);
    printf("  GPU kernels    : %d\n", r.gpu_kernels);
    printf("  Boundary bytes : %zu (%.1f KB)\n",
           r.boundary_bytes, r.boundary_bytes / 1024.0);
    printf("  Latency        : %.2f us (%.3f ms) +/- %.2f us\n",
           r.latency_us, r.latency_us / 1000.0, r.stddev_us);
    return 0;
  }

  // ---- sweep 모드: minimal model.tflite [sweep [s]] ----
  int step = 1;
  if (argc >= 4) step = std::atoi(argv[3]);
  if (step < 1) step = 1;
  printf("[MODE ] sweep, step=%d\n\n", step);

  printf("%-5s | %-9s | %-9s | %-14s | %-16s\n",
         "k", "CPU nodes", "GPU krnl", "boundary(KB)", "latency(ms)");
  printf("--------------------------------------------------------------------\n");

  CutResult best;
  best.latency_us = 1e18;
  // k=0(전부 GPU) 부터 k=total_nodes(전부 CPU) 까지 훑는다.
  for (int k = 0; k <= total_nodes; k += step) {
    CutResult r = RunAtCut(*model, k, /*verbose=*/false);
    if (!r.ok) {
      printf("%-5d | (failed)\n", k);
      continue;
    }
    printf("%-5d | %-9d | %-9d | %-14.1f | %-8.3f (+/-%.3f)\n",
           r.k, r.cpu_nodes, r.gpu_kernels,
           r.boundary_bytes / 1024.0,
           r.latency_us / 1000.0, r.stddev_us / 1000.0);
    if (r.latency_us < best.latency_us) best = r;
  }
  printf("--------------------------------------------------------------------\n");
  printf("\n>>> BEST cut k=%d : latency %.3f ms, boundary %.1f KB, CPU %d / GPU-krnl %d\n",
         best.k, best.latency_us / 1000.0, best.boundary_bytes / 1024.0,
         best.cpu_nodes, best.gpu_kernels);
  return 0;
}
