// ============================================================================
//  오프라인 프로파일러 (offline profiler)
// ============================================================================
//
//  [역할] 온라인 스케줄러가 쓸 "비용표(cost table)"를 만든다.
//    - 입력: 모델 .tflite 경로들
//    - 각 모델을 전부 CPU / 전부 GPU 로 여러 번 돌려 평균 실행시간(C, G) 측정
//    - 화면에 표로 출력 + CSV 파일로 저장 (온라인 단계가 이 파일을 읽어 배분)
//
//  [중요] 모델 구분은 '데이터'가 아니라 '어떤 .tflite를 로드했나'로 결정된다.
//    더미 입력(0)은 시간 측정용 채움일 뿐. 모델별로 인터프리터를 따로 만들어 잰다.
//    추론 시간은 입력 픽셀값과 무관(연산 횟수 동일)하므로 더미 0으로도 시간은 정확.
//
//  [사용법]
//    profile <m1.tflite> <m2.tflite> ... [--out profile.csv] [--runs 30]
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <limits>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

static const double INF = std::numeric_limits<double>::infinity();

// 한 모델의 프로파일 결과.
struct Prof {
  std::string path;
  double C = INF, C_std = 0;   // CPU 평균/표준편차 (ms)
  double G = INF, G_std = 0;   // GPU 평균/표준편차 (ms)
  bool   gpu_ok = false;
};

// 더미 입력 채우기: 모든 입력 텐서를 0으로. (모델 정체성과 무관, 시간 측정용)
static void FillInputsZero(tflite::Interpreter* interp) {
  for (int idx : interp->inputs()) {
    TfLiteTensor* t = interp->tensor(idx);
    if (t && t->data.raw) std::memset(t->data.raw, 0, t->bytes);
  }
}

// warmup(가열) 후 runs회 Invoke → 평균(ms)과 표준편차(ms).
static double TimedRun(tflite::Interpreter* interp, int warmup, int runs, double* std_out) {
  for (int i = 0; i < warmup; ++i)                 // (가열) 첫 느린 회들 버림
    if (interp->Invoke() != kTfLiteOk) { *std_out = 0; return INF; }
  std::vector<double> s; s.reserve(runs);
  for (int i = 0; i < runs; ++i) {                 // (본 측정)
    auto t0 = std::chrono::high_resolution_clock::now();
    if (interp->Invoke() != kTfLiteOk) { *std_out = 0; return INF; }   // (1) 실제 실행
    auto t1 = std::chrono::high_resolution_clock::now();
    s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  double mean = 0; for (double x : s) mean += x; mean /= runs;
  double var = 0;  for (double x : s) var += (x - mean) * (x - mean); var /= runs;
  *std_out = std::sqrt(var);
  return mean;
}

// ── 전부 CPU 로 측정 (delegate 없음, 1스레드) ──────────────
static double MeasureCPU(const tflite::FlatBufferModel& model, int runs, double* std_out) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&interp) != kTfLiteOk || !interp) return INF;
  if (interp->AllocateTensors() != kTfLiteOk) return INF;   // (2) 텐서 메모리 확보
  FillInputsZero(interp.get());
  return TimedRun(interp.get(), /*warmup=*/5, runs, std_out);
}

// ── 전부 GPU 로 측정 (GPU delegate 전체 적용) ──────────────
static double MeasureGPU(const tflite::FlatBufferModel& model, int runs,
                         double* std_out, bool* gpu_ok) {
  *gpu_ok = false;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&interp) != kTfLiteOk || !interp) return INF;

  TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
  opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);          // (3) delegate 생성
  if (interp->ModifyGraphWithDelegate(gpu) != kTfLiteOk) {         // (4) 그래프에 적용
    TfLiteGpuDelegateV2Delete(gpu);
    return INF;                                                   // 위임 실패 → CPU 전용
  }
  if (interp->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return INF; }
  FillInputsZero(interp.get());
  double ms = TimedRun(interp.get(), /*warmup=*/10, runs, std_out);
  interp.reset();                                                 // 인터프리터 먼저 파괴
  TfLiteGpuDelegateV2Delete(gpu);                                 // (5) delegate 해제
  *gpu_ok = std::isfinite(ms);
  return ms;
}

// 경로에서 파일명만 잘라냄.
static std::string BaseName(const std::string& p) {
  size_t s = p.find_last_of("/\\");
  return (s == std::string::npos) ? p : p.substr(s + 1);
}

int main(int argc, char* argv[]) {
  // ---- 인자 파싱 ----
  std::vector<std::string> models;
  std::string out_path = "profile.csv";
  int runs = 30;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)       out_path = argv[++i];
    else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = std::atoi(argv[++i]);
    else models.push_back(argv[i]);
  }
  if (models.empty()) {
    fprintf(stderr, "Usage: %s <m1.tflite> ... [--out profile.csv] [--runs 30]\n", argv[0]);
    return 1;
  }
  printf("[PROFILE] models=%zu, runs=%d, out=%s\n\n", models.size(), runs, out_path.c_str());

  // ---- 각 모델 순회: 로드 → C 측정 → G 측정 → 수집 ----
  std::vector<Prof> profs;
  for (const auto& path : models) {
    Prof pr; pr.path = path;
    auto model = tflite::FlatBufferModel::BuildFromFile(path.c_str());  // (6) 파일 로드
    if (!model) { fprintf(stderr, "  load failed: %s\n", path.c_str()); profs.push_back(pr); continue; }
    printf("  measuring %s ...\n", BaseName(path).c_str());
    pr.C = MeasureCPU(*model, runs, &pr.C_std);
    pr.G = MeasureGPU(*model, runs, &pr.G_std, &pr.gpu_ok);
    profs.push_back(pr);
  }

  // ---- 화면 출력 ----
  printf("\n%-3s | %-28s | %14s | %14s | %8s\n", "id", "model", "C (ms)", "G (ms)", "C/G");
  printf("-------------------------------------------------------------------------------\n");
  for (size_t i = 0; i < profs.size(); ++i) {
    const Prof& p = profs[i];
    double ratio = (p.gpu_ok && p.G > 0) ? p.C / p.G : 0.0;
    printf("%-3zu | %-28s | %8.2f +-%-3.1f | %8.2f +-%-3.1f | %8.2f%s\n",
           i, BaseName(p.path).substr(0, 28).c_str(),
           p.C, p.C_std, p.G, p.G_std, ratio,
           p.gpu_ok ? "" : "  (GPU X)");
  }
  printf("-------------------------------------------------------------------------------\n");

  // ---- CSV 저장 (온라인 스케줄러가 읽을 비용표) ----
  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
  std::fprintf(f, "model_path,C_ms,G_ms,gpu_ok,C_over_G\n");
  for (const auto& p : profs) {
    double ratio = (p.gpu_ok && p.G > 0) ? p.C / p.G : 0.0;
    std::fprintf(f, "%s,%.4f,%.4f,%d,%.4f\n",
                 p.path.c_str(), p.C, p.G, p.gpu_ok ? 1 : 0, ratio);
  }
  std::fclose(f);
  printf("\n[SAVED] cost table -> %s\n", out_path.c_str());
  return 0;
}
