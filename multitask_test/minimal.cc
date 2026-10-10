// ============================================================================
//  다중 task CPU/GPU 통짜 배분기  (Milestone 1)
// ============================================================================
//
//  [이 프로그램이 하는 일 — 큰 그림]
//   "서로 다른 모델 N개가 동시에 도착했다"고 가정하고, 각 task(=모델)를
//   쪼개지 않고 통째로 CPU 또는 GPU 한쪽에 보내는 최적 배정을 찾는다.
//
//   흐름:
//     PART 1) 각 모델을 전부 CPU / 전부 GPU 로 돌려 실행시간 C_i, G_i 를 실측
//     PART 2) baseline 계산 (전부 GPU / 전부 CPU)
//     PART 3) 최적 배정 = 오프라인 + N 작음 → 2^N 전수탐색으로 진짜 최적
//     PART 4) 그리디 트레이스(설명용) = 비율 큰 것부터 GPU로
//     PART 5) 결과 출력 = 세 방식의 makespan 비교 + speedup
//
//  [자원 모델 — 왜 이렇게 계산하나]
//   - GPU = 직렬 단일 자원.  GPU에 배정된 task는 공유 GPU에서 하나씩 순서대로
//           실행되므로,  GPU_finish = Σ G_i  (배정된 것들의 단순 합).
//   - CPU = P코어 병렬 자원.  CPU에 배정된 task들은 서로 다른 코어에서 동시에
//           돌 수 있으므로,  CPU_finish = LPT makespan(배정된 것들, P).
//   - 목표(동시 도착이니 "전부 끝나는 시간"을 최소화):
//           min  max( GPU_finish , CPU_finish )
//
//  [사용법]
//     multitask <m1.tflite> <m2.tflite> ... [--cpu-cores P]
//     (P 생략 시 하드웨어 코어 수를 자동 사용)
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <thread>
#include <limits>

// TFLite 핵심 헤더 (internal_split_test와 동일한 세트)
#include "tensorflow/lite/interpreter.h"          // tflite::Interpreter
#include "tensorflow/lite/interpreter_builder.h"  // tflite::InterpreterBuilder
#include "tensorflow/lite/kernels/register.h"     // BuiltinOpResolver
#include "tensorflow/lite/model_builder.h"        // FlatBufferModel
#include "tensorflow/lite/delegates/gpu/delegate.h" // GPU delegate (V2)

// 조건이 거짓이면 파일:라인 찍고 종료. (internal_split_test의 CHECK와 동일)
#define CHECK(x)                                             \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// "무한대" = 측정 실패나 GPU 불가를 표현하는 sentinel 값.
static const double INF = std::numeric_limits<double>::infinity();

// 한 개의 task(=모델) 하나에 대한 측정 결과를 담는 구조체.
struct Task {
  std::string path;      // 모델 파일 경로
  double C = INF;        // 전부 CPU 로 돌릴 때 실행시간 (ms, 1스레드)
  double G = INF;        // 전부 GPU 로 돌릴 때 실행시간 (ms)
  bool   gpu_ok = false; // GPU delegate 적용에 성공했는가 (실패 시 CPU 전용)
};

// ----------------------------------------------------------------------------
//  [공통 헬퍼] 측정에 쓰는 작은 함수들
// ----------------------------------------------------------------------------

// 인터프리터의 모든 입력 텐서를 0으로 채운다.
//  - 지연만 재면 되므로 실제 데이터는 필요 없다(더미 입력).
//  - internal_split_test는 입력 1개만 memset 했지만, 여기선 여러 모델을 받으므로
//    입력이 여러 개일 수 있어 전부 순회한다.
static void FillInputsZero(tflite::Interpreter* interp) {
  for (int idx : interp->inputs()) {
    TfLiteTensor* t = interp->tensor(idx);
    if (t && t->data.raw) std::memset(t->data.raw, 0, t->bytes);
  }
}

// warmup 후 runs회 Invoke 하여 평균 실행시간(ms)을 반환.
//  - GPU는 첫 몇 회가 셰이더 컴파일로 매우 느리므로 warmup이 필수.
//  - 실패하면 INF 반환.
static double TimedRun(tflite::Interpreter* interp, int warmup, int runs) {
  for (int i = 0; i < warmup; ++i)                 // (1) 예열: 측정에서 제외
    if (interp->Invoke() != kTfLiteOk) return INF;
  double sum = 0;
  for (int i = 0; i < runs; ++i) {                 // (2) 본 측정: runs회 평균
    auto t0 = std::chrono::high_resolution_clock::now();
    if (interp->Invoke() != kTfLiteOk) return INF;
    auto t1 = std::chrono::high_resolution_clock::now();
    sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return sum / runs;
}

// 모델 하나를 "전부 CPU"(delegate 없음, 1스레드)로 돌린 평균 시간 → C_i.
static double MeasureCPU(const tflite::FlatBufferModel& model) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);                        // 1스레드 = 단일 코어 기준선
  if (builder(&interp) != kTfLiteOk || !interp) return INF;
  if (interp->AllocateTensors() != kTfLiteOk) return INF;
  FillInputsZero(interp.get());
  return TimedRun(interp.get(), /*warmup=*/5, /*runs=*/30);
}

// 모델 하나를 "전부 GPU"(GPU delegate 전체 적용)로 돌린 평균 시간 → G_i.
//  - 성공하면 *gpu_ok = true, 실패(미지원 op 등)하면 INF + *gpu_ok = false.
//  - delegate 생명주기 순서(생성 → ModifyGraph → Allocate → ... → 인터프리터
//    먼저 파괴 → delegate 삭제)는 internal_split_test와 동일. 순서 어기면 크래시.
static double MeasureGPU(const tflite::FlatBufferModel& model, bool* gpu_ok) {
  *gpu_ok = false;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&interp) != kTfLiteOk || !interp) return INF;

  // GPU delegate 옵션: 지속 속도(SUSTAINED_SPEED) 선호. 전체 그래프를 GPU 후보로.
  TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
  opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&opts);

  // 이 순간 TFLite가 지원 op를 GPU 커널로 치환한다(통짜 위임).
  if (interp->ModifyGraphWithDelegate(gpu) != kTfLiteOk) {
    TfLiteGpuDelegateV2Delete(gpu);
    return INF;                                    // GPU 위임 실패 → CPU 전용 처리
  }
  if (interp->AllocateTensors() != kTfLiteOk) {
    TfLiteGpuDelegateV2Delete(gpu);
    return INF;
  }
  FillInputsZero(interp.get());
  double ms = TimedRun(interp.get(), /*warmup=*/10, /*runs=*/30);
  interp.reset();                                  // ★ 인터프리터 먼저 파괴
  TfLiteGpuDelegateV2Delete(gpu);                  //   그 다음 delegate 삭제
  *gpu_ok = std::isfinite(ms);
  return ms;
}

// ----------------------------------------------------------------------------
//  [비용함수] 자원 모델을 makespan 숫자로 바꾸는 함수들
// ----------------------------------------------------------------------------

// CPU set을 P코어에 LPT(Longest Processing Time first)로 스케줄한 makespan.
//  - task당 1코어를 쓴다고 가정, 각 task 시간 = C_i (1스레드 측정값).
//  - LPT 규칙: 긴 것부터 "가장 한가한 코어"에 하나씩 얹는다 → 부하 균형 근사 최적.
//  - 반환값 = 가장 늦게 끝나는 코어의 완료시각 = CPU 쪽 전체 완료시간.
static double CpuMakespan(const std::vector<double>& cpu_times, int P) {
  if (cpu_times.empty()) return 0.0;               // CPU에 아무것도 없으면 0
  std::vector<double> t = cpu_times;
  std::sort(t.begin(), t.end(), std::greater<double>());  // 긴 것부터
  std::vector<double> load(P, 0.0);                // 각 코어의 누적 부하
  for (double x : t) {
    int m = 0;                                     // 가장 한가한 코어 찾기
    for (int i = 1; i < P; ++i) if (load[i] < load[m]) m = i;
    load[m] += x;                                  // 그 코어에 task 얹기
  }
  return *std::max_element(load.begin(), load.end());     // 최댓값 = makespan
}

// 하나의 배정(assign[i]==1 → task i는 GPU, 0 → CPU)에 대한 전체 makespan.
//   makespan = max( GPU 직렬합 ΣG , CPU 병렬 makespan )
static double Makespan(const std::vector<Task>& tasks,
                       const std::vector<char>& assign, int P) {
  double gpu = 0.0;                                // GPU는 직렬 → 단순 합산
  std::vector<double> cpu;                         // CPU는 병렬 → 리스트로 모아 LPT
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (assign[i]) gpu += tasks[i].G;
    else           cpu.push_back(tasks[i].C);
  }
  return std::max(gpu, CpuMakespan(cpu, P));
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char* argv[]) {

  // ==========================================================================
  //  PART 0 — 인자 파싱
  //   모델 경로들을 모으고, --cpu-cores P 옵션이 있으면 P를 읽는다.
  //   (P = CPU 병렬 코어 수. 생략 시 하드웨어 코어 수.)
  // ==========================================================================
  std::vector<std::string> models;
  int P = (int)std::thread::hardware_concurrency();
  if (P < 1) P = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--cpu-cores") == 0 && i + 1 < argc) {
      P = std::atoi(argv[++i]);
      if (P < 1) P = 1;
    } else {
      models.push_back(argv[i]);                   // 나머지는 전부 모델 경로
    }
  }
  if (models.empty()) {
    fprintf(stderr,
            "Usage: %s <m1.tflite> <m2.tflite> ... [--cpu-cores P]\n", argv[0]);
    return 1;
  }
  const int N = (int)models.size();
  printf("[CONFIG] tasks=%d, cpu-cores(P)=%d\n\n", N, P);

  // ==========================================================================
  //  PART 1 — 각 모델의 C_i / G_i 실측
  //   모델마다 전부 CPU / 전부 GPU 로 돌려 평균 실행시간을 잰다.
  //   이 두 값이 이후 모든 배분 계산의 입력이다.
  // ==========================================================================
  printf("Measuring per-model CPU(C) / GPU(G) times ...\n");
  std::vector<Task> tasks(N);
  for (int i = 0; i < N; ++i) {
    tasks[i].path = models[i];
    auto model = tflite::FlatBufferModel::BuildFromFile(models[i].c_str());
    if (!model) { fprintf(stderr, "  [%d] load failed: %s\n", i, models[i].c_str()); continue; }
    tasks[i].C = MeasureCPU(*model);                       // C_i
    tasks[i].G = MeasureGPU(*model, &tasks[i].gpu_ok);     // G_i (+ gpu_ok)
  }

  // 측정 결과표: C, G, 그리고 비율 C/G(=GPU 이득 크기).
  printf("\n%-3s | %-28s | %10s | %10s | %8s\n",
         "id", "model", "C (ms)", "G (ms)", "C/G");
  printf("---------------------------------------------------------------------------\n");
  for (int i = 0; i < N; ++i) {
    const Task& t = tasks[i];
    std::string name = t.path;                     // 경로에서 파일명만 잘라 표시
    size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    if (name.size() > 28) name = name.substr(0, 28);
    double ratio = (t.gpu_ok && t.G > 0) ? t.C / t.G : 0.0;
    printf("%-3d | %-28s | %10.3f | %10.3f | %8.2f%s\n",
           i, name.c_str(), t.C, t.G, ratio,
           t.gpu_ok ? "" : "  (GPU X -> CPU only)");
  }
  printf("---------------------------------------------------------------------------\n\n");

  // ==========================================================================
  //  PART 2 — baseline 계산 (전부 GPU / 전부 CPU)
  //   비교 기준선 두 개.
  //    - ALL-GPU = 지금 delegate가 하는 것(전부 GPU) → GPU에 다 쌓임(pile-up)
  //    - ALL-CPU = 전부 CPU
  //   GPU 불가 모델은 baseline에서도 어쩔 수 없이 CPU로 둔다.
  // ==========================================================================
  std::vector<char> all_gpu(N, 1), all_cpu(N, 0);
  for (int i = 0; i < N; ++i) if (!tasks[i].gpu_ok) all_gpu[i] = 0;
  double ms_all_gpu = Makespan(tasks, all_gpu, P);   // = ΣG (직렬)
  double ms_all_cpu = Makespan(tasks, all_cpu, P);   // = LPT(all, P)

  // ==========================================================================
  //  PART 3 — 최적 배정 (전수탐색 2^N)
  //   배정은 오프라인에 한 번 계산하면 되고 N(동시 task 수)이 작으므로,
  //   가능한 모든 CPU/GPU 조합(2^N개)을 다 돌려 makespan 최소를 고른다 = 진짜 최적.
  //   각 조합에서 GPU 불가 모델을 GPU로 넣는 건 불가능하므로 건너뛴다.
  // ==========================================================================
  std::vector<char> best(N, 0);
  double best_ms = INF;
  if (N <= 20) {                                   // 2^20 까지만 전수(안전장치)
    for (unsigned long long mask = 0; mask < (1ULL << N); ++mask) {
      std::vector<char> a(N, 0);
      bool feasible = true;
      for (int i = 0; i < N; ++i) {
        bool g = (mask >> i) & 1ULL;               // mask의 i번째 비트 = task i를 GPU로?
        if (g && !tasks[i].gpu_ok) { feasible = false; break; }  // 불가능 조합 제외
        a[i] = g ? 1 : 0;
      }
      if (!feasible) continue;
      double ms = Makespan(tasks, a, P);
      if (ms < best_ms) { best_ms = ms; best = a; } // 최소 makespan 갱신
    }
  }

  // ==========================================================================
  //  PART 4 — 그리디 트레이스 (설명/PPT용)
  //   최적해와 별개로, "비율 큰 것부터 GPU로 옮기며 makespan이 줄면 채택"하는
  //   그리디의 단계별 진행을 보여준다. (슬라이드의 예시 표와 같은 형태)
  //     - GPU_finish 는 옮길수록 커지고, CPU_finish 는 작아진다.
  //     - makespan 이 더 이상 안 줄어드는(=GPU가 병목이 되는) 순간 정지.
  // ==========================================================================
  printf("Greedy trace (move highest C/G to GPU while makespan improves):\n");
  {
    // GPU 가능한 task만 후보로, C/G 내림차순 정렬(GPU 이득 큰 순).
    std::vector<int> order;
    for (int i = 0; i < N; ++i) if (tasks[i].gpu_ok) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return (tasks[a].C / tasks[a].G) > (tasks[b].C / tasks[b].G);
    });
    std::vector<char> a(N, 0);                      // 시작: 전부 CPU
    double cur = Makespan(tasks, a, P);
    printf("  start (all CPU): makespan = %.3f ms\n", cur);
    for (int id : order) {
      std::vector<char> trial = a;
      trial[id] = 1;                               // 이 task를 GPU로 시험 이동
      double ms = Makespan(tasks, trial, P);
      printf("  + move task %d to GPU: makespan = %.3f ms%s\n",
             id, ms, (ms < cur) ? "" : "   (worse -> stop)");
      if (ms < cur) { a = trial; cur = ms; }       // 개선되면 채택
      else break;                                  // 악화되면 정지(교차점)
    }
  }
  printf("\n");

  // ==========================================================================
  //  PART 5 — 결과 출력 (세 방식 비교 + speedup)
  // ==========================================================================
  // 배정 하나를 "makespan + GPU/CPU에 들어간 task 목록"으로 예쁘게 출력하는 헬퍼.
  auto print_plan = [&](const char* label, const std::vector<char>& a, double ms) {
    printf("[%s] makespan = %.3f ms\n", label, ms);
    printf("    GPU: ");
    for (int i = 0; i < N; ++i) if (a[i]) printf("%d ", i);
    printf("\n    CPU: ");
    for (int i = 0; i < N; ++i) if (!a[i]) printf("%d ", i);
    printf("\n");
  };

  printf("================ RESULT ================\n");
  print_plan("ALL-GPU (current delegate)", all_gpu, ms_all_gpu);
  print_plan("ALL-CPU", all_cpu, ms_all_cpu);
  if (N <= 20) {
    print_plan("SCHEDULED (optimal split)", best, best_ms);
    // speedup = (두 baseline 중 더 나은 것) / (우리 최적 배정)
    double base = std::min(ms_all_gpu, ms_all_cpu);
    printf("\nSpeedup vs best baseline: %.2fx  (%.3f -> %.3f ms)\n",
           base / best_ms, base, best_ms);
  } else {
    printf("[SCHEDULED] N>20: brute force skipped (add greedy/DP for large N)\n");
  }
  printf("=======================================\n");
  return 0;
}
