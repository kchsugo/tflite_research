// ============================================================================
//  온라인 실행기 (online executor)  — 파이프라인 3단계 중 마지막
// ============================================================================
//
//  [역할] 오프라인 비용표(profile.csv)를 읽고, 실제 도착 워크로드를 받아
//    1) 배분(어느 인스턴스를 CPU/GPU로) 결정,
//    2) 배분대로 ★실제로 동시 실행★해서 벽시계 makespan을 측정,
//    3) 예측 makespan(산수 모델) vs 실측 makespan을 비교한다.
//
//  [실행 모델]
//    - GPU = 직렬 단일 자원 → '메인 스레드'가 GPU 인스턴스들을 하나씩 Invoke.
//      (GPU 컨텍스트는 만든 스레드에서 쓰는 게 안전하므로 메인에 고정)
//    - CPU = P코어 병렬 → 'P개 워커 스레드'가 공유 큐에서 하나씩 집어 Invoke.
//      (인터프리터는 스레드 안전이 아니므로 CPU 인스턴스마다 자기 인터프리터)
//    - 둘을 동시에 돌리고, 전부 끝나는 시각 = makespan.
//
//  [준비/측정 경계] 인터프리터 생성 + warmup 은 타이머 '밖'(일회성 셋업 제외).
//    측정 구간은 순수 Invoke 동시 실행 구간만.
//
//  [사용법]
//    online --profile profile.csv <model.tflite:count> ... [--cpu-cores P]
//    예) online --profile profile.csv ..\models\BiseNet.tflite:5 ..\models\mobilenet_v2_full.tflite:6
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <limits>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

static const double INF = std::numeric_limits<double>::infinity();
enum Device { CPU = 0, GPU = 1 };

// 워크로드에 등장하는 '모델 종류' 하나. (같은 모델의 여러 인스턴스가 이걸 공유)
struct ModelInfo {
  std::string path, name;
  double C = INF, G = INF;   // 비용표에서 읽은 값 (ms)
  bool   gpu_ok = false;
  std::unique_ptr<tflite::FlatBufferModel> fb;   // 로드된 모델(인터프리터 생성용)
};

// 실제 도착한 잡 하나(인스턴스). model = ModelInfo 인덱스.
struct Instance { int model; };

// ---------------------------------------------------------------------------
//  공통 헬퍼
// ---------------------------------------------------------------------------
static std::string BaseName(const std::string& p) {
  size_t s = p.find_last_of("/\\");
  return (s == std::string::npos) ? p : p.substr(s + 1);
}
static void FillInputsZero(tflite::Interpreter* it) {
  for (int idx : it->inputs()) {
    TfLiteTensor* t = it->tensor(idx);
    if (t && t->data.raw) std::memset(t->data.raw, 0, t->bytes);
  }
}

// CPU 인터프리터 생성 (delegate 없음, 1스레드) + 할당 + 더미입력 + warmup.
static std::unique_ptr<tflite::Interpreter> BuildCPU(const tflite::FlatBufferModel& m) {
  tflite::ops::builtin::BuiltinOpResolver r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  if (it->AllocateTensors() != kTfLiteOk) return nullptr;
  FillInputsZero(it.get());
  for (int i = 0; i < 2; ++i) it->Invoke();   // warmup
  return it;
}

// GPU 인터프리터 생성 (delegate 적용) + 할당 + 더미입력 + warmup.
// delegate 포인터는 out_del로 돌려받아 나중에 삭제해야 함.
static std::unique_ptr<tflite::Interpreter> BuildGPU(const tflite::FlatBufferModel& m,
                                                     TfLiteDelegate** out_del) {
  *out_del = nullptr;
  tflite::ops::builtin::BuiltinOpResolver r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* d = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(d) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  FillInputsZero(it.get());
  for (int i = 0; i < 3; ++i) it->Invoke();   // warmup (셰이더 컴파일 포함)
  *out_del = d;
  return it;
}

// ---------------------------------------------------------------------------
//  비용함수 (예측 makespan)
// ---------------------------------------------------------------------------
// CPU 집합을 P코어에 LPT로 스케줄한 makespan.
static double CpuMakespan(std::vector<double> c, int P) {
  if (c.empty()) return 0.0;
  std::sort(c.begin(), c.end(), std::greater<double>());
  std::vector<double> load(P, 0.0);
  for (double x : c) { int m = 0; for (int i = 1; i < P; ++i) if (load[i] < load[m]) m = i; load[m] += x; }
  return *std::max_element(load.begin(), load.end());
}
// 배분(dev)의 예측 makespan = max(ΣG_gpu, CpuMakespan(C_cpu, P)).
static double PredMakespan(const std::vector<Instance>& inst, const std::vector<ModelInfo>& M,
                           const std::vector<char>& dev, int P) {
  double g = 0; std::vector<double> c;
  for (size_t i = 0; i < inst.size(); ++i) {
    if (dev[i] == GPU) g += M[inst[i].model].G;
    else               c.push_back(M[inst[i].model].C);
  }
  return std::max(g, CpuMakespan(c, P));
}

// ---------------------------------------------------------------------------
//  배분 결정
//   - N<=20: 전수탐색(2^N)으로 예측 makespan 최소 = 최적.
//   - 그 외: C/G 내림차순 prefix 스캔(모든 지점 평가 후 최소 선택; 조기정지 버그 없음).
//   - GPU 불가 모델은 항상 CPU 고정.
// ---------------------------------------------------------------------------
//   전략: 전부 GPU에서 시작 → CPU에 싸게 얹을 수 있는 것부터(C/G 오름차순) 하나씩 CPU로.
//   ★규칙: offload 후 예측 CPU측 > 예측 GPU측 이면 그 offload는 취소하고 중단.
//          → CPU측이 GPU측을 절대 안 넘음 → makespan≈GPU측≤ALL-GPU (손해 방지, 대개 이득).
//   CPU 경합 보정: 예측 CPU시간에 ALPHA(≈1.5)를 곱해 "실제가 예측보다 느림"을 반영 → 과잉 offload 방지.
static const double ALPHA_CPU = 1.5;
static std::vector<char> Assign(const std::vector<Instance>& inst, const std::vector<ModelInfo>& M, int P) {
  int N = (int)inst.size();
  std::vector<char> dev(N, GPU);
  double gpu_side = 0; std::vector<double> cpu;
  for (int i = 0; i < N; ++i) {
    if (!M[inst[i].model].gpu_ok) { dev[i] = CPU; cpu.push_back(ALPHA_CPU * M[inst[i].model].C); }
    else                          { gpu_side += M[inst[i].model].G; }
  }
  double cur_max = std::max(gpu_side, CpuMakespan(cpu, P));

  std::vector<int> order;               // GPU에 있는 것들, C/G 오름차순(CPU에 싼 것 먼저)
  for (int i = 0; i < N; ++i) if (dev[i] == GPU) order.push_back(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return (M[inst[a].model].C / M[inst[a].model].G) < (M[inst[b].model].C / M[inst[b].model].G);
  });

  for (int id : order) {
    double gi = M[inst[id].model].G, ci = ALPHA_CPU * M[inst[id].model].C;
    double new_gpu = gpu_side - gi;
    std::vector<double> new_cpu = cpu; new_cpu.push_back(ci);
    double new_cpu_side = CpuMakespan(new_cpu, P);
    double new_max = std::max(new_gpu, new_cpu_side);
    // ★ CPU측이 GPU측을 넘으면 취소하고 중단. 개선될 때만 채택.
    if (new_cpu_side <= new_gpu && new_max < cur_max) {
      dev[id] = CPU; gpu_side = new_gpu; cpu = new_cpu; cur_max = new_max;
    } else {
      break;                            // 더 offload하면 CPU가 GPU를 넘음 → 중단
    }
  }
  return dev;
}

// ---------------------------------------------------------------------------
//  ★실제 실행★ : 배분(dev)대로 동시 실행하고 벽시계 makespan(ms) 반환
// ---------------------------------------------------------------------------
static double RunScenario(const std::vector<Instance>& inst, const std::vector<ModelInfo>& M,
                          const std::vector<char>& dev, int P,
                          double* gpu_side_out = nullptr, double* cpu_side_out = nullptr) {
  int N = (int)inst.size();

  // --- 준비(타이머 밖): 인터프리터 생성 + warmup ---
  std::map<int, std::unique_ptr<tflite::Interpreter>> gpu_it;  // 모델당 1개(공유, 직렬)
  std::vector<TfLiteDelegate*> gpu_dels;
  std::vector<std::unique_ptr<tflite::Interpreter>> cpu_it(N); // CPU 인스턴스마다 1개
  std::vector<int> gpu_list, cpu_list;                          // 실행할 인스턴스 인덱스

  for (int i = 0; i < N; ++i) {
    int m = inst[i].model;
    if (dev[i] == GPU) {
      if (!gpu_it.count(m)) {                    // 이 모델의 GPU 인터프리터 아직 없으면 생성
        TfLiteDelegate* d = nullptr;
        auto it = BuildGPU(*M[m].fb, &d);
        if (!it) { fprintf(stderr, "GPU build failed: %s\n", M[m].name.c_str()); return INF; }
        gpu_it[m] = std::move(it); gpu_dels.push_back(d);
      }
      gpu_list.push_back(i);
    } else {
      cpu_it[i] = BuildCPU(*M[m].fb);
      if (!cpu_it[i]) { fprintf(stderr, "CPU build failed: %s\n", M[m].name.c_str()); return INF; }
      cpu_list.push_back(i);
    }
  }
  // CPU는 긴 것부터(LPT 근사) 처리하도록 정렬
  std::sort(cpu_list.begin(), cpu_list.end(),
            [&](int a, int b) { return M[inst[a].model].C > M[inst[b].model].C; });

  // --- 실행(타이머 안) ---
  int nw = std::min(P, (int)cpu_list.size());
  std::vector<double> worker_end(std::max(nw,1), 0.0);   // 각 CPU 워커의 종료시각(상대)
  std::atomic<int> next(0);                      // CPU 공유 큐 인덱스

  auto t0 = std::chrono::high_resolution_clock::now();
  auto cpu_worker = [&](int wi) {
    int k;
    while ((k = next.fetch_add(1)) < (int)cpu_list.size())
      cpu_it[cpu_list[k]]->Invoke();             // 자기 인터프리터라 스레드 안전
    worker_end[wi] = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
  };

  std::vector<std::thread> workers;              // CPU 워커 (병렬)
  for (int i = 0; i < nw; ++i) workers.emplace_back(cpu_worker, i);

  for (int gi : gpu_list)                         // 메인 스레드 = GPU 워커(직렬), CPU와 동시 진행
    gpu_it[inst[gi].model]->Invoke();
  double gpu_side = std::chrono::duration<double, std::milli>(     // GPU측 완료시각
      std::chrono::high_resolution_clock::now() - t0).count();
  if (gpu_list.empty()) gpu_side = 0.0;

  for (auto& w : workers) w.join();               // CPU까지 다 끝날 때까지 대기

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double cpu_side = cpu_list.empty() ? 0.0 : *std::max_element(worker_end.begin(), worker_end.end());
  if (gpu_side_out) *gpu_side_out = gpu_side;
  if (cpu_side_out) *cpu_side_out = cpu_side;

  // --- 정리: 인터프리터 먼저, delegate 나중 ---
  gpu_it.clear(); cpu_it.clear();
  for (auto* d : gpu_dels) TfLiteGpuDelegateV2Delete(d);
  return ms;
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // ---- 인자 파싱 ----
  std::string profile_path = "profile.csv";
  std::string tag = "run";
  bool no_cpu_baseline = false;
  int P = (int)std::thread::hardware_concurrency(); if (P < 1) P = 1;
  std::vector<std::pair<std::string,int>> spec;   // (모델경로, 개수)
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_path = argv[++i];
    else if (std::strcmp(argv[i], "--tag") == 0 && i + 1 < argc) tag = argv[++i];
    else if (std::strcmp(argv[i], "--no-cpu-baseline") == 0) no_cpu_baseline = true;
    else if (std::strcmp(argv[i], "--cpu-cores") == 0 && i + 1 < argc) { P = std::atoi(argv[++i]); if (P<1) P=1; }
    else {
      std::string s = argv[i]; int cnt = 1;
      size_t c = s.find_last_of(':');            // "path:count" — 뒤가 숫자일 때만 count로
      if (c != std::string::npos && c + 1 < s.size() &&
          s.find_first_not_of("0123456789", c + 1) == std::string::npos) {
        cnt = std::atoi(s.c_str() + c + 1); s = s.substr(0, c);
      }
      spec.emplace_back(s, cnt < 1 ? 1 : cnt);
    }
  }
  if (spec.empty()) {
    fprintf(stderr, "Usage: %s --profile profile.csv <model.tflite:count> ... [--cpu-cores P]\n", argv[0]);
    return 1;
  }

  // ---- 비용표 로드 (basename -> {C,G,gpu_ok}) ----
  std::map<std::string, std::tuple<double,double,int>> table;
  if (FILE* f = std::fopen(profile_path.c_str(), "r")) {
    char line[1024]; bool header = true;
    while (std::fgets(line, sizeof(line), f)) {
      if (header) { header = false; continue; }           // 헤더 스킵
      std::string L(line);
      // CSV: model_path,C_ms,G_ms,gpu_ok,C_over_G
      std::vector<std::string> f5; size_t p = 0;
      for (int k = 0; k < 5; ++k) { size_t q = L.find(',', p); f5.push_back(L.substr(p, q-p)); if (q==std::string::npos) break; p = q+1; }
      if (f5.size() < 4) continue;
      table[BaseName(f5[0])] = { std::atof(f5[1].c_str()), std::atof(f5[2].c_str()), std::atoi(f5[3].c_str()) };
    }
    std::fclose(f);
  } else { fprintf(stderr, "cannot open profile: %s\n", profile_path.c_str()); return 1; }

  // ---- 모델 로드 + 인스턴스 펼치기 ----
  std::vector<ModelInfo> M;
  std::vector<Instance> inst;
  for (auto& sp : spec) {
    ModelInfo mi; mi.path = sp.first; mi.name = BaseName(sp.first);
    auto lk = table.find(mi.name);
    if (lk == table.end()) { fprintf(stderr, "not in profile: %s (run profile first)\n", mi.name.c_str()); return 1; }
    mi.C = std::get<0>(lk->second); mi.G = std::get<1>(lk->second); mi.gpu_ok = std::get<2>(lk->second) != 0;
    mi.fb = tflite::FlatBufferModel::BuildFromFile(mi.path.c_str());
    if (!mi.fb) { fprintf(stderr, "load failed: %s\n", mi.path.c_str()); return 1; }
    int idx = (int)M.size(); M.push_back(std::move(mi));
    for (int k = 0; k < sp.second; ++k) inst.push_back({idx});
  }
  printf("[CONFIG] instances=%zu, cpu-cores(P)=%d, profile=%s\n", inst.size(), P, profile_path.c_str());
  printf("[WORKLOAD] ");
  for (auto& sp : spec) printf("%s x%d  ", BaseName(sp.first).c_str(), sp.second);
  printf("\n\n");

  // ---- 배분 결정 ----
  std::vector<char> sched = Assign(inst, M, P);
  std::vector<char> all_gpu(inst.size(), GPU), all_cpu(inst.size(), CPU);
  for (size_t i = 0; i < inst.size(); ++i) if (!M[inst[i].model].gpu_ok) all_gpu[i] = CPU;

  // 배분 요약 출력
  auto count_dev = [&](const std::vector<char>& d, char want){ int n=0; for(char x:d) if(x==want) n++; return n; };
  int sched_gpu = count_dev(sched,GPU), sched_cpu = count_dev(sched,CPU);
  printf("[ASSIGN scheduled] GPU=%d, CPU=%d instances\n", sched_gpu, sched_cpu);
  printf("[ASSIGN per-model]");
  for (size_t m = 0; m < M.size(); ++m) {
    int g = 0, c = 0;
    for (size_t i = 0; i < inst.size(); ++i)
      if (inst[i].model == (int)m) { if (sched[i] == GPU) g++; else c++; }
    printf("  %s: GPU%d/CPU%d", M[m].name.c_str(), g, c);
  }
  printf("\n");

  // ---- 예측 makespan ----
  double pred = PredMakespan(inst, M, sched, P);

  // ---- 실측 (준비+실행) ----
  printf("running scenarios (build+warmup then timed) ...\n");
  double sched_gpu_side = 0, sched_cpu_side = 0;
  double ms_sched  = RunScenario(inst, M, sched,   P, &sched_gpu_side, &sched_cpu_side);
  double ms_allgpu = RunScenario(inst, M, all_gpu, P);
  double ms_allcpu = no_cpu_baseline ? -1.0 : RunScenario(inst, M, all_cpu, P);

  // ---- 결과 ----
  printf("\n================= RESULT (makespan, ms) =================\n");
  printf("  ALL-GPU  (measured) : %8.2f\n", ms_allgpu);
  if (ms_allcpu >= 0) printf("  ALL-CPU  (measured) : %8.2f\n", ms_allcpu);
  else                printf("  ALL-CPU  (measured) :   (skipped)\n");
  printf("  SCHEDULED(measured) : %8.2f   [GPU측 %.1f / CPU측 %.1f]\n", ms_sched, sched_gpu_side, sched_cpu_side);
  printf("  SCHEDULED(predicted): %8.2f   <- 산수 모델\n", pred);
  double base = (ms_allcpu >= 0) ? std::min(ms_allgpu, ms_allcpu) : ms_allgpu;
  double speedup = base / ms_sched;
  double gap = (pred / ms_sched - 1.0) * 100.0;
  printf("\n  speedup (measured, vs best baseline): %.2fx  (%.2f -> %.2f)\n", speedup, base, ms_sched);
  printf("  predict/measure gap : %+.1f%%  (예측이 실측 대비)\n", gap);
  printf("========================================================\n");
  // 집계용 한 줄 (driver가 grep). 열: tag,instances,gpu,cpu,all_gpu,all_cpu,sched,pred,speedup,gap%
  printf("SUMMARY,%s,%zu,%d,%d,%.2f,%.2f,%.2f,%.2f,%.3f,%.1f,%.2f,%.2f\n",
         tag.c_str(), inst.size(), sched_gpu, sched_cpu,
         ms_allgpu, ms_allcpu, ms_sched, pred, speedup, gap,
         sched_gpu_side, sched_cpu_side);
  return 0;
}
