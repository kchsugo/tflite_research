// ============================================================================
//  오프라인 split-profiler
// ----------------------------------------------------------------------------
//  목적: 각 모델을 1컷 내부 분할했을 때, 두 방향 각각의 "균형 cut"과
//        그 지점의 (cpu_stage, gpu_stage, boundary)를 측정한다.
//        → 온라인 Jackson 스케줄러의 입력(각 task의 두 스테이지 시간).
//
//  방향:
//    CPU-first : first_delegate_node_index=k  → 노드 0..k-1 CPU, k.. GPU
//    GPU-first : last_delegate_node_index=k   → 노드 0..k GPU, k+1.. CPU
//    (둘 다 TFLITE_DEBUG_DELEGATE 필요)
//
//  스테이지 분리 측정:
//    Profiler로 노드별 시간을 받아, execution_plan 순서로
//      delegate 노드(node.delegate!=null) 시간 합 = gpu_stage
//      나머지 노드 시간 합                        = cpu_stage
//    (per-op GPU 시간이 아니라 'delegate 커널 노드 1개'의 시간을 GPU-stage로)
//
//  균형 cut: f(k)=cpu_stage-gpu_stage 가 방향별로 단조 → binary search.
//
//  사용법:
//    split_profiler <m1.tflite> ... [--out split_profile.csv] [--runs 10]
// ============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <limits>
#include <climits>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/core/api/profiler.h"

#ifndef TFLITE_DEBUG_DELEGATE
#error "TFLITE_DEBUG_DELEGATE not defined — check CMake add_compile_definitions"
#endif

static const double INF = std::numeric_limits<double>::infinity();
enum Dir { CPU_FIRST = 0, GPU_FIRST = 1 };

// ---- 노드별 시간 수집 Profiler ----
class StageProfiler : public tflite::Profiler {
 public:
  struct E { EventType type; double us; };
  std::vector<E> events;
  uint32_t BeginEvent(const char* tag, EventType t, int64_t, int64_t) override {
    starts_.push_back(std::chrono::high_resolution_clock::now());
    types_.push_back(t);
    return (uint32_t)(starts_.size() - 1);
  }
  void EndEvent(uint32_t h) override {
    auto end = std::chrono::high_resolution_clock::now();
    if (h < starts_.size())
      events.push_back({types_[h],
        std::chrono::duration<double, std::micro>(end - starts_[h]).count()});
  }
  void clear() { starts_.clear(); types_.clear(); events.clear(); }
 private:
  std::vector<std::chrono::high_resolution_clock::time_point> starts_;
  std::vector<EventType> types_;
};

struct Cut {
  int    k = 0;
  double cpu_ms = 0, gpu_ms = 0;
  double cost = INF;                 // 처리량 목적: max(cpu_ms, gpu_ms). gpu_ms엔 경계 이미 포함.
  size_t boundary_bytes = 0;
  int    cpu_nodes = 0, gpu_nodes = 0;
  bool   ok = false;
};

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

// delegate 경계를 넘는(producer/consumer의 delegated 상태가 다른) 비상수 텐서 bytes 합.
static size_t BoundaryBytes(tflite::Interpreter* it) {
  const auto& plan = it->execution_plan();
  std::vector<int> producedByDelegate(it->tensors_size(), -1);  // -1 unknown, 0 cpu, 1 gpu
  for (int nid : plan) {
    const TfLiteNode& n = it->node_and_registration(nid)->first;
    int d = (n.delegate != nullptr) ? 1 : 0;
    if (n.outputs)
      for (int i = 0; i < n.outputs->size; ++i)
        if (n.outputs->data[i] >= 0) producedByDelegate[n.outputs->data[i]] = d;
  }
  size_t bytes = 0;
  std::vector<char> counted(it->tensors_size(), 0);
  for (int nid : plan) {
    const TfLiteNode& n = it->node_and_registration(nid)->first;
    int d = (n.delegate != nullptr) ? 1 : 0;
    if (!n.inputs) continue;
    for (int i = 0; i < n.inputs->size; ++i) {
      int ti = n.inputs->data[i];
      if (ti < 0 || counted[ti]) continue;
      const TfLiteTensor* t = it->tensor(ti);
      if (!t || t->allocation_type == kTfLiteMmapRo) continue;   // 상수 제외
      if (producedByDelegate[ti] != -1 && producedByDelegate[ti] != d) {
        bytes += t->bytes; counted[ti] = 1;
      }
    }
  }
  return bytes;
}

// 절단점 k, 방향 dir 에서 스테이지 시간 측정.
static Cut Measure(const tflite::FlatBufferModel& model, Dir dir, int k, int runs) {
  Cut r; r.k = k;
  // XNNPACK(기본 CPU delegate) 비활성: 안 그러면 CPU 연산이 delegate 노드로 융합돼
  // 노드 분류/cut/프로파일이 깨진다. CPU part는 원본 노드로 남겨야 함.
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder builder(model, resolver);
  builder.SetNumThreads(1);
  if (builder(&it) != kTfLiteOk || !it) return r;

  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  if (dir == CPU_FIRST) { o.first_delegate_node_index = k;   o.last_delegate_node_index = INT_MAX; }
  else                  { o.first_delegate_node_index = 0;   o.last_delegate_node_index = k;      }
  TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(gpu) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return r; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return r; }
  FillInputsZero(it.get());

  // execution_plan 노드별 delegate 여부
  const auto plan = it->execution_plan();
  std::vector<char> deleg(plan.size());
  for (size_t i = 0; i < plan.size(); ++i) {
    bool d = it->node_and_registration(plan[i])->first.delegate != nullptr;
    deleg[i] = d ? 1 : 0;
    if (d) r.gpu_nodes++; else r.cpu_nodes++;
  }
  r.boundary_bytes = BoundaryBytes(it.get());

  StageProfiler prof;
  it->SetProfiler(&prof);
  for (int i = 0; i < 2; ++i) it->Invoke();          // warmup (셰이더 컴파일 포함)

  // CPU-stage = profiler의 CPU(비delegate) 노드 시간 합 (동기라 신뢰성 있음)
  // GPU-stage = 전체 Invoke 벽시계 - CPU-stage (GPU는 비동기라 노드 시간 못 믿음;
  //             Invoke는 GPU 끝까지 블록하므로 전체 시간에 실제 GPU가 포함됨. 경계도 여기 포함)
  double cpu_sum = 0, tot_sum = 0;
  for (int run = 0; run < runs; ++run) {
    prof.clear();
    auto t0 = std::chrono::high_resolution_clock::now();
    if (it->Invoke() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return r; }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    int oi = 0; double c = 0;
    for (auto& e : prof.events) {
      if (e.type == tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT) {
        if (oi < (int)plan.size() && !deleg[oi]) c += e.us;   // CPU 노드만
        oi++;
      }
    }
    cpu_sum += c; tot_sum += total_us;
  }
  it->SetProfiler(nullptr);
  r.cpu_ms = cpu_sum / runs / 1000.0;
  double tot_ms = tot_sum / runs / 1000.0;
  r.gpu_ms = std::max(0.0, tot_ms - r.cpu_ms);   // GPU + 경계
  r.ok = true;
  it.reset();
  TfLiteGpuDelegateV2Delete(gpu);
  return r;
}

// 방향 dir 의 최적 cut을 '전체 k 스윕'으로 찾는다.
//   목적함수 = 처리량: cost(k) = max(cpu_stage, gpu_stage).
//     - gpu_ms엔 경계(PCIe) 전송이 이미 포함 → boundary 항을 따로 더하지 않음(이중계산 방지).
//     - k=0(CPU-first)=통짜 GPU도 후보라, 어떤 split도 통짜를 못 이기면 통짜(k=0)가 선택된다.
//   균형(cpu≈gpu)이 아니라 실제 cost 최소 지점을 고른다. cost는 boundary/오버헤드로 비단조라
//   binary search 대신 스윕. node가 많으면 grid로 stride를 둬 측정 횟수를 제한.
//   curve!=null 이면 스윕한 모든 점을 기록(cut-cost 곡선 시각화·BW 투영용).
static Cut FindBest(const tflite::FlatBufferModel& model, Dir dir, int N, int runs,
                    int grid, std::vector<std::pair<Dir,Cut>>* curve) {
  Cut best; double best_cost = INF;
  int stride = std::max(1, (N + grid - 1) / grid);   // 대략 grid개 점
  for (int k = 0; k <= N; k += stride) {
    Cut r = Measure(model, dir, k, runs);
    if (!r.ok) continue;
    r.cost = std::max(r.cpu_ms, r.gpu_ms);
    if (curve) curve->push_back({dir, r});
    if (r.cost < best_cost) { best_cost = r.cost; best = r; }
  }
  return best;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> models;
  std::string out_path = "split_profile.csv";
  std::string curve_path = "split_curve.csv";
  int runs = 10;
  int grid = 20;                 // 방향당 스윕 점 수 (측정 시간 상한)
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    else if (std::strcmp(argv[i], "--curve") == 0 && i + 1 < argc) curve_path = argv[++i];
    else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) grid = std::max(2, std::atoi(argv[++i]));
    else models.push_back(argv[i]);
  }
  if (models.empty()) {
    fprintf(stderr, "Usage: %s <m1.tflite> ... [--out split_profile.csv] [--curve split_curve.csv] [--runs 10] [--grid 20]\n", argv[0]);
    return 1;
  }

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
  std::fprintf(f, "model,direction,k,cpu_stage_ms,gpu_stage_ms,boundary_kb,cpu_nodes,gpu_nodes\n");
  // cut-cost 곡선 전체(모든 스윕 점) — 시각화 / 통합메모리 BW 투영용
  FILE* fc = std::fopen(curve_path.c_str(), "w");
  if (fc) std::fprintf(fc, "model,direction,k,cpu_stage_ms,gpu_stage_ms,cost_ms,boundary_kb,cpu_nodes,gpu_nodes\n");

  printf("\n%-26s | %-9s | %-4s | %9s | %9s | %9s | %10s | %s\n",
         "model", "direction", "k", "cpu(ms)", "gpu(ms)", "cost(ms)", "bound(KB)", "verdict");
  printf("-------------------------------------------------------------------------------------------------------\n");

  for (const auto& path : models) {
    auto model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!model) { fprintf(stderr, "load failed: %s\n", path.c_str()); continue; }
    // 노드 총수
    int N = 0;
    {
      tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates res;
      std::unique_ptr<tflite::Interpreter> it;
      tflite::InterpreterBuilder b(*model, res); b.SetNumThreads(1);
      if (b(&it) == kTfLiteOk && it) N = it->nodes_size();
    }
    std::string nm = BaseName(path);

    // 참고: 전부 GPU (k=0, CPU-first) / 전부 CPU (k=N)
    Cut allg = Measure(*model, CPU_FIRST, 0, runs);
    Cut allc = Measure(*model, CPU_FIRST, N, runs);

    double g_whole = allg.gpu_ms;   // 통짜 GPU 기준 (split이 이걸 이겨야 의미)
    std::vector<std::pair<Dir,Cut>> curve;
    for (int d = 0; d < 2; ++d) {
      Cut b = FindBest(*model, (Dir)d, N, runs, grid, &curve);
      const char* dname = (d == CPU_FIRST) ? "CPU-first" : "GPU-first";
      // split이 통짜를 이기는지 표시(cost < g_whole). k=0(CPU-first)이면 사실상 통짜.
      const char* verdict = (b.cost < g_whole - 1e-6) ? "SPLIT<whole" : "whole win";
      printf("%-26s | %-9s | %-4d | %9.3f | %9.3f | %9.3f | %10.1f | %s\n",
             nm.substr(0, 26).c_str(), dname, b.k, b.cpu_ms, b.gpu_ms, b.cost,
             b.boundary_bytes / 1024.0, verdict);
      std::fprintf(f, "%s,%s,%d,%.4f,%.4f,%.2f,%d,%d\n",
                   nm.c_str(), dname, b.k, b.cpu_ms, b.gpu_ms,
                   b.boundary_bytes / 1024.0, b.cpu_nodes, b.gpu_nodes);
    }
    // cut-cost 곡선 전체 기록
    if (fc) for (auto& pr : curve) {
      const char* dn = (pr.first == CPU_FIRST) ? "CPU-first" : "GPU-first";
      const Cut& c = pr.second;
      std::fprintf(fc, "%s,%s,%d,%.4f,%.4f,%.4f,%.2f,%d,%d\n",
                   nm.c_str(), dn, c.k, c.cpu_ms, c.gpu_ms, c.cost,
                   c.boundary_bytes / 1024.0, c.cpu_nodes, c.gpu_nodes);
    }
    // 참고행
    printf("%-26s | %-9s | %-4d | %9.3f | %9.3f | %10s\n",
           nm.substr(0, 26).c_str(), "ALL-GPU", 0, allg.cpu_ms, allg.gpu_ms, "-");
    printf("%-26s | %-9s | %-4d | %9.3f | %9.3f | %10s\n",
           nm.substr(0, 26).c_str(), "ALL-CPU", N, allc.cpu_ms, allc.gpu_ms, "-");
    std::fprintf(f, "%s,ALL-GPU,0,%.4f,%.4f,0,0,0\n", nm.c_str(), allg.cpu_ms, allg.gpu_ms);
    std::fprintf(f, "%s,ALL-CPU,%d,%.4f,%.4f,0,0,0\n", nm.c_str(), N, allc.cpu_ms, allc.gpu_ms);
    printf("--------------------------------------------------------------------------------------\n");
  }
  std::fclose(f);
  if (fc) { std::fclose(fc); printf("[SAVED] %s (cut-cost curve)\n", curve_path.c_str()); }
  printf("[SAVED] %s\n", out_path.c_str());
  return 0;
}
