// ============================================================================
//  온라인 Jackson 스케줄러 (jackson_online.cc)
// ----------------------------------------------------------------------------
//  split_profile.csv (오프라인 산출) 를 읽어, 여러 task를 internal-split로
//  CPU/GPU 동시 실행하고 makespan을 잰다. ALL-GPU와 비교.
//
//  흐름:
//    1. split_profile.csv → 모델별 (CPU-first/GPU-first: k, cpu_stage, gpu_stage), G(whole)
//    2. 워크로드(model:count) 파싱 → 인스턴스
//    3. 방향 배정: per-task 최적(= max(cpu,gpu) 작은 방향)
//    4. Jackson 순서(Johnson×2) 계산 → 디스패치 순서(방향 섞어 인터리브)
//    5. internal-split 인터프리터 pre-build(+warmup) → 스레드풀 동시 실행 → makespan
//    6. ALL-GPU baseline 동일 측정 → 비교
//
//  ★ 한계(정직): TFLite Invoke는 원자적이라 CPU-part/GPU-part를 따로 큐에 못 넣음.
//     그래서 "진짜 Jackson 부품 스케줄"이 아니라, Jackson 순서로 '통짜 Invoke를
//     디스패치 + 방향 섞기 + 스레드 동시성'으로 CPU/GPU 겹침을 근사한다.
//
//  사용법:
//    jackson_online --profile split_profile.csv <model.tflite:count> ... [--threads T]
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
#include <climits>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#ifndef TFLITE_DEBUG_DELEGATE
#error "TFLITE_DEBUG_DELEGATE not defined"
#endif

static const double INF = std::numeric_limits<double>::infinity();
enum Dir { CPU_FIRST = 0, GPU_FIRST = 1 };

struct DirProf { int k = 0; double cpu = 0, gpu = 0; bool has = false; };
struct MProf {
  std::string name, path;
  double Gwhole = INF;                 // ALL-GPU 통짜 시간
  DirProf cf, gf;                      // CPU-first / GPU-first
  std::unique_ptr<tflite::FlatBufferModel> fb;
};
struct Instance { int m; Dir dir; };

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

// internal-split 인터프리터: dir/k 적용. whole-GPU면 dir=CPU_FIRST,k=0.
static std::unique_ptr<tflite::Interpreter> BuildSplit(
    const tflite::FlatBufferModel& model, Dir dir, int k, TfLiteDelegate** out_del) {
  *out_del = nullptr;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;  // XNNPACK off
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(model, resolver); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  if (dir == CPU_FIRST) { o.first_delegate_node_index = k; o.last_delegate_node_index = INT_MAX; }
  else                  { o.first_delegate_node_index = 0; o.last_delegate_node_index = k;      }
  TfLiteDelegate* gpu = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(gpu) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return nullptr; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(gpu); return nullptr; }
  FillInputsZero(it.get());
  for (int i = 0; i < 3; ++i) it->Invoke();   // warmup
  *out_del = gpu;
  return it;
}

// Johnson's rule: p1<=p2 인 것 p1오름, 나머지 p2내림.
static std::vector<int> Johnson(std::vector<int> ids,
    const std::vector<double>& p1, const std::vector<double>& p2) {
  std::vector<int> A, B;
  for (int i : ids) (p1[i] <= p2[i] ? A : B).push_back(i);
  std::sort(A.begin(), A.end(), [&](int a, int b) { return p1[a] < p1[b]; });
  std::sort(B.begin(), B.end(), [&](int a, int b) { return p2[a] > p2[b]; });
  A.insert(A.end(), B.begin(), B.end());
  return A;
}

// 배정(inst) → internal-split 인터프리터들 pre-build+warmup → 디스패치순 동시실행 → makespan(ms)
static double RunConcurrent(const std::vector<MProf>& M, const std::vector<Instance>& inst,
                            const std::vector<int>& order, int T) {
  int N = (int)inst.size();
  std::vector<std::unique_ptr<tflite::Interpreter>> its(N);
  std::vector<TfLiteDelegate*> dels(N, nullptr);
  for (int i = 0; i < N; ++i) {
    const MProf& mp = M[inst[i].m];
    int k = (inst[i].dir == CPU_FIRST) ? mp.cf.k : mp.gf.k;
    TfLiteDelegate* d = nullptr;
    its[i] = BuildSplit(*mp.fb, inst[i].dir, k, &d);
    if (!its[i]) { fprintf(stderr, "build failed: %s\n", mp.name.c_str()); return INF; }
    dels[i] = d;
  }
  // 실행 (타이머 안): 워커 T개가 order 순서로 하나씩 집어 Invoke
  std::atomic<int> next(0);
  auto worker = [&]() {
    int p;
    while ((p = next.fetch_add(1)) < N) its[order[p]]->Invoke();
  };
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> ws;
  for (int i = 0; i < std::min(T, N); ++i) ws.emplace_back(worker);
  for (auto& w : ws) w.join();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  its.clear();
  for (auto* d : dels) if (d) TfLiteGpuDelegateV2Delete(d);
  return ms;
}

int main(int argc, char* argv[]) {
  std::string profile = "split_profile.csv";
  int T = 4;
  std::vector<std::pair<std::string,int>> spec;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile = argv[++i];
    else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) { T = std::atoi(argv[++i]); if (T<1) T=1; }
    else {
      std::string s = argv[i]; int cnt = 1;
      size_t c = s.find_last_of(':');
      if (c != std::string::npos && c + 1 < s.size() &&
          s.find_first_not_of("0123456789", c + 1) == std::string::npos) {
        cnt = std::atoi(s.c_str() + c + 1); s = s.substr(0, c);
      }
      spec.emplace_back(s, cnt < 1 ? 1 : cnt);
    }
  }
  if (spec.empty()) { fprintf(stderr, "Usage: %s --profile split_profile.csv <m.tflite:cnt> ... [--threads T]\n", argv[0]); return 1; }

  // ---- split_profile.csv 로드 (basename → 방향별 프로파일) ----
  std::map<std::string, MProf> table;
  if (FILE* f = std::fopen(profile.c_str(), "r")) {
    char line[512]; bool header = true;
    while (std::fgets(line, sizeof(line), f)) {
      if (header) { header = false; continue; }
      std::string L(line); std::vector<std::string> c; size_t p = 0;
      for (int i = 0; i < 8; ++i) { size_t q = L.find(',', p); c.push_back(L.substr(p, q - p)); if (q == std::string::npos) break; p = q + 1; }
      if (c.size() < 5) continue;
      std::string nm = c[0], dir = c[1];
      MProf& mp = table[nm]; mp.name = nm;
      int k = std::atoi(c[2].c_str()); double cpu = std::atof(c[3].c_str()), gpu = std::atof(c[4].c_str());
      if (dir == "CPU-first") { mp.cf = {k, cpu, gpu, true}; }
      else if (dir == "GPU-first") { mp.gf = {k, cpu, gpu, true}; }
      else if (dir == "ALL-GPU") { mp.Gwhole = gpu; }
    }
    std::fclose(f);
  } else { fprintf(stderr, "cannot open %s\n", profile.c_str()); return 1; }

  // ---- 모델 로드 + 인스턴스 펼치기 + 방향 배정(per-task 최적) ----
  std::vector<MProf> M;
  std::map<std::string,int> idx;
  std::vector<Instance> inst;
  for (auto& sp : spec) {
    std::string nm = BaseName(sp.first);
    if (!table.count(nm)) { fprintf(stderr, "not in profile: %s\n", nm.c_str()); return 1; }
    if (!idx.count(nm)) {
      MProf mp = std::move(table[nm]); mp.path = sp.first;
      mp.fb = tflite::FlatBufferModel::BuildFromFile(sp.first.c_str());
      if (!mp.fb) { fprintf(stderr, "load failed: %s\n", sp.first.c_str()); return 1; }
      idx[nm] = (int)M.size(); M.push_back(std::move(mp));
    }
    int mi = idx[nm];
    // per-task 최적 방향 = max(cpu,gpu) 작은 쪽
    double mcf = std::max(M[mi].cf.cpu, M[mi].cf.gpu);
    double mgf = std::max(M[mi].gf.cpu, M[mi].gf.gpu);
    Dir best = (mgf < mcf) ? GPU_FIRST : CPU_FIRST;
    for (int r = 0; r < sp.second; ++r) inst.push_back({mi, best});
  }
  int N = (int)inst.size();
  printf("[CONFIG] instances=%d, threads=%d\n[WORKLOAD] ", N, T);
  for (auto& sp : spec) printf("%s x%d  ", BaseName(sp.first).c_str(), sp.second);
  printf("\n");

  // ---- 방향별 stage로 Jackson 순서 → 디스패치 순서(인터리브) ----
  std::vector<double> p1(N), p2(N);   // 각 인스턴스의 (첫기계, 둘째기계) 시간
  std::vector<int> CG, GC;
  for (int i = 0; i < N; ++i) {
    const MProf& mp = M[inst[i].m];
    if (inst[i].dir == CPU_FIRST) { p1[i] = mp.cf.cpu; p2[i] = mp.cf.gpu; CG.push_back(i); }
    else                          { p1[i] = mp.gf.gpu; p2[i] = mp.gf.cpu; GC.push_back(i); }
  }
  std::vector<int> jcg = Johnson(CG, p1, p2);   // CPU-first 그룹
  std::vector<int> jgc = Johnson(GC, p1, p2);   // GPU-first 그룹
  // 디스패치 순서 = GC/CG 인터리브(방향 섞어 동시성 확보)
  std::vector<int> order;
  for (size_t i = 0; i < std::max(jgc.size(), jcg.size()); ++i) {
    if (i < jgc.size()) order.push_back(jgc[i]);
    if (i < jcg.size()) order.push_back(jcg[i]);
  }
  int ncg = (int)CG.size(), ngc = (int)GC.size();
  printf("[ASSIGN] CPU-first(CG)=%d, GPU-first(GC)=%d\n", ncg, ngc);

  // ---- SCHEDULED 실측 ----
  double ms_sched = RunConcurrent(M, inst, order, T);

  // ---- ALL-GPU baseline: 전부 whole-GPU (dir=CPU_FIRST + k=0) ----
  //   RunConcurrent는 dir/k를 M[inst.m]에서 읽으므로, cf.k를 임시로 0(전부 GPU)으로.
  std::vector<Instance> allg(inst);
  for (auto& x : allg) x.dir = CPU_FIRST;
  std::vector<int> orderg(N); for (int i = 0; i < N; ++i) orderg[i] = i;
  std::vector<int> savedk(M.size());
  for (size_t i = 0; i < M.size(); ++i) { savedk[i] = M[i].cf.k; M[i].cf.k = 0; }
  double ms_allgpu = RunConcurrent(M, allg, orderg, T);
  for (size_t i = 0; i < M.size(); ++i) M[i].cf.k = savedk[i];

  printf("\n============ RESULT (makespan, ms) ============\n");
  printf("  ALL-GPU   (measured): %8.2f\n", ms_allgpu);
  printf("  SCHEDULED (measured): %8.2f\n", ms_sched);
  printf("  speedup vs ALL-GPU  : %.2fx\n", ms_allgpu / ms_sched);
  printf("===============================================\n");
  return 0;
}
