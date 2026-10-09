// ============================================================================
//  통합 플래너 (unified.cc)
// ----------------------------------------------------------------------------
//  하나의 배치(N개 task)에 대해, 각 task를 세 모드 중 최적으로 배정한다:
//     M_GPU  (통짜 GPU)  /  M_CPU (통짜 CPU, P코어 분산)  /  M_SPLIT (1컷 내부분할)
//  목적: makespan = max( GPU레인,  CPU레인(P코어) )  최소화.
//        → "분할이 이득이면 분할(A/jackson 로직), 통짜가 최적이라도 GPU가 병목이면
//           통짜 task를 CPU로 offload(B 로직)"를 한 목적함수가 동시에 수행.
//
//  [비용값 = 런타임 캘리브레이션] 정적 표(α 매직넘버) 대신 시작 시 실측:
//     g_eff = 통짜 GPU 연속실행 per-task 시간
//     c_eff = 통짜 CPU를 P개 동시 실행할 때 per-task 시간 (경합 포함)
//     split stage = split_profile.csv (방향별 cpu/gpu, XNNPACK-off 측정)
//   GPU 시간은 XNNPACK 무관이라 split_gpu와 g_eff는 같은 단위로 비교 가능.
//   CPU는 whole=XNNPACK-on(빠름), split_cpu=XNNPACK-off(느림) → split에 보수적(문제 없음).
//
//  [실행] GPU=메인 스레드 직렬(모델당 공유 컨텍스트). CPU/SPLIT=P워커 풀(인스턴스별 인터프리터).
//         측정 구간은 순수 Invoke 동시실행. 준비/warmup은 타이머 밖.
//
//  사용법:
//    unified --split-profile split_profile.csv <model.tflite:count> ... [--cpu-cores P] [--runs R]
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

static const double INF = std::numeric_limits<double>::infinity();
enum Mode { M_GPU = 0, M_CPU = 1, M_SPLIT = 2 };
enum Dir  { CPU_FIRST = 0, GPU_FIRST = 1 };
static const char* MODE_NAME[3] = {"GPU", "CPU", "SPLIT"};

struct ModelInfo {
  std::string path, name;
  double g_eff = INF, c_eff = INF;             // g_eff=실제 쓸 GPU per-task(배칭 반영), c_eff=경합 CPU
  double g1 = INF;                             // batch=1 GPU per-task (참고)
  int    bB = 1;                               // 이 모델에 쓸 배치 크기 (이득 없으면 1 → VRAM 절약)
  bool   gpu_ok = false;
  // split 후보 (split_profile.csv, 선택 방향)
  bool   split_ok = false;
  int    split_dir = CPU_FIRST, split_k = 0;
  double split_cpu = INF, split_gpu = INF;
  std::unique_ptr<tflite::FlatBufferModel> fb;
};
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
static double NowMs() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// 통짜 CPU 인터프리터 (XNNPACK on = 실제 빠른 CPU 경로), 1스레드.
static std::unique_ptr<tflite::Interpreter> BuildCPU(const tflite::FlatBufferModel& m) {
  tflite::ops::builtin::BuiltinOpResolver r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  if (it->AllocateTensors() != kTfLiteOk) return nullptr;
  FillInputsZero(it.get());
  for (int i = 0; i < 2; ++i) it->Invoke();
  return it;
}
// 통짜 GPU 인터프리터 (delegate 전체). B>1이면 입력을 batch 차원으로 묶음(배칭).
static std::unique_ptr<tflite::Interpreter> BuildGPU(const tflite::FlatBufferModel& m,
                                                     TfLiteDelegate** out_del, int B=1) {
  *out_del = nullptr;
  tflite::ops::builtin::BuiltinOpResolver r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  if (B > 1) {                                  // 입력 batch 차원 교체 → delegate/할당 전에
    int in = it->inputs()[0]; TfLiteTensor* t = it->tensor(in);
    if (!t || !t->dims || t->dims->size < 1) return nullptr;
    std::vector<int> dims(t->dims->data, t->dims->data + t->dims->size); dims[0] = B;
    if (it->ResizeInputTensor(in, dims) != kTfLiteOk) return nullptr;
  }
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* d = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(d) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  FillInputsZero(it.get());
  for (int i = 0; i < 3; ++i) it->Invoke();
  *out_del = d;
  return it;
}
// 내부 분할 인터프리터 (XNNPACK off = 노드 분리 유지, dir/k 적용).
static std::unique_ptr<tflite::Interpreter> BuildSplit(const tflite::FlatBufferModel& m,
                                                       Dir dir, int k, TfLiteDelegate** out_del) {
  *out_del = nullptr;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return nullptr;
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  if (dir == CPU_FIRST) { o.first_delegate_node_index = k; o.last_delegate_node_index = INT_MAX; }
  else                  { o.first_delegate_node_index = 0; o.last_delegate_node_index = k;      }
  TfLiteDelegate* d = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(d) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return nullptr; }
  FillInputsZero(it.get());
  for (int i = 0; i < 3; ++i) it->Invoke();
  *out_del = d;
  return it;
}

// P코어 LPT makespan (online.cc와 동일).
static double CpuMakespan(std::vector<double> c, int P) {
  if (c.empty()) return 0.0;
  std::sort(c.begin(), c.end(), std::greater<double>());
  std::vector<double> load(P, 0.0);
  for (double x : c) { int m = 0; for (int i = 1; i < P; ++i) if (load[i] < load[m]) m = i; load[m] += x; }
  return *std::max_element(load.begin(), load.end());
}

// ---------------------------------------------------------------------------
//  캘리브레이션: 실제 g_eff / c_eff 측정 (정적 표·α 대체)
// ---------------------------------------------------------------------------
// 배칭 per-task GPU 시간: batch B로 묶어 재고 total/B. OOM이면 B 반감 폴백. usedB 반환.
static double CalibG(const tflite::FlatBufferModel& m, int B, int R, int* usedB) {
  for (int b = B; b >= 1; b /= 2) {
    TfLiteDelegate* d = nullptr; auto it = BuildGPU(m, &d, b);
    if (!it) { if (b == 1) break; else continue; }
    double sum = 0;
    for (int i = 0; i < R; ++i) { double t0 = NowMs(); it->Invoke(); sum += NowMs() - t0; }
    it.reset(); TfLiteGpuDelegateV2Delete(d);
    if (usedB) *usedB = b;
    return sum / R / b;                         // per-task
  }
  return INF;
}
// P개 통짜 CPU 인터프리터를 동시 실행할 때의 per-task 시간(경합 포함).
static double CalibC(const tflite::FlatBufferModel& m, int P, int R) {
  std::vector<std::unique_ptr<tflite::Interpreter>> its(P);
  for (int i = 0; i < P; ++i) { its[i] = BuildCPU(m); if (!its[i]) return INF; }
  double sum = 0;
  for (int r = 0; r < R; ++r) {
    double t0 = NowMs();
    std::vector<std::thread> ts;
    for (int i = 0; i < P; ++i) ts.emplace_back([&, i]{ its[i]->Invoke(); });
    for (auto& t : ts) t.join();
    sum += NowMs() - t0;                       // P개가 함께 끝난 벽시계 ≈ 경합 하 per-task
  }
  return sum / R;
}

// ---------------------------------------------------------------------------
//  배정 (3모드 그리디): 전부 GPU에서 시작 → makespan 줄이는 move를 탐욕적으로.
//   move 후보: task를 CPU로 / SPLIT로. 균형 가드(CPU레인 ≤ GPU레인) 유지, 개선될 때만.
// ---------------------------------------------------------------------------
static std::vector<char> Assign(const std::vector<Instance>& inst,
                                const std::vector<ModelInfo>& M, int P) {
  int N = (int)inst.size();
  int Pc = std::max(1, P - 1);                          // GPU 구동용 코어 1개 예약
  std::vector<char> dev(N, M_GPU);
  double gpu = 0; std::vector<double> cpu;              // GPU레인 합, CPU레인 아이템들
  for (int i = 0; i < N; ++i) {
    const ModelInfo& m = M[inst[i].model];
    if (m.gpu_ok) gpu += m.g_eff;
    else { dev[i] = M_CPU; cpu.push_back(m.c_eff); }    // GPU 불가면 CPU 고정
  }
  auto makespan = [&](double g, std::vector<double>& c){ return std::max(g, CpuMakespan(c, Pc)); };
  double cur = makespan(gpu, cpu);

  while (true) {
    int best_i = -1; char best_mode = 0; double best_ms = cur;
    double best_gpu = gpu; std::vector<double> best_cpu;
    for (int i = 0; i < N; ++i) {
      if (dev[i] != M_GPU) continue;
      const ModelInfo& m = M[inst[i].model];
      // 후보 1: 통짜 CPU
      {
        double ng = gpu - m.g_eff;
        std::vector<double> nc = cpu; nc.push_back(m.c_eff);
        double ncpu = CpuMakespan(nc, Pc);
        double nm = std::max(ng, ncpu);
        if (ncpu <= ng + 1e-9 && nm < best_ms - 1e-9) {   // 균형 가드 + 개선
          best_ms = nm; best_i = i; best_mode = M_CPU; best_gpu = ng; best_cpu = nc;
        }
      }
      // 후보 2: 분할 (gpu_stage는 GPU레인 유지, cpu_stage는 CPU레인)
      if (m.split_ok) {
        double ng = gpu - (m.g_eff - m.split_gpu);        // GPU에서 던 양 = g_eff - split_gpu
        std::vector<double> nc = cpu; nc.push_back(m.split_cpu);
        double ncpu = CpuMakespan(nc, Pc);
        double nm = std::max(ng, ncpu);
        if (ncpu <= ng + 1e-9 && nm < best_ms - 1e-9) {
          best_ms = nm; best_i = i; best_mode = M_SPLIT; best_gpu = ng; best_cpu = nc;
        }
      }
    }
    if (best_i < 0) break;                                 // 개선 move 없음 → 종료
    dev[best_i] = best_mode; gpu = best_gpu; cpu = best_cpu; cur = best_ms;
  }
  return dev;
}

// ---------------------------------------------------------------------------
//  실행: 배정대로 동시 실행 → makespan(ms) + GPU측/CPU측 완료시각(사용률 지표)
// ---------------------------------------------------------------------------
static double RunScenario(const std::vector<Instance>& inst, const std::vector<ModelInfo>& M,
                          const std::vector<char>& dev, int P, bool use_batch,
                          double* gpu_side_out, double* cpu_side_out) {
  int N = (int)inst.size();
  std::vector<TfLiteDelegate*> dels;
  std::vector<std::unique_ptr<tflite::Interpreter>> gpu_interps;        // 배칭 GPU 인터프리터(유지)
  std::vector<std::pair<tflite::Interpreter*,int>> gpu_sched;           // (interp, 반복수) 메인 실행
  std::vector<std::unique_ptr<tflite::Interpreter>> pool_it(N);         // CPU/SPLIT 인스턴스별
  std::vector<int> pool_list;
  std::map<int,int> gpu_count;                                          // 모델별 GPU task 수
  bool has_gpu = false;

  for (int i = 0; i < N; ++i) {
    int m = inst[i].model;
    if (dev[i] == M_GPU) { gpu_count[m]++; has_gpu = true; }
    else if (dev[i] == M_CPU) {
      pool_it[i] = BuildCPU(*M[m].fb);
      if (!pool_it[i]) { fprintf(stderr, "CPU build fail %s\n", M[m].name.c_str()); return INF; }
      pool_list.push_back(i);
    } else { // M_SPLIT
      TfLiteDelegate* d = nullptr;
      pool_it[i] = BuildSplit(*M[m].fb, (Dir)M[m].split_dir, M[m].split_k, &d);
      if (!pool_it[i]) { fprintf(stderr, "SPLIT build fail %s\n", M[m].name.c_str()); return INF; }
      dels.push_back(d); pool_list.push_back(i);
    }
  }
  // 모델별 GPU task를 배칭: B=min(batch,n)로 묶어 ceil(n/B)번 Invoke. OOM이면 B 반감 폴백.
  for (auto& kv : gpu_count) {
    int m = kv.first, n = kv.second;
    int B = use_batch ? std::min(M[m].bB, n) : 1;   // 모델별 배치(이득 없는 모델은 bB=1)
    std::unique_ptr<tflite::Interpreter> it; TfLiteDelegate* d = nullptr; int useB = 1;
    for (int b = B; b >= 1; b /= 2) { it = BuildGPU(*M[m].fb, &d, b); if (it) { useB = b; break; } if (b == 1) break; }
    if (!it) { fprintf(stderr, "GPU build fail %s\n", M[m].name.c_str()); return INF; }
    dels.push_back(d);
    int full = n / useB, rem = n % useB;
    if (full > 0) { gpu_sched.push_back({it.get(), full}); gpu_interps.push_back(std::move(it)); }
    if (rem > 0) {
      TfLiteDelegate* d2 = nullptr; auto it2 = BuildGPU(*M[m].fb, &d2, rem);
      if (!it2) { fprintf(stderr, "GPU rem build fail %s\n", M[m].name.c_str()); return INF; }
      dels.push_back(d2); gpu_sched.push_back({it2.get(), 1}); gpu_interps.push_back(std::move(it2));
    }
  }
  // CPU/SPLIT은 긴 것 먼저 (LPT 근사)
  std::sort(pool_list.begin(), pool_list.end(), [&](int a, int b) {
    double ca = (dev[a] == M_CPU) ? M[inst[a].model].c_eff : M[inst[a].model].split_cpu;
    double cb = (dev[b] == M_CPU) ? M[inst[b].model].c_eff : M[inst[b].model].split_cpu;
    return ca > cb;
  });

  // GPU 작업이 있으면 구동용 코어 1개 예약 (CPU 풀은 P-1)
  int cpu_cores = has_gpu ? std::max(1, P - 1) : P;
  int nw = std::min(cpu_cores, (int)pool_list.size());
  std::atomic<int> next(0);
  std::vector<double> worker_end(std::max(nw, 1), 0.0);
  double t0 = NowMs();
  auto worker = [&](int wi) {
    int k;
    while ((k = next.fetch_add(1)) < (int)pool_list.size()) pool_it[pool_list[k]]->Invoke();
    worker_end[wi] = NowMs() - t0;
  };
  std::vector<std::thread> ws;
  for (int i = 0; i < nw; ++i) ws.emplace_back(worker, i);
  for (auto& pr : gpu_sched) for (int r = 0; r < pr.second; ++r) pr.first->Invoke();  // 메인 = GPU 직렬(배칭)
  double gpu_side = has_gpu ? NowMs() - t0 : 0.0;
  for (auto& w : ws) w.join();
  double ms = NowMs() - t0;
  double cpu_side = pool_list.empty() ? 0.0 : *std::max_element(worker_end.begin(), worker_end.end());
  if (gpu_side_out) *gpu_side_out = gpu_side;
  if (cpu_side_out) *cpu_side_out = cpu_side;

  gpu_interps.clear(); pool_it.clear();
  for (auto* d : dels) TfLiteGpuDelegateV2Delete(d);
  return ms;
}

// ---------------------------------------------------------------------------
//  split_profile.csv 로드 → 모델별 최적 분할 방향/stage
// ---------------------------------------------------------------------------
struct SplitRow { int k; double cpu, gpu; bool has=false; };
static void LoadSplitProfile(const std::string& path,
                             std::map<std::string, std::map<std::string, SplitRow>>& out) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return;
  char line[512]; bool header = true;
  while (std::fgets(line, sizeof(line), f)) {
    if (header) { header = false; continue; }
    std::string L(line); std::vector<std::string> c; size_t p = 0;
    for (int i = 0; i < 9; ++i) { size_t q = L.find(',', p); c.push_back(L.substr(p, q-p)); if (q==std::string::npos) break; p = q+1; }
    if (c.size() < 5) continue;
    SplitRow r; r.k = std::atoi(c[2].c_str()); r.cpu = std::atof(c[3].c_str()); r.gpu = std::atof(c[4].c_str()); r.has = true;
    out[c[0]][c[1]] = r;
  }
  std::fclose(f);
}

int main(int argc, char* argv[]) {
  std::string split_profile = "split_profile.csv";
  int P = (int)std::thread::hardware_concurrency(); if (P < 1) P = 1;
  int R = 5;
  int batch = 16;              // GPU 배칭 묶음 크기 (1이면 배칭 끔)
  bool enable_split = false;   // fp32에선 split이 negative-sum + 실행 불일치 → 기본 off (프레임엔 유지)
  std::vector<std::pair<std::string,int>> spec;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--split-profile") == 0 && i+1 < argc) split_profile = argv[++i];
    else if (std::strcmp(argv[i], "--cpu-cores") == 0 && i+1 < argc) { P = std::atoi(argv[++i]); if (P<1) P=1; }
    else if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc) { R = std::atoi(argv[++i]); if (R<1) R=1; }
    else if (std::strcmp(argv[i], "--batch") == 0 && i+1 < argc) { batch = std::atoi(argv[++i]); if (batch<1) batch=1; }
    else if (std::strcmp(argv[i], "--enable-split") == 0) enable_split = true;
    else {
      std::string s = argv[i]; int cnt = 1;
      size_t c = s.find_last_of(':');
      if (c != std::string::npos && c+1 < s.size() && s.find_first_not_of("0123456789", c+1) == std::string::npos) {
        cnt = std::atoi(s.c_str()+c+1); s = s.substr(0, c);
      }
      spec.emplace_back(s, cnt < 1 ? 1 : cnt);
    }
  }
  if (spec.empty()) {
    fprintf(stderr, "Usage: %s --split-profile split_profile.csv <model.tflite:count> ... [--cpu-cores P] [--runs R]\n", argv[0]);
    return 1;
  }

  std::map<std::string, std::map<std::string, SplitRow>> sp;
  LoadSplitProfile(split_profile, sp);

  // 모델 로드 + 인스턴스 펼치기
  std::vector<ModelInfo> M; std::vector<Instance> inst;
  for (auto& s : spec) {
    ModelInfo mi; mi.path = s.first; mi.name = BaseName(s.first);
    mi.fb = tflite::FlatBufferModel::BuildFromFile(s.first.c_str());
    if (!mi.fb) { fprintf(stderr, "load fail %s\n", s.first.c_str()); return 1; }
    // split 후보: 방향별 min max(cpu,gpu)
    auto it = sp.find(mi.name);
    if (it != sp.end()) {
      double best = INF;
      for (const char* dn : {"CPU-first", "GPU-first"}) {
        auto r = it->second.find(dn);
        if (r == it->second.end() || !r->second.has) continue;
        double key = std::max(r->second.cpu, r->second.gpu);
        if (key < best) { best = key; mi.split_ok = true;
          mi.split_dir = (std::strcmp(dn,"CPU-first")==0) ? CPU_FIRST : GPU_FIRST;
          mi.split_k = r->second.k; mi.split_cpu = r->second.cpu; mi.split_gpu = r->second.gpu; }
      }
    }
    int idx = (int)M.size(); M.push_back(std::move(mi));
    for (int k = 0; k < s.second; ++k) inst.push_back({idx});
  }

  // 캘리브레이션 (모델별 g_eff=배칭 per-task, c_eff=경합 per-task)
  printf("[CALIB] P=%d, runs=%d, batch=%d  (실측, 정적표·α 미사용)\n", P, R, batch);
  for (auto& m : M) {
    int b1 = 1, uB = 1;
    m.g1   = CalibG(*m.fb, 1, R, &b1);           // batch=1 per-task
    double gB = CalibG(*m.fb, batch, R, &uB);    // 배칭 per-task
    m.gpu_ok = (m.g1 != INF);
    // 배칭이 유의미하게 빠를 때만 채택(가벼운 모델). 무거운 모델은 이득 ~0 + VRAM 위험 → batch=1.
    if (gB != INF && gB < 0.85 * m.g1) { m.g_eff = gB; m.bB = uB; }
    else                               { m.g_eff = m.g1; m.bB = 1; }
    m.c_eff = CalibC(*m.fb, P, R);
    m.split_ok = enable_split && m.split_ok && m.split_cpu > 0.02 && m.split_gpu < m.g_eff;
    printf("  %-28s g1=%.3f  g_eff=%.3f(B=%d)  c_eff=%.3f  split=%s\n",
           m.name.c_str(), m.g1, m.g_eff, m.bB, m.c_eff, m.split_ok?"y":"n");
  }

  // 배정
  std::vector<char> dev = Assign(inst, M, P);
  std::vector<char> all_gpu(inst.size(), M_GPU);
  for (size_t i = 0; i < inst.size(); ++i) if (!M[inst[i].model].gpu_ok) all_gpu[i] = M_CPU;

  int cg=0,cc=0,cs=0; for(char x:dev){ if(x==M_GPU)cg++; else if(x==M_CPU)cc++; else cs++; }
  printf("\n[ASSIGN] GPU=%d  CPU=%d  SPLIT=%d  (총 %zu)\n", cg, cc, cs, inst.size());

  // 실측: 3개 비교 — naive(ALL-GPU,배치1) / batched(ALL-GPU,배치B) / UNIFIED(배정+배칭)
  printf("running (build+warmup then timed) ...\n");
  double sg=0, sc=0;
  double ms   = RunScenario(inst, M, dev,     P, true,  &sg, &sc);          // 배정 + 배칭
  double ag1  = RunScenario(inst, M, all_gpu, P, false, nullptr, nullptr);  // naive 전부 GPU, 배치 1
  double agB  = RunScenario(inst, M, all_gpu, P, true,  nullptr, nullptr);  // 전부 GPU, 배칭만

  double bal = (std::max(sg,sc) > 1e-6) ? std::min(sg,sc)/std::max(sg,sc)*100.0 : 100.0;
  printf("\n================ RESULT (makespan, ms) ================\n");
  printf("  ALL-GPU  naive (batch=1)   : %8.2f   <- 기준선\n", ag1);
  printf("  ALL-GPU  batched (B=%d)     : %8.2f   (배칭 기여: %.2fx)\n", batch, agB, ag1/agB);
  printf("  UNIFIED  (배정+배칭)        : %8.2f   [GPU측 %.1f / CPU측 %.1f]\n", ms, sg, sc);
  printf("  사용률 균형 (min/max)       : %.0f%%\n", bal);
  printf("  speedup vs naive           : %.2fx   (배칭+배정 합산)\n", ag1 / ms);
  printf("======================================================\n");
  printf("SUMMARY,%zu,%d,%d,%d,%.2f,%.2f,%.2f,%.3f,%.1f,%.1f,%.0f\n",
         inst.size(), cg, cc, cs, ag1, agB, ms, ag1/ms, sg, sc, bal);
  return 0;
}
