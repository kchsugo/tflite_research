// ============================================================================
//  배칭 측정 (batch_test.cc)
//  각 모델의 입력을 batch 차원으로 B개 쌓아 GPU로 한 번에 Invoke → per-task 시간 측정.
//  목적: (1) ResizeInputTensor로 배치가 되는 모델인가, (2) 배칭하면 per-task가 얼마나 주나.
//  사용법: batch_test <m1.tflite> ... [--runs 10]
// ============================================================================
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

static std::string BaseName(const std::string& p){ size_t s=p.find_last_of("/\\"); return s==std::string::npos?p:p.substr(s+1); }
static double NowMs(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
static void FillZero(tflite::Interpreter* it){ for(int i:it->inputs()){ auto*t=it->tensor(i); if(t&&t->data.raw) std::memset(t->data.raw,0,t->bytes);} }

// B배치로 GPU 실행. 성공 시 per-Invoke 총시간(ms) 반환, 실패 시 -1.
static double RunBatch(const tflite::FlatBufferModel& m, int B, int runs) {
  tflite::ops::builtin::BuiltinOpResolver r;
  std::unique_ptr<tflite::Interpreter> it;
  tflite::InterpreterBuilder b(m, r); b.SetNumThreads(1);
  if (b(&it) != kTfLiteOk || !it) return -1;
  int in = it->inputs()[0];
  TfLiteTensor* t = it->tensor(in);
  if (!t || !t->dims || t->dims->size < 1) return -1;
  std::vector<int> dims(t->dims->data, t->dims->data + t->dims->size);
  dims[0] = B;                                   // batch 차원 교체
  if (it->ResizeInputTensor(in, dims) != kTfLiteOk) return -1;
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  TfLiteDelegate* d = TfLiteGpuDelegateV2Create(&o);
  if (it->ModifyGraphWithDelegate(d) != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return -1; }
  if (it->AllocateTensors() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return -1; }
  FillZero(it.get());
  for (int i = 0; i < 3; ++i) if (it->Invoke() != kTfLiteOk) { TfLiteGpuDelegateV2Delete(d); return -1; }
  double sum = 0;
  for (int i = 0; i < runs; ++i) { double t0 = NowMs(); if (it->Invoke()!=kTfLiteOk){TfLiteGpuDelegateV2Delete(d);return -1;} sum += NowMs()-t0; }
  it.reset(); TfLiteGpuDelegateV2Delete(d);
  return sum / runs;
}

int main(int argc, char* argv[]) {
  int runs = 10; std::vector<std::string> models;
  for (int i=1;i<argc;++i){ if(!std::strcmp(argv[i],"--runs")&&i+1<argc) runs=atoi(argv[++i]); else models.push_back(argv[i]); }
  if (models.empty()){ fprintf(stderr,"Usage: %s <m.tflite> ... [--runs 10]\n",argv[0]); return 1; }

  for (auto& path : models) {
    auto m = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!m){ fprintf(stderr,"load fail %s\n",path.c_str()); continue; }
    printf("\n=== %s ===\n", BaseName(path).c_str());
    printf("  %4s | %10s | %12s | %8s\n","B","total(ms)","per-task(ms)","speedup");
    double base = -1;
    for (int B : {1,2,4,8,16,32}) {
      double tot = RunBatch(*m, B, runs);
      if (tot < 0) { printf("  %4d |   (batch %d 미지원/실패)\n", B, B); continue; }
      double per = tot / B;
      if (base < 0) base = per;
      printf("  %4d | %10.2f | %12.3f | %7.2fx\n", B, tot, per, base/per);
    }
  }
  return 0;
}
