#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/profiling/profiler.h"

struct LayerEvent {
    std::string tag;
    double elapsed_us;
};

class SimpleProfiler : public tflite::Profiler {
public:
    std::vector<LayerEvent> events;

    uint32_t BeginEvent(const char* tag, EventType event_type,
                        int64_t event_metadata1, int64_t event_metadata2) override {
        timestamps_.push_back(std::chrono::high_resolution_clock::now());
        tags_.push_back(tag ? tag : "");
        return timestamps_.size() - 1;
    }

    void EndEvent(uint32_t event_handle) override {
        auto end = std::chrono::high_resolution_clock::now();
        if (event_handle < timestamps_.size()) {
            double us = std::chrono::duration<double, std::micro>(
                end - timestamps_[event_handle]).count();
            events.push_back({tags_[event_handle], us});
        }
    }

private:
    std::vector<std::chrono::high_resolution_clock::time_point> timestamps_;
    std::vector<std::string> tags_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder builder(*model, resolver);
    builder.SetNumThreads(1);
    builder(&interpreter);

    // ★ XNNPACK delegate 비활성화
    interpreter->SetAllowFp16PrecisionForFp32(false);
    // 핵심: XNNPACK이 자동 적용되지 않도록 환경변수 또는 빌드 옵션 필요

    SimpleProfiler profiler;
    interpreter->SetProfiler(&profiler);
    interpreter->AllocateTensors();

    float* input = interpreter->typed_input_tensor<float>(0);
    if (input) std::memset(input, 0, interpreter->tensor(interpreter->inputs()[0])->bytes);

    interpreter->Invoke();
    profiler.events.clear();

    if (interpreter->Invoke() != kTfLiteOk) { printf("Invoke failed!\n"); return 1; }

    printf("\n%-5s | %-25s | %12s\n", "Node", "Op", "Time (us)");
    printf("----------------------------------------------\n");
    double total = 0;
    for (size_t i = 0; i < profiler.events.size(); ++i) {
        printf("%-5zu | %-25s | %12.2f\n", i,
               profiler.events[i].tag.c_str(),
               profiler.events[i].elapsed_us);
        total += profiler.events[i].elapsed_us;
    }
    printf("----------------------------------------------\n");
    printf("%-5s | %-25s | %12.2f\n", "", "TOTAL", total);

    return 0;
}
