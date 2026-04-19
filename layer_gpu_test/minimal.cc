#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.tflite>\n", argv[0]);
        return 1;
    }

    auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    if (!model) { printf("Failed to load model\n"); return 1; }

    tflite::ops::builtin::BuiltinOpResolver resolver;

    // ========================================
    // 1) CPU-only 실행 (per-layer profiling)
    // ========================================
    printf("\n============================================================\n");
    printf("  [1] CPU-ONLY Per-Layer Profiling (no delegate)\n");
    printf("============================================================\n");
    {
        std::unique_ptr<tflite::Interpreter> cpu_interp;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder.SetNumThreads(1);
        builder(&cpu_interp);

        cpu_interp->AllocateTensors();

        float* input = cpu_interp->typed_input_tensor<float>(0);
        if (input) std::memset(input, 0, cpu_interp->tensor(cpu_interp->inputs()[0])->bytes);

        // Warmup
        for (int w = 0; w < 3; ++w) cpu_interp->Invoke();

        // 노드 정보 수집
        struct NodeInfo {
            int index;
            std::string op_name;
        };
        std::vector<NodeInfo> nodes;

        for (int i = 0; i < cpu_interp->nodes_size(); ++i) {
            auto* nr = cpu_interp->node_and_registration(i);
            auto& reg = nr->second;
            const char* name = "";
            if (reg.builtin_code == tflite::BuiltinOperator_CUSTOM) {
                name = reg.custom_name ? reg.custom_name : "CUSTOM";
            } else {
                name = tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg.builtin_code));
            }
            nodes.push_back({i, name ? name : "Unknown"});
        }

        // Per-layer 시간 측정: 각 노드를 개별 invoke 대신,
        // 전체 Invoke 1회의 총 시간 + 노드별 비율은 구할 수 없으므로
        // 전체 CPU 시간만 정확히 측정 후, profiler로 비율 파악

        // 방법: SimpleProfiler 사용
        struct LayerEvent { std::string tag; double us; };
        class SimpleProfiler : public tflite::Profiler {
        public:
            std::vector<LayerEvent> events;
            uint32_t BeginEvent(const char* tag, EventType event_type,
                                int64_t m1, int64_t m2) override {
                ts_.push_back(std::chrono::high_resolution_clock::now());
                tags_.push_back(tag ? tag : "");
                types_.push_back(event_type);
                return ts_.size() - 1;
            }
            void EndEvent(uint32_t h) override {
                auto end = std::chrono::high_resolution_clock::now();
                if (h < ts_.size()) {
                    double us = std::chrono::duration<double, std::micro>(end - ts_[h]).count();
                    events.push_back({tags_[h], us});
                }
            }
        private:
            std::vector<std::chrono::high_resolution_clock::time_point> ts_;
            std::vector<std::string> tags_;
            std::vector<EventType> types_;
        };

        SimpleProfiler profiler;
        cpu_interp->SetProfiler(&profiler);

        // Profiled run
        cpu_interp->Invoke();

        // "Invoke" / "invoke" 같은 래퍼 이벤트 제외하고 출력
        printf("\n%-5s | %-30s | %12s | %-6s\n", "ID", "Op", "Time (us)", "Device");
        printf("----------------------------------------------------------------------\n");
        double cpu_total = 0;
        int op_idx = 0;
        for (size_t i = 0; i < profiler.events.size(); ++i) {
            // 래퍼 이벤트 필터링 (대소문자 "Invoke", "invoke")
            if (profiler.events[i].tag == "Invoke" || profiler.events[i].tag == "invoke")
                continue;
            printf("%-5d | %-30s | %12.2f | CPU\n",
                   op_idx, profiler.events[i].tag.c_str(), profiler.events[i].us);
            cpu_total += profiler.events[i].us;
            op_idx++;
        }
        printf("----------------------------------------------------------------------\n");
        printf("%-5s | %-30s | %12.2f | CPU\n", "", "TOTAL", cpu_total);
        printf("CPU ops: %d\n", op_idx);
    }

    // ========================================
    // 2) GPU delegate 실행 (전체 시간 측정)
    // ========================================
    printf("\n============================================================\n");
    printf("  [2] GPU Delegate Profiling\n");
    printf("============================================================\n");
    {
        std::unique_ptr<tflite::Interpreter> gpu_interp;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder.SetNumThreads(1);
        builder(&gpu_interp);

        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);

        if (gpu_interp->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk) {
            printf("GPU delegate failed!\n");
            TfLiteGpuDelegateV2Delete(gpu_delegate);
            return 1;
        }

        gpu_interp->AllocateTensors();
        float* input = gpu_interp->typed_input_tensor<float>(0);
        if (input) std::memset(input, 0, gpu_interp->tensor(gpu_interp->inputs()[0])->bytes);

        // Execution Plan 출력
        printf("\n%-5s | %-30s | %-6s\n", "Node", "Op", "Device");
        printf("-----------------------------------------------------\n");
        int gpu_count = 0, cpu_count = 0;
        for (int i = 0; i < gpu_interp->nodes_size(); ++i) {
            auto* nr = gpu_interp->node_and_registration(i);
            auto& node = nr->first;
            auto& reg = nr->second;

            const char* name = "";
            if (reg.builtin_code == tflite::BuiltinOperator_CUSTOM) {
                name = reg.custom_name ? reg.custom_name : "CUSTOM";
            } else {
                name = tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg.builtin_code));
            }

            bool delegated = (node.delegate != nullptr);

            // 원본 op (delegate에 흡수됨)은 스킵, delegate 노드와 남은 CPU 노드만 출력
            if (delegated || (int)i >= (gpu_interp->nodes_size() - 1 - 0)) {
                // 간단히: delegate 노드만 따로 체크
            }

            if (delegated) gpu_count++; else cpu_count++;
        }

        // 실제 실행되는 노드만 출력 (delegate 노드 + CPU fallback 노드)
        // delegate에 흡수된 원본 op은 실행 안 됨
        // → delegate 노드는 node.delegate != nullptr 인 것
        // → CPU fallback은 delegate가 거부한 것 (이 모델에선 없음)
        for (int i = 0; i < gpu_interp->nodes_size(); ++i) {
            auto* nr = gpu_interp->node_and_registration(i);
            auto& node = nr->first;
            auto& reg = nr->second;

            bool delegated = (node.delegate != nullptr);
            // 원본 op 중 delegate에 흡수된 건 실행 안 됨 — 스킵
            // delegate가 새로 만든 노드(DELEGATE)만 + 남은 CPU 노드만 출력
            if (!delegated && i < gpu_interp->nodes_size() - gpu_count)
                continue; // 흡수된 원본 노드

            const char* name = "";
            if (reg.builtin_code == tflite::BuiltinOperator_CUSTOM) {
                name = reg.custom_name ? reg.custom_name : "CUSTOM";
            } else {
                name = tflite::EnumNameBuiltinOperator(
                    static_cast<tflite::BuiltinOperator>(reg.builtin_code));
            }

            printf("%-5d | %-30s | %-6s\n", i,
                   name ? name : "Unknown",
                   delegated ? "GPU" : "CPU");
        }
        printf("-----------------------------------------------------\n");

        // Warmup (5회 — GPU는 초기화 오버헤드 큼)
        for (int w = 0; w < 5; ++w) gpu_interp->Invoke();

        // 시간 측정 (10회 평균)
        const int RUNS = 10;
        double total_us = 0;
        for (int r = 0; r < RUNS; ++r) {
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_interp->Invoke();
            auto t2 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        }
        double avg_us = total_us / RUNS;

        printf("\nGPU Inference (%d runs avg): %.2f us (%.2f ms)\n",
               RUNS, avg_us, avg_us / 1000.0);

        TfLiteGpuDelegateV2Delete(gpu_delegate);
    }

    return 0;
}
