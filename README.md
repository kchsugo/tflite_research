# tflite_research

Experiments on **heterogeneous CPU/GPU execution in TensorFlow Lite** — from a plain baseline,
through per-layer profiling and hybrid CPU↔GPU pipelines (memcpy vs. zero‑copy), to a single‑interpreter
**internal split** that partitions a model across CPU and GPU *without pre-splitting the `.tflite` file*.

Each folder is a small, self‑contained `minimal.cc` + `CMakeLists.txt` that builds against the
TensorFlow Lite source tree and runs one focused experiment.

---

## Motivation

A TFLite model normally runs entirely on one backend. Offloading part of a network to the GPU while
keeping the rest on the CPU (heterogeneous execution) can be faster, but it raises two questions this
repo works through step by step:

1. **How do we measure** where time goes, per layer, on CPU vs. GPU?
2. **How do we move activations across the CPU/GPU boundary cheaply** — and can we avoid the copy entirely (zero‑copy)?
3. **Can the split happen inside a single interpreter**, choosing the best cut point automatically, instead of manually slicing the model into two files?

The experiments below build toward the last question.

---

## Repository structure

| Folder | Backend / mode | What it does |
| --- | --- | --- |
| `base_minimal_cmake/` | CPU, no delegate | The stock TFLite “minimal” example — load a model, allocate, invoke, print interpreter state. Reference baseline. |
| `layer_cpu_test/` | CPU only | Per‑layer latency profiling on CPU via a custom `tflite::Profiler`. |
| `layer_gpu_test/` | CPU + GPU delegate | One binary that (1) profiles the model per layer on CPU, then (2) applies the GPU delegate and prints the execution plan (which nodes land on GPU vs. CPU) with averaged latency. |
| `cpu_gpu_memcpy_test/` | Hybrid, CPU→GPU | Two models / two interpreters. Run the CPU part, `memcpy` its output into the GPU part’s input, run the GPU part. Baseline hybrid transfer. |
| `gpu_cpu_memcpy_test/` | Hybrid, GPU→CPU | Same idea in the opposite order (GPU first, then CPU), using manual `memcpy` (no zero‑copy). |
| `cpu_gpu_zerocopy_test/` | Hybrid, zero‑copy | Makes the second interpreter’s input tensor **directly reference** the first interpreter’s output buffer via `SetCustomAllocationForTensor`, removing the boundary `memcpy`. Verifies pointer identity and benchmarks the pipeline. |
| `cpufirst_gpu_zerocopy_test/` | Hybrid, zero‑copy (CPU→GPU) | Zero‑copy variant that invokes the CPU part first, then the GPU part. |
| `internal_split_test/` | **Single interpreter, internal split** | Loads **one** original `.tflite`, keeps nodes `0..k-1` on CPU and delegates nodes `k..` to the GPU inside a single interpreter, then sweeps `k` to find the fastest cut. No file pre‑splitting, no user‑side memcpy. |
| `models/` | — | `mobilenet_v2_full.tflite` sample model used by the experiments. |

---

## The experiments, in order

### 1. Baseline — `base_minimal_cmake/`
The unmodified TFLite minimal example. Loads a model, runs one inference, and dumps the interpreter
state. No delegate, no profiling — the starting point everything else diverges from.

```bash
./minimal <model.tflite>
```

### 2. Per‑layer CPU profiling — `layer_cpu_test/`
Attaches a custom `SimpleProfiler` (subclass of `tflite::Profiler`) to record the wall‑clock time of
each operator, then prints an `ID | Op | Time (us)` table plus a total. XNNPACK is intentionally kept
out of the way so the per‑op timings reflect the reference CPU kernels.

```bash
./minimal <model.tflite>
```

### 3. CPU vs. GPU per‑layer view — `layer_gpu_test/`
A single binary with two phases:

* **[1] CPU‑only profiling** — per‑layer timing table (as above).
* **[2] GPU delegate** — applies `TfLiteGpuDelegateV2`, walks the execution plan, and marks each node
  `GPU` (absorbed into the delegate kernel) or `CPU` (fallback), then reports warmup + averaged GPU latency.

This is where you can see how many ops the GPU delegate fuses and whether any nodes fall back to CPU.
(`about`: the model also contains CPU‑only operators.)

```bash
./minimal <model.tflite>
```

### 4. Hybrid transfer with memcpy — `cpu_gpu_memcpy_test/`, `gpu_cpu_memcpy_test/`
Splits work across **two interpreters** (a CPU model and a GPU model) and connects them by copying the
first stage’s output into the second stage’s input with `memcpy`. `cpu_gpu_*` runs CPU→GPU; `gpu_cpu_*`
runs GPU→CPU. These quantify the cost of the naive boundary copy.

```bash
# CPU part first, then GPU part
./minimal <cpu_model.tflite> <gpu_model.tflite>
```

### 5. Zero‑copy transfer — `cpu_gpu_zerocopy_test/`, `cpufirst_gpu_zerocopy_test/`
Removes the boundary copy by giving the downstream interpreter a **custom allocation** that points at the
upstream interpreter’s output buffer (`SetCustomAllocationForTensor`). The program checks that the two
tensor pointers are identical (`✅ ZERO‑COPY`), falls back to `memcpy` if not, and benchmarks GPU / CPU /
end‑to‑end pipeline latency over multiple runs.

```bash
# cpu_gpu_zerocopy_test expects the GPU part first, then the CPU part:
./minimal <gpu_part.tflite> <cpu_part.tflite>
```

### 6. Internal split (current direction) — `internal_split_test/`
Instead of pre‑splitting the model into two files, this loads **one** original `.tflite` into **one**
interpreter and sets the GPU delegate option `first_delegate_node_index = k`. Nodes `0..k-1` stay on the
CPU; nodes `k..` become the GPU partition. Because everything lives in a single arena, the boundary
tensor is shared automatically — there is no user‑side `memcpy` or zero‑copy bookkeeping. The tool can
sweep `k` and report the cut point with the lowest latency.

```bash
./minimal <model.tflite>            # sweep every cut point, report the best k
./minimal <model.tflite> <k>        # single cut at k: per-node assignment + boundary bytes + latency
./minimal <model.tflite> sweep <s>  # sweep with step s
```

> **Build requirement:** `first_delegate_node_index` lives behind `#ifdef TFLITE_DEBUG_DELEGATE`.
> The `CMakeLists.txt` therefore calls `add_compile_definitions(TFLITE_DEBUG_DELEGATE)` **before**
> `add_subdirectory(...)` so that the TFLite library and this executable are compiled with the *same*
> macro and the delegate‑options struct has a matching ABI. Changing this define forces a one‑time full
> rebuild of the GPU delegate.

---

## Building

Each experiment builds against the TensorFlow Lite source tree via `add_subdirectory`, so you need a
local checkout of TensorFlow.

**Prerequisites**
* A TensorFlow source checkout (provides `tensorflow/lite`)
* CMake ≥ 3.16 and a C++17 compiler
* For the GPU experiments: an OpenCL‑capable GPU + drivers (the TFLite GPU delegate uses OpenCL)
* XNNPACK is disabled and the GPU delegate is enabled in the CMake files

```bash
# 1) Get TensorFlow (once)
git clone https://github.com/tensorflow/tensorflow.git

# 2) Configure + build one experiment
cd tflite_research/internal_split_test
mkdir build && cd build
cmake -DTENSORFLOW_SOURCE_DIR=/absolute/path/to/tensorflow ..
cmake --build . -j

# 3) Run
./minimal ../../models/mobilenet_v2_full.tflite
```

Notes:
* If `TENSORFLOW_SOURCE_DIR` is not set, the CMake files fall back to a sibling `../tensorflow`
  (and `layer_gpu_test/` assumes it sits inside the TF tree). Set `-DTENSORFLOW_SOURCE_DIR=...` explicitly
  to be safe.
* The first build compiles TensorFlow Lite from source and can take a while.

---

## Key concepts referenced

* **GPU delegate execution plan** — after `ModifyGraphWithDelegate`, iterate `execution_plan()` and check
  `node.delegate != nullptr` to tell which nodes run on GPU vs. CPU.
* **memcpy vs. zero‑copy** — the boundary between two interpreters can be bridged by copying
  (`memcpy`, O(n) per step, 2×N memory) or by sharing the buffer pointer
  (`SetCustomAllocationForTensor`, zero copy, N memory).
* **`first_delegate_node_index` (internal split)** — a single option that places the CPU/GPU cut *inside*
  one interpreter, so the boundary tensor is shared in the arena and no manual transfer code is needed.

---

## Roadmap

The hybrid memcpy/zero‑copy experiments motivated the move to **internal split**: one file, one
interpreter, an automatically shared boundary, and an offline sweep to pick the optimal cut point `k`.
Next steps include sweeping on additional models (e.g. YOLOv4‑tiny, BiseNet) and extending beyond a
single contiguous cut toward arbitrary node sets via a custom delegate.

---

## Author

ChanHyung Kim — Department of AI Software, Soongsil University.
