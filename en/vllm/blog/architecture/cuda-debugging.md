---
source: https://vllm.ai/blog/2025-08-11-cuda-debugging
lang: en
fetched: 2026-09-05
---

# CUDA Core Dump: An Effective Tool to Debug Memory Access Issues and Beyond

Chinese: [zh/vllm/blog/architecture/cuda-debugging.md](../../../../zh/vllm/blog/architecture/cuda-debugging.md)

2025-08-11. **Kaichao You**. Study note. No mechanism figures on the page (logo only; not copied locally). Follow-up — hanging kernels and source lines: [cuda-debugging-source](cuda-debugging-source.md).

**TL;DR from the page:** Hit `an illegal memory access was encountered`, enable CUDA core dump. Set the env vars below, rerun to collect the dump, then `cuda-gdb`.

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
```

## Introduction

CUDA kernel tests that hit illegal memory access (IMA) with no obvious next step. That pain showed up repeatedly while building vLLM.

This post is the lower-level debugging path they use for messy vLLM issues, including IMA. A typical PyTorch error:

```text
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

The hard part is that sentence: kernel errors may be reported asynchronously on **some other** API call, so the stack **might be wrong**. The authors’ experience: Python stacks for these exceptions are **almost always wrong and pretty worthless**. The message suggests `CUDA_LAUNCH_BLOCKING=1`. Two remaining problems:

1. Many launches use `kernel<<<>>>` without checking launch status — the page names [this PyTorch code](https://github.com/pytorch/pytorch/blob/5e320eea665f773b78f6d3bfdbb1898b8e09e051/aten/src/ATen/native/cuda/SortStable.cu#L117). Then even `CUDA_LAUNCH_BLOCKING=1` cannot name the bad kernel.
2. If the IMA is inside a kernel **in a CUDA graph**, `CUDA_LAUNCH_BLOCKING=1` only shows a problem at graph launch, still not which kernel.

Pinpointing this needs an immediate reaction when IMA happens. Users cannot do that themselves — the CUDA **driver** has to.

[CUDA core dump](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-core-dump-support) is that mechanism: dump GPU state when IMA occurs, then inspect which kernel and what the illegal access was.

## What is a core dump?

A GPU is a massively parallel processor; many ideas have CPU counterparts.

A [core dump](https://en.wikipedia.org/wiki/Core_dump) is CPU + OS: when a program crashes, the OS can record memory and runtime state for later analysis. A crash is hardware-level. The CPU hits a `trap`; the OS takes over (default: kill the process; optionally emit a core dump. Example: `ulimit -c 1`, and `echo "core.%e.%p" > /proc/sys/kernel/core_pattern` for the path).

GPU core dump needs GPU hardware and the GPU driver. When a GPU thread crashes, hardware raises an exception to the driver, which handles it immediately. Per [forum discussion](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389), the default is often to mark the current CUDA context unusable, **not** to terminate the process.

## How to enable CUDA core dump

Enabling is `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`. For a usable workflow, add:

1. By default the dump lands in the current directory with no printed path. `CUDA_COREDUMP_SHOW_PROGRESS=1` prints progress and, most importantly, the **file path** when done.
2. Container jobs often destroy the filesystem on failure. `CUDA_COREDUMP_FILE` sets a template onto persistent storage: `CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"`. `%h` hostname, `%p` PID, `%t` timestamp.
3. Default dump saves the **entire** GPU context. LLM inference that fills VRAM makes a full dump impractical (hundreds of GiB). `CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'` skips those memories and shrinks the file. `skip_constbank_memory` is **missing from the docs** but supported; it can be necessary [when many GPU threads hit errors at once](https://forums.developer.nvidia.com/t/cuda-core-dump-does-not-work-properly-when-many-device-assert-happens/342410).

The docs also mention `skip_abort` in `CUDA_COREDUMP_GENERATION_FLAGS` so the CPU process does not abort after the dump, letting it print its own trace. Experiments found a serious [bug](https://forums.developer.nvidia.com/t/cuda-core-dump-with-skip-abort-will-ignore-an-illegal-memory-access-error/341802/3): GPU IMA can be **ignored**, later code keeps running, memory may already be corrupt. Unacceptable for training, undesirable for inference. Treat it as **unreliable; do not use**.

The docs say `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` also generates a CPU coredump by default. In practice that CPU dump has little useful information and is hard to analyze.

For **live** state: `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` — no core dump; GPU execution stops immediately and waits for a debugger (`cuda-gdb`) while full GPU memory is still intact. Less automatic; more hands.

Recommended combo:

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"
```

## Examples

Code to check that CUDA core dump actually helps.

### Debugging improper kernel launch

```cpp
// test.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define cuda_check(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void illegalMemoryAccessKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * 2) {
        for (int i = 0; i < 10000; i++) {
            data[idx - 1000000000 + i] = idx;   // illegal for idx == 0
        }
    }
}

__global__ void normalKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = idx;
    }
}

int main() {
    printf("CUDA Illegal Memory Access Test\n");
    int size = 100;
    int* h_data = (int*)malloc(size * sizeof(int));
    int* d_data;
    for (int i = 0; i < size; i++) h_data[i] = 0;
    cuda_check(cudaMalloc(&d_data, (unsigned long long)(size) * sizeof(int)));
    cuda_check(cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    printf("Launching kernel with out-of-bounds access...\n");
    illegalMemoryAccessKernel<<<numBlocks, blockSize>>>(d_data, size);
    normalKernel<<<numBlocks, blockSize>>>(d_data, size);
    cuda_check(cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]);
    printf("\n");
    cuda_check(cudaDeviceSynchronize());
    printf("Test completed.\n");
    cuda_check(cudaFree(d_data));
    free(h_data);
    return 0;
}
```

Two kernels launch back to back (`illegalMemoryAccessKernel` then `normalKernel`). The error is `CUDA Error at test.cu:62 - cudaMemcpy(...): an illegal memory access was encountered` — first visible on `cudaMemcpy`. Even with `CUDA_LAUNCH_BLOCKING=1`, you still cannot name the kernel.

With the core-dump env vars:

```text
[06:43:15.209195] coredump: Detected an exception of type CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS (14)
[06:43:15.209202] coredump:   - Device: 0
[06:43:15.209206] coredump:   - SM: 124
[06:43:15.209208] coredump:   - Warp: 0
[06:43:15.209210] coredump:   - PC 0x7462c3bac310
[06:43:15.209477] coredump: Stack trace (lane masks: active 0xFFFFFFFF, valid 0xFFFFFFFF):
[06:43:15.209486] coredump:   #0	0x7462c3bac620	_Z25illegalMemoryAccessKernelPii

[00:40:46.806153] coredump: Writing ELF file to /tmp/cuda_coredump_xxx.1799919.1754898045

[1]    1799919 IOT instruction (core dumped)  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 = = ./test3
```

After a GPU thread hits IMA, the CPU writes a coredump immediately, then a CPU exception kills the process. File: `/tmp/cuda_coredump_xxx.1799919.1754898045`. Open with `cuda-gdb` (`target cudacore /path/to/coredump_file`; `cudacore` means a CUDA dump):

```bash
$ cuda-gdb
(cuda-gdb) target cudacore /tmp/cuda_coredump_xxx.1799919.1754898045
Opening GPU coredump: /tmp/cuda_coredump_xxx.1799919.1754898045

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7f31abb9f6d0  illegalMemoryAccessKernel(int*, int)
[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0]
#0  0x00007f31abb9f6e0 in illegalMemoryAccessKernel(int*, int)<<<(1,1,1),(256,1,1)>>> ()
```

The exception is `illegalMemoryAccessKernel` at `kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0`.

### Debugging kernel exceptions in CUDA graphs

A harder case: an IMA kernel inside a CUDA graph.

```python
# core_dump.py
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class CupyWrapper:
    data_ptr: int
    size_in_bytes: int

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size_in_bytes,),
            "typestr": '|u1',
            "data": (self.data_ptr, False),
            "version": 3,
        }

def from_buffer(data_ptr: int, size_in_bytes: int) -> torch.Tensor:
    out = torch.as_tensor(CupyWrapper(data_ptr, size_in_bytes))
    assert data_ptr == out.data_ptr(), "not zero-copy convert, something must be wrong!"
    return out


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 30)
        self.num_called = 0

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        self.num_called += 1
        if self.num_called > 1:
            y = from_buffer(x.data_ptr(), x.numel() * 1024 * 1024)
            y.fill_(1)  # triggers IMA
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 10).to(device)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input device: {input_tensor.device}")
    with torch.no_grad():
        output = model(input_tensor)  # warmup
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = model(input_tensor)
        g.replay()
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Output: {output.sum()}")
    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    print(f"Model device: {next(model.parameters()).device}")
```

Direct run only fails when `output.sum()` synchronizes the device:

```text
Using device: cuda
Input shape: torch.Size([4, 10])
Input device: cuda:0
Output shape: torch.Size([4, 30])
Output device: cuda:0
Traceback (most recent call last):
  File "core_dump.py", line 76, in <module>
    print(f"Output: {output.sum()}")
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Which kernel? CUDA kernels run asynchronously.

With `CUDA_LAUNCH_BLOCKING=1`, the error moves to `g.replay()`:

```text
Using device: cuda
Input shape: torch.Size([4, 10])
Input device: cuda:0
Traceback (most recent call last):
  File "core_dump.py", line 71, in <module>
    g.replay()
  File "/uv_envs/py310/lib/python3.10/site-packages/torch/cuda/graphs.py", line 88, in replay
    super().replay()
RuntimeError: CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

A kernel inside the graph failed. Conventional tools stop there.

With the recommended env vars, `cuda-gdb` names the kernel:

```text
(cuda-gdb) target cudacore /tmp/cuda_coredump_flow-matic.1929094.1754901120
Opening GPU coredump: /tmp/cuda_coredump_flow-matic.1929094.1754901120

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7fc2afba5e30  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)
[Current focus set to CUDA kernel 0, grid 9, block (17454,0,0), thread (0,0,0), device 0, sm 0, warp 1, lane 0]
#0  0x00007fc2afba5e70 in void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)<<<(40960,1,1),(128,1,1)>>> ()
```

A `fill`, grid size **40960** — suspiciously large. In source: `y = from_buffer(x.data_ptr(), x.numel() * 1024 * 1024); y.fill_(1);` stretches `x` by a million and fills with 1s → IMA.

On some GPUs this line is `invalid argument` instead of IMA, because the grid exceeds the limit. Then CUDA core dump **does not fire**. Turn the expansion factor `1024 * 1024` down a bit so the grid stays legal, then dump again.

## Limitations and considerations

1. In theory CUDA core dump should catch various per-thread GPU exceptions. In practice, on some GPU/driver versions, errors like `operation not supported on global/shared address space` may **fail** to dump. IMA generally does dump, which covers most debugging.
2. Hardware errors such as `Invalid access of peer GPU memory over nvlink or a hardware error` are not caused by a specific thread and **will not** dump.
3. Misuse of the driver API is a [non-sticky error](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389), unrelated to the GPU itself, reported at the driver-API layer, no dump. Common case: OOM on `cudaMalloc`.
4. Multi-GPU programs often map another GPU’s memory. If that process exits, the mapping is invalid and access reports IMA — **not** a typical IMA. Common during distributed shutdown if GPUs are still communicating. Separate these false positives when using core dump.
5. Enabling core dump has a **performance cost** on CUDA kernels (check and attribute errors when threads exit). Do **not** leave it on in production. Enable after IMA is reliably reproducible, for debugging.
6. To map back to source, rebuild vLLM with debug symbols, or at least line info. The default binary omits this for size. Compile from source ([full GPU build](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#full-build-with-compilation)) with `export NVCC_PREPEND_FLAGS='-lineinfo'` or `export NVCC_PREPEND_FLAGS='-G'`. Start with `-lineinfo`; switch to `-G` only if that is not enough. With rich debug info, the dump can name the exact line. That continuation is [cuda-debugging-source](cuda-debugging-source.md).

## Conclusion

Principles and use cases for CUDA core dump: improper launches and kernel exceptions inside CUDA graphs. A strong tool for IMA and beyond.

They used it on a messy vLLM IMA: [PR #22593](https://github.com/vllm-project/vllm/pull/22593). A [Triton MRope kernel](https://github.com/vllm-project/vllm/pull/22375) assumed `head_size==rotary_dim` (full RoPE). When `head_size!=rotary_dim` (partial RoPE) it IMA’d — the case for [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V). Without dump the error was `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:453 'an illegal memory access was encountered'`, badly misleading. With dump it pointed at the MRope kernel, then a fix. That case was **misconfigured kernel parameters**; finding the kernel was enough. For harder IMA, isolate a minimal repro and use [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool).

vLLM wants easy, fast, cheap LLM serving; easy debugging is part of that. More debugging notes later. Stories: PR on the [blog repo](https://github.com/vllm-project/vllm-project.github.io).

## Acknowledgement

Ze Long, Vikram Sharma Mailthody, Jeremy Iverson, and Sandarbh Jain (NVIDIA) for discussions. Lucas Wilkinson (Red Hat) for polishing the draft.
