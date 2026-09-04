---
source: https://vllm.ai/blog/2025-08-11-cuda-debugging
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# CUDA core dump：illegal memory access 落到哪只 kernel

英文对照：[en/vllm/blog/architecture/cuda-debugging.md](../../../../en/vllm/blog/architecture/cuda-debugging.md)  
原文：https://vllm.ai/blog/2025-08-11-cuda-debugging  
2025-08-11。署名 **Kaichao You**。学习笔记。原文没有机制图（只有 logo，本地不搬）。续篇——挂死的 kernel、对回源码行：[cuda-debugging-source](cuda-debugging-source.md)。

**原文 TL;DR：** 撞上 `an illegal memory access was encountered`，打开 CUDA core dump。设下面这些环境变量，再跑一遍收集 coredump，然后用 `cuda-gdb` 看。

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t"
```

## 引言

写 CUDA kernel 时，测试动不动就 illegal memory access（后文 **IMA**），却不知道从哪查。vLLM 这种高性能推理引擎里，这痛反复出现。

这篇写的是他们用来查复杂问题（包括 IMA）的一套偏底层的办法。PyTorch 会抛这种错：

```text
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

难处就在那句：CUDA kernel 错误可能在 **别的** API 调用上异步报出来，下面的栈 **不一定对**。作者经验：这类异常的 Python 栈 **几乎总是错的，基本没用**。报错建议加 `CUDA_LAUNCH_BLOCKING=1`。仍有两道坎：

1. 许多人用 `kernel<<<>>>` 启动，却不去查 launch 状态——原文点名 [这段 PyTorch](https://github.com/pytorch/pytorch/blob/5e320eea665f773b78f6d3bfdbb1898b8e09e051/aten/src/ATen/native/cuda/SortStable.cu#L117)。这种情况下，即便 `CUDA_LAUNCH_BLOCKING=1`，也定位不到坏掉的那只 kernel。
2. IMA 若发生在 **CUDA graph 里的 kernel**，`CUDA_LAUNCH_BLOCKING=1` 也只能告诉你 graph launch 出问题，仍然点不出是哪一只。

要准，得在 IMA **当场** 反应。这不是用户代码能做的——要 CUDA **驱动** 自己支持。

[CUDA core dump](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-core-dump-support) 就是干这个的：illegal memory access 发生时，驱动把 GPU 状态 dump 下来，事后再查是哪只 kernel、怎样越界。

## 什么是 core dump

GPU 是大规模并行处理器，许多概念能在 CPU 上找到对应物。

[Core dump](https://en.wikipedia.org/wiki/Core_dump) 是 CPU 和操作系统一起提供的：程序崩了，OS 可以记下内存、运行时状态，留给后面分析。崩溃是硬件层的事。CPU 执行某些指令出错，进入 `trap`；OS 接手，跑异常处理（默认直接杀进程；也可以配成生成 core dump。例如 `ulimit -c 1` 打开，`echo "core.%e.%p" > /proc/sys/kernel/core_pattern` 指定路径）。

GPU 上的 core dump 要 GPU 硬件和 GPU 驱动合作。某条 GPU 线程崩了，硬件抛异常给驱动，驱动立刻处理。但按 [论坛讨论](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389)，驱动默认往往只是把当前 CUDA context 标成不可用，**并不**终止进程。

## 怎样打开 CUDA core dump

打开本身很简单：`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`。要好用，再加几项：

1. 默认把 coredump 写在当前目录，还不打路径。`CUDA_COREDUMP_SHOW_PROGRESS=1` 会打印进度和细节；最要紧的是结束后打出 **文件路径**，后面才找得到。
2. 很多任务跑在容器里，失败容器就拆，文件留不住。用 `CUDA_COREDUMP_FILE` 指定模板，写到持久盘：`CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"`。`%h` 主机名，`%p` PID，`%t` dump 时间戳。
3. 默认会存 **整份** GPU context。大模型推理几乎占满显存，完整 dump 不现实（几百 GiB）。`CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'` 跳过 GPU / shared / local / constbank 内存，把文件缩小。`skip_constbank_memory` **文档没写**，实际支持；[许多 GPU 线程同时炸](https://forums.developer.nvidia.com/t/cuda-core-dump-does-not-work-properly-when-many-device-assert-happens/342410) 时常常用得上。

文档还提到：在 `CUDA_COREDUMP_GENERATION_FLAGS` 里加 `skip_abort`，dump 完 CPU 进程不 abort，好让 CPU 自己再打一份错误栈。实验下来有明显 [bug](https://forums.developer.nvidia.com/t/cuda-core-dump-with-skip-abort-will-ignore-an-illegal-memory-access-error/341802/3)：GPU 上的 IMA 可能被 **吞掉**，后面的代码继续跑，内存可能已经坏了。训练不可接受，推理也不想要。这面旗 **一般不可靠，不推荐**。

文档还说打开 `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` 默认也会生成 CPU coredump。实践里 CPU dump 几乎没有有用信息，也难分析。

若要 **活着的** 现场：`CUDA_DEVICE_WAITS_ON_EXCEPTION=1`——不用 core dump，异常时 GPU 立刻停住，等你 attach（如 `cuda-gdb`），整卡显存还在。自动化差，要人手。

推荐组合：

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"
```

## 例子

用代码核一下 CUDA core dump 管不管用。

### 不规范的 kernel launch

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
            data[idx - 1000000000 + i] = idx;   // idx == 0 时越界
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

连续 launch 两只 kernel（`illegalMemoryAccessKernel` 和 `normalKernel`）。报错会是：`CUDA Error at test.cu:62 - cudaMemcpy(...): an illegal memory access was encountered`——错在 `cudaMemcpy` 的返回值上才被看见。即便 `CUDA_LAUNCH_BLOCKING=1`，也点不出是哪一只 kernel。

加上 core dump 相关环境变量之后，能看到类似：

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

GPU 线程一触发 IMA，CPU 立刻写 coredump，再触发 CPU 异常，进程直接死。得到 `/tmp/cuda_coredump_xxx.1799919.1754898045`。用 `cuda-gdb` 打开（`target cudacore /path/to/coredump_file`，`cudacore` 指 CUDA 上的 coredump）：

```bash
$ cuda-gdb
(cuda-gdb) target cudacore /tmp/cuda_coredump_xxx.1799919.1754898045
Opening GPU coredump: /tmp/cuda_coredump_xxx.1799919.1754898045

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7f31abb9f6d0  illegalMemoryAccessKernel(int*, int)
[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0]
#0  0x00007f31abb9f6e0 in illegalMemoryAccessKernel(int*, int)<<<(1,1,1),(256,1,1)>>> ()
```

异常来自 `illegalMemoryAccessKernel`：`kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0`。

### CUDA graph 里的 kernel 异常

更绕的例子：把会 IMA 的 kernel 塞进 CUDA graph。

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
            y.fill_(1)  # 触发 IMA
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

直接跑，错要等到 `output.sum()` 触发设备同步才冒出来：

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

不知道是哪只 kernel：CUDA kernel 异步执行。

加上 `CUDA_LAUNCH_BLOCKING=1`，错挪到 `g.replay()`：

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

能推断 CUDA graph 里某只 kernel 炸了。常规办法到此为止。

加上推荐的那组环境变量之后，`cuda-gdb` 能点出 kernel：

```text
(cuda-gdb) target cudacore /tmp/cuda_coredump_flow-matic.1929094.1754901120
Opening GPU coredump: /tmp/cuda_coredump_flow-matic.1929094.1754901120

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7fc2afba5e30  void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)
[Current focus set to CUDA kernel 0, grid 9, block (17454,0,0), thread (0,0,0), device 0, sm 0, warp 1, lane 0]
#0  0x00007fc2afba5e70 in void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)<<<(40960,1,1),(128,1,1)>>> ()
```

这是一只 `fill`，grid **40960** 大得不正常。对照源码：`y = from_buffer(x.data_ptr(), x.numel() * 1024 * 1024); y.fill_(1);` 把 `x` 的长度硬放大一百万倍再填 1，于是 IMA。

有的 GPU 上这行会变成 `invalid argument` 而不是 IMA——grid 超过上限。那种情况 **触发不了** CUDA core dump；把放大因子 `1024 * 1024` 略降一点，避开 grid 上限，才能再走 dump。

## 限制

1. 理论上，某条 GPU 线程引起的多种异常都该能抓住。实践里，某些 GPU / 驱动版本上，`operation not supported on global/shared address space` 一类 **未必** 触发 dump。幸好 IMA 一般能可靠触发，够用大多数调试。
2. 硬件类错误，例如 `Invalid access of peer GPU memory over nvlink or a hardware error`，不是某条线程引起的，归不到某一 GPU 线程，**不会** dump。
3. 用错 driver API 算 [non-sticky error](https://forums.developer.nvidia.com/t/difference-in-error-handling-between-driver-api-and-runtime-api/336389)，跟 GPU 本身无关，在 driver API 层报，不触发 dump。常见例子：`cudaMalloc` 时 OOM。
4. 多卡通信常把别的 GPU 的内存 map 过来。对端进程退出，映射失效，再访问会报 IMA——**不是**典型的 IMA。分布式程序关机顺序乱了就常见。用 core dump 时要把这类假阳性分开。
5. 打开 core dump 对 CUDA kernel **有性能税**（线程退出时要检查、归因）。**不要**在生产默认打开。能稳定复现 IMA 之后再开，用来查。
6. 要对回源码行，建议带 debug 符号重编 vLLM，至少编进行号。默认二进制为了体积 **没有** 这些。要从源码编，见 [GPU 完整编译](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#full-build-with-compilation)，并 `export NVCC_PREPEND_FLAGS='-lineinfo'` 或 `export NVCC_PREPEND_FLAGS='-G'`。先 `-lineinfo`，不够再 `-G`。行信息够了，dump 才能指到那一行。这一步的下文是 [cuda-debugging-source](cuda-debugging-source.md)。

## 收束

CUDA core dump 的原理和用例：不规范 launch、CUDA graph 里的 kernel 异常，IMA 以及更远一点的问题，它都用得上。

他们刚用这套查过 vLLM 里一次复杂 IMA，见 [PR #22593](https://github.com/vllm-project/vllm/pull/22593)。给 MRope 加了 [Triton kernel](https://github.com/vllm-project/vllm/pull/22375)，里面藏着 `head_size==rotary_dim`（full RoPE）的隐含假设。`head_size!=rotary_dim`（partial RoPE）就会 IMA——新模型 [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V) 正是这种情况。没有 dump 时，错报成 `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:453 'an illegal memory access was encountered'`，非常误导。有 dump，直接落到 MRope kernel，再修。这个例子是 **kernel 参数配错**，找到那只 kernel 往往就够。更复杂的 IMA 仍要把 kernel 隔离成最小复现，再用 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool) 往下挖。

vLLM 要的是好用、快、便宜的 LLM serving；好查也是其中一块。后面还会继续写调试技巧。故事可以提到 [博客仓库](https://github.com/vllm-project/vllm-project.github.io) 提 PR。

## 致谢

NVIDIA 的 Ze Long、Vikram Sharma Mailthody、Jeremy Iverson、Sandarbh Jain 参与讨论。Red Hat 的 Lucas Wilkinson 帮忙改草稿。
