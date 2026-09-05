---
source: https://vllm.ai/blog/2025-12-03-improved-cuda-debugging
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 挂死的 kernel 对回源码行

英文对照：[en/vllm/blog/architecture/cuda-debugging-source.md](../../../../en/vllm/blog/architecture/cuda-debugging-source.md)  
原文：https://vllm.ai/blog/2025-12-03-improved-cuda-debugging  
2025-12-03。署名 **Kaichao You（vLLM）**。学习笔记。接几个月前的 [CUDA core dump](cuda-debugging.md)：先点出是哪只 kernel；这篇再往下，点到 **哪一行**。

IMA dump 已经能在异步执行里指出出事的 kernel。用的人多了，下一步要的是更细：触发问题的那一行源码。这篇先讲怎样抓住 **挂死** 的 kernel，再讲怎样把复杂 kernel 对回源码。

## 怎样找到挂死的 kernel

算力涨得比带宽快，访存模式越来越绕。近年旗舰数据中心 GPU 上的异步访存，高性能 kernel 要配复杂同步。同步又容易赛跑、死锁，代码一大就更明显。

GPU kernel 一挂，进程往往冻住，连 Ctrl-C 都停不了。最笨的办法是杀进程——根因信息为零，只能猜、对半分改动、反复跑测试。

> **原文 NOTE：** 为什么 CUDA kernel 挂着时 Ctrl-C 停不了？Ctrl-C 发 SIGINT。若进程在跑 Python，解释器把 SIGINT 收成 `KeyboardInterrupt`，排到 **回到 Python** 之后再抛。可进程若卡在等 GPU 的底层 CUDA API 上，当时 **没有** Python 在跑，异常排不上队。下面的 `conditional_hang.py` 若要用 Ctrl-C 停掉，脚本开头加 `import signal; signal.signal(signal.SIGINT, signal.SIG_DFL)`，让解释器别拦 SIGINT。代价：停下来时 **没有** Python 栈。

更好的路：CUDA 驱动的 **user induced GPU core dump**。驱动在 OS 里打开 pipe，往里写就能触发 dump。触发后 GPU 状态进 coredump，能看见 GPU 里在干什么，最要紧的是 **哪只 kernel 挂着**。

一只条件挂死的例子：

```python
# save as conditional_hang.py

import triton
import triton.language as tl
import torch


@triton.jit
def conditional_hang_kernel(x_ptr,
                            flag,          # int32 scalar
                            n_elements,    # int32 scalar
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if flag == 1:
        x = x + 1
        tl.store(x_ptr + offs, x, mask=mask)
    else:
        # flag 不变，flag == 0 时等于死循环
        while flag == 0:
            x = x + 1
            tl.store(x_ptr + offs, x, mask=mask)


x = torch.ones(16, dtype=torch.float32, device="cuda")
n_elements = x.numel()
BLOCK_SIZE = 16

conditional_hang_kernel[(1,)](
   x, flag=1, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE,
)
print("After flag=1:", x)  # 应当全是 2

conditional_hang_kernel[(1,)](
   x, flag=0, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE,
)

# print 会挂：打印 x 要同步设备，kernel 永不结束
print("After flag=0:", x)

# 下面到不了
x = x + 2
torch.cuda.synchronize()
```

这段会一直挂。打开用户触发的 GPU core dump：

```bash
CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
CUDA_COREDUMP_PIPE="/tmp/cuda_coredump_pipe_%h.%p.%t" \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t" \
python conditional_hang.py
```

代码还在空转时，往 pipe 里写 1MB 零来触发 dump（`echo` 可能被 pipe 缓冲挡住，不一定够）：

```bash
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_coredump_pipe_hostname.3000837.1764236276
```

原终端会打出 dump 进度：

```text
[01:39:15.256278] coredump: Writing ELF file to /tmp/cuda_coredump_hostname.3000837.1764236276
[01:39:15.256350] coredump: Writing out global memory (0 bytes)
[01:39:15.256354] coredump: Writing out device table
[01:39:15.292027] coredump: Writing out metadata
[01:39:15.292039] coredump: Finalizing
[01:39:15.292124] coredump: Writing done
[01:39:15.292128] coredump: All done (took 00s)
```

`cuda-gdb` 打开 dump，能看见挂在哪：

```text
Opening GPU coredump: /tmp/cuda_coredump_hostname.3000837.1764236276
[Current focus set to CUDA kernel 0, grid 53, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0]
#0  0x00007f2e6fbff300 in conditional_hang_kernel<<<(1,1,1),(128,1,1)>>> () at conditional_hang.py:31
31                  tl.store(x_ptr + offs, x, mask=mask)
```

不仅是 `conditional_hang_kernel`，连挂死的那一行都在。以前连是哪只 kernel 都看不见。

小麻烦：pipe 路径由驱动动态生成，不好找。用 `CUDA_COREDUMP_PIPE` 设模板，再看进程的 fd：

```bash
$ ls /proc/3037675/fd/ -alth | grep /tmp/cuda_coredump_pipe_
lr-x------ 1 user user 64 Nov 27 01:50 98 -> /tmp/cuda_coredump_pipe_hostname.3037675.1764237014
```

## 怎样把复杂 kernel 对回源码

上一篇写过：`export NVCC_PREPEND_FLAGS='-lineinfo'` 把行信息编进二进制，dump 才能对回源码行。查了几件真实问题之后，他们发现 `cuda-gdb` **默认展示行信息的方式不够**：

1. 有些复杂 kernel，即便二进制里有行信息，`cuda-gdb` 仍对不到出事的那一行。
2. 即便对到了，也只显示编译器 inline **之后的最后一行**。C++ 大量靠 inline 拿掉函数调用税，要看完整 inline 栈才能懂。

一个 IMA 的 Python 例子：

```python
# save as illegal_memory_access.py

from dataclasses import dataclass
import torch

@dataclass
class TensorWrapper:
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


def from_buffer(data_ptr: int, size_in_bytes: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(TensorWrapper(data_ptr, size_in_bytes), device=device).view(dtype)

data = from_buffer(123456, 1024, device="cuda:0", dtype=torch.uint8)

index = torch.ones(10, device="cuda", dtype=torch.int32) + 100
print(data[index])
```

PyTorch **>= 2.9.0**（且包含 [这枚 commit](https://github.com/pytorch/pytorch/commit/dae7710bf2561e9e8a8dc76fd30c68e25bd755b8)；否则会看到 `RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device.`）。会触发 IMA。

先开 CUDA core dump 跑：

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t" \
python illegal_memory_access.py
```

进度会点名 kernel：

```text
_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_
```

名字里是 PyTorch 的 `index_elementwise_kernel`。要对回源码行，需要带 `export NVCC_PREPEND_FLAGS='-lineinfo'` **从源码编 PyTorch**，再跑一遍。

二进制带行信息之后，`cuda-gdb` 打开 dump 能看到行（路径按原文当时的机器）：

```text
(cuda-gdb) target cudacore /tmp/cuda_coredump_flow-matic.3756036.1764250282
Opening GPU coredump: /tmp/cuda_coredump_flow-matic.3756036.1764250282
[Current focus set to CUDA kernel 0, grid 4, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 3, lane 0]

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x7ff533bb91d0  ...
#0  void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at
::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<
long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayR
ef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_imp
l<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::Ten
sorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&,
c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})<<<(1,1,1),(128,1,1)>>> ()
    at /data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:203 in _ZZN2at6native17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS
_18TensorIteratorBaseEN3c108ArrayRefIlEES8_ENKUlPcPKclE_clES9_SB_l inlined from IndexKernel.cu:118
203         *reinterpret_cast<scalar_t*>(out_data) = *reinterpret_cast<const scalar_t*>(in_data + offset);
```

再在 `cuda-gdb` 里 `info symbol $errorpc`：

```text
(cuda-gdb) info symbol $errorpc
void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) + 11472 in section .text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_ of /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn
```

`cuda-gdb` 解开编译产物，`/tmp/cuda-dbg/.../elf....o.4gyLzn` 是含 `index_elementwise_kernel` 的 cubin。出错位置 `0x7ff533bb91d0`。用 `nvdisasm` 反汇编，对到那一行：

```bash
$ nvdisasm -ndf -c -gi /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > output.txt
$ grep -C20 7ff533bb91d0 output.txt
```

能看到完整 inline 栈。默认 `cuda-gdb` 只显示最后一次 inline。原文反汇编片段：

```text
        /*7ff533bb9190*/                   IMAD.IADD R19, R23, 0x1, R3 ;
.L_x_27840:
	//## File "/data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu", line 203 inlined at "/data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu", line 118
	//## File "/data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu", line 118 inlined at "/data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu", line 37
	//## File "/data/youkaichao/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu", line 37
        /*7ff533bb91a0*/                   ULDC.64 UR4, c[0x0][0x480] ;
        /*7ff533bb91b0*/                   IADD3 R2, P0, P1, R22, UR4, R2 ;
        /*7ff533bb91c0*/                   IADD3.X R3, R19, UR5, RZ, P0, P1 ;
        /*7ff533bb91d0*/                   LDG.E.U8 R3, desc[UR36][R2.64] ;
```

命令简注：

- `-ndf`：反汇编后关掉 dataflow analyzer
- `-c`：只打印 code section
- `-gi`：用 `.debug_line` 注源码行，并带 inline 信息
- `-C20`：`grep` 在 PC `7ff533bb91d0` 前后各 20 行

若 cubin 里多只 kernel 共用同一 PC（`grep` 多处命中），再过滤：

```bash
$ cuobjdump -elf /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > elf.txt
$ cat elf.txt | grep ".text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_" | grep PROGBITS

  1ac 1b83f80   b200  0 80                     PROGBITS        6    3      26a .text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_
```

原文得到 CUDA function index（`nvdisasm` 的 `-fun`）为 `26a`：

```bash
$ nvdisasm -ndf -c -gi -fun 0x26a /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > output.txt
$ grep -C20 7ff533bb91d0 output.txt
```

差别：从 `cuobjdump` 的 ELF section 查出函数索引再喂给 `-fun`。

这是简化例子。真实 kernel 的 inline 可以很长。原文一段 CUTLASS / FlashAttention 的链：

```text
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/copy_sm90.hpp", line 93 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 158
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 158 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 185
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 185 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_traits.hpp", line 133
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_traits.hpp", line 133 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_atom.hpp", line 103
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_atom.hpp", line 103 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_atom.hpp", line 124
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/atom/copy_atom.hpp", line 124 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/algorithm/copy.hpp", line 211
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/algorithm/copy.hpp", line 211 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/algorithm/copy.hpp", line 412
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/algorithm/copy.hpp", line 412 inlined at "/data/youkaichao/data/vllm_flash_attn/hopper/epilogue_fwd.hpp", line 265
	//## File "/data/youkaichao/data/vllm_flash_attn/hopper/epilogue_fwd.hpp", line 265 inlined at "/data/youkaichao/data/vllm_flash_attn/hopper/flash_fwd_kernel_sm90.h", line 454
	//## File "/data/youkaichao/data/vllm_flash_attn/hopper/flash_fwd_kernel_sm90.h", line 454 inlined at "/data/youkaichao/data/vllm_flash_attn/hopper/utils.h", line 41
	//## File "/data/youkaichao/data/vllm_flash_attn/hopper/utils.h", line 41 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cutlass/device_kernel.h", line 122
	//## File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cutlass/device_kernel.h", line 122
        /*7eebf5e9eb80*/                   STSM.16.M88.4 [R13], R4 ;
        /*7eebf5e9eb90*/                   MOV R34, R26 ;
```

出错的源码调用了若干 CUTLASS 函数，包含它的函数又被上层 inline。这种时候 `cuda-gdb` 对不进行；出错位置附近甚至 **没有任何** 行信息。即便对到了，也只显示最后一帧：`File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/copy_sm90.hpp", line 93 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 158`——仍是 CUTLASS 内部展开，帮不上忙。

按上面的反汇编路径，才能看见完整 inline 链，逐帧找该负责的那一行。

本地图（原文版权仍归原站；学习对照用）：

![poisoned code](../../../../assets/vllm/blog/architecture/cuda-debugging-source/01-poisoned_code.png)

**Figure 1。** attention kernel 里一行「中毒」的代码。

**警告：** dump 要吃得开，行信息是关键。推荐 `export NVCC_PREPEND_FLAGS='-lineinfo'`：不用改编译脚本，所有编出来的 kernel 都带上。正因为透明，`ccache` 一类缓存可能 **忽略** 这面旗，复用旧产物、根本没重编。从源码编时关掉编译缓存。JIT 则去查对应工具怎么加行信息。

## 收束

两套进阶办法。一是用户触发的 dump，抓住挂死的 kernel；二是二进制里的行信息，把复杂 kernel 对回源码。IMA 一类问题尤其有用。两套一起，他们查过 [CUTLASS MLA attention backend 里一次难复现的 hang](https://github.com/vllm-project/vllm/pull/26026)——根因其实在上游 CUTLASS 示例，后来在 [v4.3.0](https://github.com/NVIDIA/cutlass/commit/b1d6e2c9b334dfa811e4183dfbd02419249e4b52) 修了。

vLLM 要的是好用、快、便宜的 LLM serving；调试也能上手是其中一块。后面还会写。故事提到 [博客仓库](https://github.com/vllm-project/vllm-project.github.io) 提 PR。

## 致谢

NVIDIA 的 Ze Long、Sandarbh Jain 参与讨论。Moonshot AI 的 Chao Hong 提供动机例子。Red Hat 的 Lucas Wilkinson 帮忙改草稿。
