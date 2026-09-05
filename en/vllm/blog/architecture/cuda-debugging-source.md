---
source: https://vllm.ai/blog/2025-12-03-improved-cuda-debugging
lang: en
fetched: 2026-09-05
---

# Tracing Hanging and Complicated GPU Kernels Down To The Source Code

Chinese: [zh/vllm/blog/architecture/cuda-debugging-source.md](../../../../zh/vllm/blog/architecture/cuda-debugging-source.md)

2025-12-03. **Kaichao You (vLLM)**. Study note. Follows the earlier [CUDA core dump](cuda-debugging.md) post: that one names the kernel; this one names the **line**.

IMA dumps already point at the failing kernel despite async GPU execution. As people adopted that, they wanted finer grain: the source line that triggered the issue. This post first covers hanging kernels, then mapping a complicated kernel back to source.

## How to find hanging kernels

Compute has grown faster than memory bandwidth, so access patterns got more intricate. Flagship datacenter GPUs added asynchronous memory access that high-performance kernels synchronize around. Those mechanisms race and deadlock, especially in large codebases.

When a GPU kernel hangs, the process typically freezes — even Ctrl-C may not stop it. Killing the process gives no root cause. Guess, bisect, rerun.

> **NOTE from the page:** Why doesn’t Ctrl-C stop a hanging CUDA kernel? Ctrl-C sends SIGINT. If the process is in Python, the interpreter turns SIGINT into `KeyboardInterrupt` and queues it until the process **returns to Python**. If it is blocked in a low-level CUDA API waiting on the GPU, **no** Python is running, so the exception never fires. For `conditional_hang.py` below, to make Ctrl-C work, add `import signal; signal.signal(signal.SIGINT, signal.SIG_DFL)` at the top so Python does not catch SIGINT. Downside: no Python stack when it stops.

A better path: the CUDA driver’s **user induced GPU core dump**. The driver opens OS pipes; writing to them triggers a dump. The dump shows GPU state and, most importantly, **which kernel is hanging**.

A conditional hang:

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
        # flag is invariant; infinite loop when flag == 0
        while flag == 0:
            x = x + 1
            tl.store(x_ptr + offs, x, mask=mask)


x = torch.ones(16, dtype=torch.float32, device="cuda")
n_elements = x.numel()
BLOCK_SIZE = 16

conditional_hang_kernel[(1,)](
   x, flag=1, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE,
)
print("After flag=1:", x)  # should be all 2s

conditional_hang_kernel[(1,)](
   x, flag=0, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE,
)

# this print hangs: printing x synchronizes the device, kernel never finishes
print("After flag=0:", x)

# never reached
x = x + 2
torch.cuda.synchronize()
```

This hangs indefinitely. Enable user-triggered GPU core dump:

```bash
CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
CUDA_COREDUMP_PIPE="/tmp/cuda_coredump_pipe_%h.%p.%t" \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t" \
python conditional_hang.py
```

While it spins, trigger a dump by writing 1MB of zeros to the pipe (`echo` may not be enough because of pipe buffering):

```bash
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_coredump_pipe_hostname.3000837.1764236276
```

The original terminal prints dump progress:

```text
[01:39:15.256278] coredump: Writing ELF file to /tmp/cuda_coredump_hostname.3000837.1764236276
[01:39:15.256350] coredump: Writing out global memory (0 bytes)
[01:39:15.256354] coredump: Writing out device table
[01:39:15.292027] coredump: Writing out metadata
[01:39:15.292039] coredump: Finalizing
[01:39:15.292124] coredump: Writing done
[01:39:15.292128] coredump: All done (took 00s)
```

`cuda-gdb` on the dump shows the hang:

```text
Opening GPU coredump: /tmp/cuda_coredump_hostname.3000837.1764236276
[Current focus set to CUDA kernel 0, grid 53, block (0,0,0), thread (0,0,0), device 0, sm 124, warp 0, lane 0]
#0  0x00007f2e6fbff300 in conditional_hang_kernel<<<(1,1,1),(128,1,1)>>> () at conditional_hang.py:31
31                  tl.store(x_ptr + offs, x, mask=mask)
```

Not only `conditional_hang_kernel`, but the exact hang line. Previously even naming the kernel was impossible.

Minor inconvenience: the pipe path is generated dynamically. Set `CUDA_COREDUMP_PIPE` to a template, then inspect the process’s file descriptors:

```bash
$ ls /proc/3037675/fd/ -alth | grep /tmp/cuda_coredump_pipe_
lr-x------ 1 user user 64 Nov 27 01:50 98 -> /tmp/cuda_coredump_pipe_hostname.3037675.1764237014
```

## How to trace down the source of a complicated kernel

The previous post noted that `export NVCC_PREPEND_FLAGS='-lineinfo'` embeds line info so a dump can map to source. After real issues, they found `cuda-gdb`’s **default line display is imperfect**:

1. For some complex kernels, `cuda-gdb` still misses the failing line even when line info is in the binary.
2. Even when it finds a line, it shows only the **last** line after compiler inlining. C++ inlines heavily; you need the full inline stack.

A concrete IMA script:

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

PyTorch **>= 2.9.0** (and [this commit](https://github.com/pytorch/pytorch/commit/dae7710bf2561e9e8a8dc76fd30c68e25bd755b8); otherwise `RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device.`). This triggers IMA.

Run with CUDA core dump:

```bash
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h.%p.%t" \
python illegal_memory_access.py
```

Progress names the kernel:

```text
_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_
```

That is PyTorch’s `index_elementwise_kernel`. To get the source line, **build PyTorch from source** with `export NVCC_PREPEND_FLAGS='-lineinfo'`, then rerun.

With line info in the binary, `cuda-gdb` on the dump shows a line (paths as on the author’s machine in the post):

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

Then `info symbol $errorpc` inside `cuda-gdb`:

```text
(cuda-gdb) info symbol $errorpc
void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) + 11472 in section .text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_ of /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn
```

`cuda-gdb` unpacks the binary; `/tmp/cuda-dbg/.../elf....o.4gyLzn` is a cubin containing `index_elementwise_kernel`. The error is at `0x7ff533bb91d0`. Disassemble with `nvdisasm`:

```bash
$ nvdisasm -ndf -c -gi /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > output.txt
$ grep -C20 7ff533bb91d0 output.txt
```

That shows the full inline stack. Default `cuda-gdb` shows only the last inline. Snippet from the post:

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

Flags:

- `-ndf`: disable dataflow analyzer after disassembly
- `-c`: code sections only
- `-gi`: annotate from `.debug_line`, including inlining
- `-C20`: `grep` context around PC `7ff533bb91d0`

If the cubin has several kernels sharing that PC (`grep` hits more than once), filter further:

```bash
$ cuobjdump -elf /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > elf.txt
$ cat elf.txt | grep ".text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_" | grep PROGBITS

  1ac 1b83f80   b200  0 80                     PROGBITS        6    3      26a .text._ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi1EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_
```

The page’s CUDA function index (`-fun`) is `26a`:

```bash
$ nvdisasm -ndf -c -gi -fun 0x26a /tmp/cuda-dbg/2123124/session1/elf.21407f80.24fe2940.o.4gyLzn > output.txt
$ grep -C20 7ff533bb91d0 output.txt
```

Difference: look up the function index from the ELF section via `cuobjdump`, then pass `-fun`.

This is a simplified example. Real kernels inline much deeper. A CUTLASS / FlashAttention chain from the post:

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

The faulty source calls CUTLASS helpers; its containing function is inlined from above. Here `cuda-gdb` cannot associate a line — often **no** line info near the error PC. Even when it shows a line, it is only the last frame: `File "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/copy_sm90.hpp", line 93 inlined at "/data/youkaichao/data/vllm_flash_attn/csrc/cutlass/include/cute/arch/util.hpp", line 158` — an internal CUTLASS expansion, still unhelpful.

The disassembly path above uncovers the full inline chain so each frame can be inspected.

Local figures (copyright remains with the original site; study copies):

![poisoned code](../../../../assets/vllm/blog/architecture/cuda-debugging-source/01-poisoned_code.png)

**Figure 1.** A line of poisoned code in the attention kernel.

**Warning:** line information is what makes dumps useful. Prefer `export NVCC_PREPEND_FLAGS='-lineinfo'` so every compiled kernel picks it up without editing scripts. Because it is transparent, `ccache`-style caches may **ignore** the flag and reuse old objects. Disable compilation caching when building from source. For JIT, check that tool’s docs for how to add line info.

## Conclusion

Two advanced techniques. First: user-triggered dumps to identify hanging kernels. Second: line info in the binary to map complex kernels back to source. Especially useful for IMA. Used together they recently debugged [a hard-to-reproduce hang in the CUTLASS MLA attention backend](https://github.com/vllm-project/vllm/pull/26026) that actually came from an upstream CUTLASS example, later fixed in [v4.3.0](https://github.com/NVIDIA/cutlass/commit/b1d6e2c9b334dfa811e4183dfbd02419249e4b52).

vLLM wants easy, fast, affordable LLM serving; accessible debugging is part of that. More notes later. Stories: PR on the [blog repo](https://github.com/vllm-project/vllm-project.github.io).

## Acknowledgement

Ze Long and Sandarbh Jain (NVIDIA) for discussions. Chao Hong (Moonshot AI) for the motivating example. Lucas Wilkinson (Red Hat) for polishing the draft.
