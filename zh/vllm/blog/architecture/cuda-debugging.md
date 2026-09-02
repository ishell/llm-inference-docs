---
source: https://vllm.ai/blog/2025-08-11-cuda-debugging
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# CUDA core dump：illegal memory access 落到哪只 kernel

英文对照：`en/vllm/blog/architecture/cuda-debugging.md`  
原文：https://vllm.ai/blog/2025-08-11-cuda-debugging  
续篇（hang + 源码行）见 [cuda-debugging-source](cuda-debugging-source.md)。

Python 栈在 IMA 上几乎总是错的：异步报错、没检查 `kernel<<<>>>` launch、CUDA graph 里只看见 graph launch。`CUDA_LAUNCH_BLOCKING=1` 救不了后两种。

驱动在异常时 dump GPU 状态。推荐组合（跳过整卡显存，否则几百 GiB）：

```
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"
```

`skip_abort` 当时有 bug，可能吞掉 IMA，**不要开**。`CUDA_DEVICE_WAITS_ON_EXCEPTION=1` 会停住等你 attach `cuda-gdb`，适合要完整显存的现场。`NVCC_PREPEND_FLAGS='-lineinfo'` 才能把 dump 对回源码行。
