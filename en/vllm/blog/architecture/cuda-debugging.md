---
source: https://vllm.ai/blog/2025-08-11-cuda-debugging
lang: en
fetched: 2026-09-01
---

# CUDA core dump: which kernel actually IMA’d

Chinese: `../../zh/vllm/blog/architecture/cuda-debugging.md`  
Follow-up (hangs + source line): [cuda-debugging-source](cuda-debugging-source.md).

Python stacks on illegal memory access are almost always wrong: async report, unchecked `kernel<<<>>>` launches, CUDA-graph launch as the only frame. `CUDA_LAUNCH_BLOCKING=1` does not fix the last two.

The driver dumps GPU state on exception. Recommended combo (skip full VRAM — hundreds of GiB otherwise):

```
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' \
CUDA_COREDUMP_FILE="/persistent_dir/cuda_coredump_%h.%p.%t"
```

`skip_abort` had a bug that could swallow IMA — **leave it off**. `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` stalls for `cuda-gdb` when you need intact memory. `NVCC_PREPEND_FLAGS='-lineinfo'` maps the dump back to source.
