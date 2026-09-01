---
source: https://vllm.ai/blog/2025-12-03-improved-cuda-debugging
lang: en
fetched: 2026-09-01
---

# Hanging kernels mapped to source lines

Chinese: `../../zh/vllm/blog/architecture/cuda-debugging-source.md`  
Follows [core dump](cuda-debugging.md).

Ctrl-C often does nothing on a hang: Python turns SIGINT into `KeyboardInterrupt`, but the process is blocked in a CUDA API. To make Ctrl-C work: `signal.signal(signal.SIGINT, signal.SIG_DFL)` — you lose the Python stack.

User-triggered dump:

```
CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
CUDA_COREDUMP_PIPE="/tmp/cuda_coredump_pipe_%h.%p.%t" \
# …same flags as the IMA dump
```

While hung, write 1MB of zeros to the pipe (`echo` may buffer):

```
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_coredump_pipe_...
```

`cuda-gdb` on the dump shows the hang line. Find the pipe via `/proc/<pid>/fd`. Fat C++ kernels: default `cuda-gdb` shows only the last inlined line; you want the full inline stack. When `lineinfo` is still wrong, the original post’s second half walks DWARF by hand.
