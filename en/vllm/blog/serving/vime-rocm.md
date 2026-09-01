---
source: https://vllm.ai/blog/2026-07-10-vime-rocm
lang: en
fetched: 2026-09-01
---

# vime on ROCm: `vllm/vime-rocm`, Qwen3-8B ~4100 tok/gpu/s on MI355X

Chinese: `../../zh/vllm/blog/serving/vime-rocm.md`  
Mainline: [vime](vime.md).

Image `vllm/vime-rocm`. Qwen3-8B on MI355X ~**4100 tok/gpu/s**. logprob delta ~**0.012** — not bit-exact, “trainable-enough”. Same knob names as CUDA ≠ same kernels. Compare their logprob delta; don’t assume bitwise.
