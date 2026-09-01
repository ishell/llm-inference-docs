---
source: https://vllm.ai/blog/2025-09-11-qwen3-next
lang: en
fetched: 2026-09-01
---

# Qwen3-Next: Gated DeltaNet interleaved with full attention; hybrid KV aligned by physical page

Chinese: `../../zh/vllm/blog/serving/qwen3-next.md`  
80B-A3B, 1:50 MoE. Then nightly. `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4`. Later 3.5/3.8: [qwen35-25k-tps](qwen35-25k-tps.md) / [qwen38](qwen38.md).

Linear attention (Flash Linear Attention Triton) interleaved with full attention, targeting 65K+. Hybrid KV manager sizes full-attention logical blocks so they occupy the same physical page as linear state, cutting fragmentation. Triton launch is CPU-heavy on decode-only, so full CUDA graph is default. MTP native. Then-roadmap: GDN kernels, prefix cache and P/D on hybrid. Qwen3.5 GDN+P/D is the sequel, not this post.
