---
source: https://vllm.ai/blog/2026-07-06-vllm-hpc-ops
lang: en
fetched: 2026-09-01
---

# HPC-Ops: mixed-length decode attention and small-expert MoE on H20

Chinese: [zh/vllm/blog/performance/hpc-ops.md](../../../../zh/vllm/blog/performance/hpc-ops.md)  
Hopper, especially H20. Attention PR #46020, MoE PR #45924.

Fixed split-KV: mixed lengths in one batch, wall time pinned by the heaviest CTA. HPC-Ops tiles KV at 64 tokens per step and buckets CTAs evenly; a persistent grid drains the task map. Mixed-length decode up to **2.95×** vs static split-KV, ~**2.25×** avg vs FlashInfer/FA. `HpcRopeNorm` fuses QK-Norm, RoPE, KV write (plus query quant in FP8) into one prologue.

MoE decode: expert GEMMs are small; gather/launch/HBM round-trips around them are costlier. Route, Gate-Up, act+quant, Down, top-k reduce become one fused FP8 path; PDL erases stage bubbles. vs Triton/CUTLASS: TP8/EP1 avg **1.59×**, TP1/EP8 **1.21×**. Together on 8×H20 Hy3: TTFT ~**−24%**, TPOT ~**−17%**.

```
--attention-backend HPC_ATTN
--moe-backend hpc
```

Attention then only for Hy3-family; MoE FP8 only. Not a universal default — Hunyuan production kernels through the backend interface into main.

Local figures (copyright remains with the original site; study copies):

![dynamic partitioning](../../../../assets/vllm/blog/performance/hpc-ops/01-dynamic-partitioning.png)

![fused moe latency](../../../../assets/vllm/blog/performance/hpc-ops/02-fused-moe-latency.png)

![decode dynamic vs static](../../../../assets/vllm/blog/performance/hpc-ops/03-decode-dynamic-vs-static.png)
