---
source: https://vllm.ai/blog/2026-07-06-vllm-hpc-ops
lang: en
fetched: 2026-09-04
---

# HPC-Ops: mixed-length Decode attention and small-expert MoE on H20

Chinese: [zh/vllm/blog/performance/hpc-ops.md](../../../../zh/vllm/blog/performance/hpc-ops.md)

2026-07-06. **Tencent Hunyuan AI Infra Team and vLLM Team**. Study note; H20 benches on the page, not your SLA. Attention [PR #46020](https://github.com/vllm-project/vllm/pull/46020), MoE [PR #45924](https://github.com/vllm-project/vllm/pull/45924). Hopper, strongest on **H20**. Attention backends in the same neighborhood: [triton-attn.md](../architecture/triton-attn.md). Hardware door: [hardware-plugin.md](../architecture/hardware-plugin.md).

**TL;DR from the page:**

- Attention and MoE kernels from [HPC-Ops](https://github.com/Tencent/hpc-ops) are first-class backends on vLLM `main`. No source edits, no long-lived fork.
- Attention: per-step load-balanced Decode scheduler + fused RoPE + QK-Norm + KV-write prologue. Mixed-length Decode up to **2.95×** vs static split-KV, **2.25×** average vs FlashInfer and FlashAttention.
- MoE: fully fused low-latency FP8 pipeline. Average **1.59×** at TP8 / EP1 and **1.21×** at TP1 / EP8 vs Triton and CUTLASS, matched output quality.
- End-to-end Hy3 on 8× H20: TTFT about **−24%**, TPOT about **−17%** vs the vLLM default backend.

## Why this matters

Production serving is mixed-length and increasingly MoE with long context; agentic traffic pushes both. Latency is decided by how kernels schedule GPU work and move data between stages, not matmul peak alone. Fixed split-KV Decode stalls on the longest request in a mixed batch. MoE expert GEMMs are small; a conventional path gathers tokens, pays a launch per stage, and bounces intermediates through HBM.

## A quick word on Hy3

Hy3 is Tencent Hunyuan’s MoE for agentic execution, coding, and long-horizon reasoning. **21B** of **295B** parameters active. **192** experts, top-8 routing, GQA (64 heads, 8 KV heads, head dim 128), **256K** context, **3.8B** MTP layer. Ships BF16 and FP8 (Hy3-FP8). This post is about the kernels that serve it, not the model card.

## HPC-Ops in vLLM

[HPC-Ops](https://github.com/Tencent/hpc-ops) is Hunyuan’s open operator library: attention, MoE, GEMM, sampling, normalization, communication-compute fusion; native BF16 and FP8; Python API meant to drop into frameworks. Hopper-optimized, especially H20. Two backends upstreamed:

| vLLM backend | What it optimizes | Precision | Merged in |
| --- | --- | --- | --- |
| Attention | Load-balanced Decode + fused RoPE/QK-Norm prologue | BF16 / FP8 | [PR #46020](https://github.com/vllm-project/vllm/pull/46020) |
| Fused MoE | Fully fused low-latency MoE pipeline | FP8 | [PR #45924](https://github.com/vllm-project/vllm/pull/45924) |

## Attention backend: dynamic load-balanced scheduling

### The challenge: mixed-length Decode in every batch

Decode attention cost scales with KV length: 16K context is roughly **16×** a 1K start. Continuous batching keeps those lengths in one launch. Existing Decode kernels map CTAs with a fixed grid (KV head × request × split-KV chunk). Split degree must be uniform:

- Fix the number of splits → longest sequence dominates; short-request CTAs finish and idle.
- Fix chunk size → split count matches the max any request needs; short requests launch empty chunks.

Wall time is the heaviest CTA.

### The solution: per-step load-balanced Decode scheduler

Three stages, a flat persistent design:

- **Assign.** Lightweight kernel slices every KV sequence into uniform **64-token** tiles. Total tiles / available CTAs = per-CTA bucket size. Tiles fill buckets in head-major, batch-minor order; overflow spills to the next CTA. Long sequences split in proportion to length; short ones do not monopolize a CTA. A minimum workload floor per CTA avoids over-splitting when total work is small (combine cost would exceed the win). Task map is computed **once per Decode step** and reused by every transformer layer that step.
- **Compute and combine.** Persistent grid: each CTA loops its task bin, writes partial output + log-sum-exp to a split buffer, advances until a terminator. No relaunch between tasks. A lightweight combine kernel reduces per-chunk partials to BF16 per (head, request).

CTAs finish together; the static long-tail stall is gone.

![dynamic partitioning](../../../../assets/vllm/blog/performance/hpc-ops/01-dynamic-partitioning.png)

**Figure 1.** Dynamic partitioning: uniform tiling and balanced bucketing.

### A fused attention prologue

QK-Norm, RoPE, KV-cache write — plus query quantization in FP8 — are normally separate memory-bound launches. `HpcRopeNorm` fuses them from the fused QKV projection, in the model’s required order (Hy3 normalizes **before** RoPE), writes K/V into the paged cache, and in FP8 emits a per-token, per-head FP8 query with its scale so attention never re-quantizes. Prefill and Decode.

### Integrating with vLLM

`HpcAttentionBackend` inherits `AttentionBackend` and registers through the standard mechanism, alongside FlashAttention and FlashInfer.

## MoE backend: fused low-latency FP8 pipeline

### The challenge: small expert GEMMs and the overhead around them

High-throughput large-batch MoE is compute-bound and existing kernels are fine. Low-latency Decode is the opposite: a handful of tokens per expert, memory-bound GEMMs, tile counts that shift every step. Conventional path: route, gather into per-expert HBM buffers, Gate-Up GEMM, activation + quant, Down GEMM, top-k weighted reduce — a launch and an HBM round-trip per stage.

### The solution

Routing and index preprocessing, Gate-Up GEMM, activation + quant, Down GEMM, and top-k weighted reduction fuse into one compact path:

- **Routing and index build.** Shared-memory counting assigns tokens with contiguous per-expert ranges (less global-atomic pressure) and builds the routing indices / per-tile task map the GEMMs consume.
- **Gate-Up GEMM.** Reads original tokens through the routing index — no standalone gather. Activation + FP8 quant is a separate fused kernel whose output Down GEMM reads directly.
- **Occupancy-first, no warp specialization.** One warp group does data movement and compute; latency hiding moves from intra-CTA software pipeline to cross-CTA hardware scheduling. Persistent grid keeps SMs full and spreads uneven per-expert tiles.
- **PDL-chained stages.** Programmatic Dependent Launch overlaps each launch with the previous tail, through the final top-k reduce (which can fold in shared-expert output).

Experts run FP8 with per-tensor and block-wise scaling; output quality matches baselines.

### Integrating with vLLM

`HPCExperts` inherits `FusedMoEExpertsModular`, registered beside DeepGEMM and Triton.

## Using HPC-Ops backends

Install from source:

```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops
make wheel
python3 -m pip install dist/*.whl
```

Attention currently **Hy3-series only**:

```bash
vllm serve tencent/Hy3 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN
```

Hy3-FP8 extra flags:

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN \
    --kv-cache-dtype fp8_e4m3 \
    --block-size 64
```

**Tip from the page:** for a custom model, replace `rope_norm` with `HpcRopeNorm` in `forward`. See PR #46020.

MoE is **FP8 models only**:

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --moe-backend hpc
```

Hardware: NVIDIA Hopper only; best on H20.

## Performance on H20

### Fused MoE vs Triton / CUTLASS

Hy3 config. Averaged over batch sizes: **1.59×** vs best baseline at TP8 / EP1, **1.21×** at TP1 / EP8. Largest gains at small-to-mid batches that dominate low-latency Decode.

**Table 1.** FusedMoE latency (µs), TP8 / EP1 (expert weights sharded across 8 ranks).

| Batch | HPC-Ops | Triton | CUTLASS |
| ---: | ---: | ---: | ---: |
| 4 | 42.0 | 56.4 | 74.5 |
| 16 | 85.7 | 124.2 | 209.2 |
| 32 | 124.0 | 184.3 | 275.6 |
| 64 | 147.2 | 374.9 | 330.3 |
| 128 | 161.5 | 302.9 | 345.3 |
| 256 | 170.1 | 310.9 | 351.6 |
| 512 | 194.5 | 331.6 | 369.2 |
| 1024 | 281.4 | 652.7 | 438.3 |
| 2048 | 491.8 | 731.5 | 794.4 |
| 4096 | 872.0 | 1366.0 | 1230.7 |
| 8192 | 1695.0 | 2216.8 | 2362.9 |
| 16384 | 3241.9 | 4329.1 | 4364.4 |

**Table 2.** FusedMoE latency (µs), TP1 / EP8.

| Batch | HPC-Ops | Triton | CUTLASS |
| ---: | ---: | ---: | ---: |
| 4 | 118.6 | 147.4 | 140.4 |
| 8 | 136.7 | 192.8 | 170.7 |
| 16 | 149.8 | 198.4 | 263.5 |
| 32 | 153.6 | 214.6 | 264.4 |
| 64 | 166.5 | 358.1 | 266.8 |
| 128 | 213.5 | 251.7 | 272.6 |
| 256 | 386.2 | 454.9 | 493.5 |
| 512 | 705.5 | 691.7 | 741.7 |
| 1024 | 1342.6 | 1369.1 | 1359.1 |
| 2048 | 2513.9 | 2668.7 | 2530.4 |

![fused moe latency](../../../../assets/vllm/blog/performance/hpc-ops/02-fused-moe-latency.png)

**Figure 2.** HPC-Ops FusedMoE on H20 — Hy3.

### Decode under mixed-length batches

FP8 Decode from uniform to highly skewed (label A×B = A requests at KV length B). Advantage vs static grows with skew: parity on small uniform batches to **2.95×** on 1×128K + 31×4K. Average **2.25×** vs the best of FlashInfer and FlashAttention.

**Table 3.** Decode latency (ms) across KV-length distributions.

| Decode scenario | HPC-Ops dynamic | HPC-Ops static | FlashInfer | FlashAttention | Dynamic vs static |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64×0.5K | 0.013 | 0.013 | 0.050 | 0.025 | 1.00× |
| 64×4K | 0.033 | 0.043 | 0.221 | 0.095 | 1.32× |
| 32×0.125K + 32×4K | 0.020 | 0.033 | 0.119 | 0.053 | 1.59× |
| 2×32K + 30×4K | 0.032 | 0.056 | 0.169 | 0.094 | 1.76× |
| 1×64K + 15×4K | 0.042 | 0.097 | 0.118 | 0.065 | 2.32× |
| 1×128K + 31×4K | 0.063 | 0.186 | 0.220 | 0.097 | 2.95× |

![decode dynamic vs static](../../../../assets/vllm/blog/performance/hpc-ops/03-decode-dynamic-vs-static.png)

**Figure 3.** Decode attention on H20 — Hy3: dynamic vs static scheduling.

### Attention vs FlashAttention / Triton / FlashInfer

vLLM attention benchmark, Prefill / Extend / Decode. HPC-Ops is at parity or faster than the fastest of the three in nearly every case.

**Table 4.** Attention latency (ms).

| Batch spec | Type | Batch | HPC-Ops | FlashAttention | Triton | FlashInfer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q512 | Prefill | 1 | 0.047 | 0.069 | 0.123 | 0.070 |
| q1ks2k | Extend | 1 | 0.406 | 0.431 | 1.132 | 0.431 |
| q2k | Prefill | 1 | 0.530 | 0.574 | 1.525 | 0.609 |
| q4k | Prefill | 1 | 2.002 | 2.093 | 5.816 | 2.144 |
| q8k | Prefill | 1 | 7.883 | 7.957 | 22.702 | 8.084 |
| 2q1ks4k | Extend | 2 | 1.835 | 1.830 | 5.046 | 1.829 |
| 8q1s1k | Decode | 8 | 0.019 | 0.031 | 0.035 | 0.021 |
| 16q1s2k | Decode | 16 | 0.054 | 0.098 | 0.106 | 0.052 |
| 32q1s1k | Decode | 32 | 0.057 | 0.102 | 0.080 | 0.058 |
| 64q1s4k | Decode | 64 | 0.299 | 0.620 | 0.510 | 0.340 |

### End-to-end: Hy3 on 8× H20

Both backends vs vLLM default. TTFT about **24%** lower on average; TPOT about **17%**, growing to about **30%** at the largest batch. Chunked Prefill and Prefix Caching **disabled** for the TTFT tables.

**Table 5.** TPOT (ms), output length = 4K.

| Batch | Baseline | HPC | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 8.00 | 7.76 | +3.0% |
| 4 | 11.14 | 10.67 | +4.2% |
| 8 | 13.49 | 11.31 | +16.2% |
| 16 | 17.98 | 13.56 | +24.6% |
| 32 | 24.13 | 18.32 | +24.1% |
| 64 | 31.10 | 21.90 | +29.6% |

**Table 6.** TTFT (ms), input length = 8k.

| Batch | Baseline | HPC | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 565.69 | 431.00 | +23.8% |
| 4 | 1920.15 | 1471.43 | +23.4% |
| 8 | 3948.22 | 3035.44 | +23.1% |
| 16 | 7807.18 | 5885.63 | +24.6% |

**Table 7.** TTFT (ms), batch size = 16, sweep input length.

| Input length | Baseline | HPC | Improvement |
| --- | ---: | ---: | ---: |
| 2k | 1792.62 | 1363.13 | +24.0% |
| 4k | 3704.27 | 2886.40 | +22.1% |
| 8k | 7807.12 | 5893.93 | +24.5% |

## What’s next / acknowledgements

Longer collaboration; more upstream as it matures. Named on the page: Tencent Hunyuan AI Infra (Sethran Liu, Chase Shao, Shengy Wei, Theo Cheng, Ryann Xue, Lando Jiang, Looper Zhao, Haank Lin, Aiden Ren, Lehua Ding, Chengv Jiang, Steven Kuang, Liqi He, Kipper Gong, Reedlau Liu, Raccoon Liu, Dick Zhu); Tencent Network Platform Department; vLLM/Inferact (Kaichao You, Yongye Zhu, Yifan Qiao); NVIDIA kernel collaborators. Baselines measured against CUTLASS/CuTe, TensorRT-LLM, FlashInfer, FlashAttention, Triton.
