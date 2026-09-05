---
source: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
lang: en
fetched: 2026-09-04
---

# DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action

Chinese: [zh/vllm/blog/architecture/deepseek-v32.md](../../../../zh/vllm/blog/architecture/deepseek-v32.md)

2025-09-29. **vLLM Team**. Day-0. Study note. Model: [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Paper: [DSA PDF](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf). Later compression stack: [deepseek-v4.md](deepseek-v4.md). GB300 V3.2 vs R1: [gb300-deepseek.md](../performance/gb300-deepseek.md). Wide-EP: [gb200-wideep.md](../serving/gb200-wideep.md). FP8 KV: [fp8-kvcache.md](../performance/fp8-kvcache.md). Out-of-tree hardware: [hardware-plugin.md](hardware-plugin.md).

Fits: DSA in continuous batching — lightning indexer, separate indexer K vs MLA KV, `ks`/`ke` causal windows, FlashMLA `block_size` 64. Does not fit: treating Day-0 as a finished EP story (the page said EP had a bug), or skipping the GB300 note that one DSA layer is ~**2.7×** MLA kernel time.

## Overview

Day-0 for DeepSeek-V3.2-Exp: DeepSeek Sparse Attention (DSA) for long context. Hard parts in vLLM: **continuous batching** and **paged attention** — prefill vs decode for the indexer, different cache layouts.

Performance path: lightning-indexer CUDA kernels in DeepGEMM; new sparse attention in FlashMLA. Blackwell with NVIDIA: **B200** and **GB200**.

![dsa explained](../../../../assets/vllm/blog/architecture/deepseek-v32/01-dsa-explained.png)

**Figure 1.** DSA: lightning indexer picks top-2048, then sparse attention (study copy; copyright remains with the original site).

## Usage (then)

Recipes: [DeepSeek-V3.2-Exp](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html). Initial support still moving: [PR #25869](https://github.com/vllm-project/vllm/pull/25869). Known issues: [#25877](https://github.com/vllm-project/vllm/issues/25877).

**16×H100 / 8×H200 / 8×B200**, tensor parallel (expert parallel had a slight bug they were fixing):

```
vllm serve deepseek-ai/DeepSeek-V3.2-Exp --tensor-parallel-size 8
```

Scale: one-click Kubernetes via `llm-d` was promised that week — P/D disaggregation with NIXL, then route to data-parallel ranks. Docs “soon.”

Test with **long input or long expected output**. Compare to V3.1-Terminus (same data mix, continued pretrain). Accuracy vs official numbers still being verified; an earlier weight snapshot matched expected GSM8K and GPQA-Diamond, similar to V3.1-Terminus.

## Top-K sparse attention in vLLM

### Cache and quantization

Lightning indexer has its **own** cached K for indexing — a second K cache per token, allocated **apart** from MLA KV.

![mla indexer block](../../../../assets/vllm/blog/architecture/deepseek-v32/02-mla-indexer-block.png)

**Figure.** MLA KV vs indexer key layout (study copy).

MLA FP8 KV per token: **656 bytes**:

- First **512** bytes: quantized NoPE — 512× `float8_e4m3`
- Next **16** bytes: 4× `float32` scales (one per 128 `float8_e4m3`)
- Last **128** bytes: RoPE — 64× `bfloat16`, **not** quantized

Indexer keys are stored **per block**. Reasons `block_size` **64** only: this layout, and FlashMLA is cut that way. First `block_size * head_dim` entries are values; the rest are scales:

```
x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
x_fp8[ :, block_size * head_dim :] = scales.view(num_blocks, block_size).view(dtype=torch.uint8)
```

One token’s indexer cache is **not contiguous**.

### Masked computation / batching

Each new query goes through the indexer → top **2048** tokens to attend. Query shape `(h, d)`; context `(n, d)`; logits `(n, h)`; weight by head weights → `(n,)`; emit `(2048,)` indices, pad `-1` if fewer than 2048.

DeepGEMM:

```
logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
```

Several queries from **one** request (prefill): Q `(q, h, d)`, context still `(n, d)`, logits `(q, n, h)` → `(q, n)` after head weights → `(q, 2048)` indices. Causality: each query only sees tokens before it. `ks` / `ke` are `(q,)` ints: here `ks` all zeros, `ke = range(n - q, n)`.

**Multiple requests:** concatenate Q to `(q1+…+qb, h, d)` and context to `(n1+…+nb, d)`. Logits `(q1+…+qb, n1+…+nb, h)` → `(q1+…+qb, 2048)` indices. `ks` / `ke` length `q1+…+qb`.

Page’s `ks`: `[0] * q1 + [q1] * q2 + …` (repeat). `ke`: `range(n1-q1, n1) + range(n2-q2, n2) + …` **plus the `ks` offset**.

After logits: `topk`. High batch × long context **materializes the full logits** before row-wise topk — the performance pit.

### Fusion, more kernels, Blackwell

Low-hanging fruit then:

- Fused top-k (TileLang kernel from DeepSeek as reference)
- Quantize MLA latent and indexer keys **as they hit the page table** (the new, different scheme)

Out-of-the-box Blackwell; first-class on later model releases.

## Ongoing (as of the post)

DSA optimization barely started:

- Architectures beyond Hopper / Blackwell
- AMD and TPU; extensible backends — [vllm-ascend](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0rc0) and [vllm-mlu](https://github.com/Cambricon/vllm-mlu) already had V3.2
- Wide EP and disaggregation testing
- End-to-end RL loop
- DeepSeek’s “masked MHA mode for short sequence prefilling”
- Hadamard transforms **removed** in this release (no accuracy effect observed); more investigation later

GB300 post later: one DSA layer ~**2.7×** MLA kernel time.

## Acknowledgements

- **vLLM:** Chen Zhang, Yongye Zhu, Kaichao You, Simon Mo, Zhuohan Li
- **Red Hat:** Lucas Wilkinson, Matt Bonanni, Wentao Ye, Nicolo Lucchesi, Michael Goin, Robert Shaw, Tyler Michael Smith
- **Meta:** Lucia Fang, Xiaozhu Meng, Lu Fang
- **NVIDIA:** Ray Wang, Barry Kang, Daniel Campora, Julien Demouth, Siyuan Fu, Zeyu Wang, Pen Chun Li

Thanks to DeepSeek for the model, techniques, kernels, and trust in vLLM.
