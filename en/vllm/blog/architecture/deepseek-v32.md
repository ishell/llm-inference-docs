---
source: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
lang: en
fetched: 2026-09-04
---

# DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action

Chinese: [zh/vllm/blog/architecture/deepseek-v32.md](../../../../zh/vllm/blog/architecture/deepseek-v32.md)

2025-09-29. **vLLM Team**. Day-0 for [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) and DeepSeek Sparse Attention (DSA) ([paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)). Later compression / 1M-context stack: [deepseek-v4.md](deepseek-v4.md). GB300 V3.2 vs R1 (one DSA layer ~**2.7×** MLA kernel time): [gb300-deepseek.md](../performance/gb300-deepseek.md). Wide-EP: [gb200-wideep.md](../serving/gb200-wideep.md). FP8 KV layout cousin: [fp8-kvcache.md](../performance/fp8-kvcache.md). Study note of the first drop; they were still verifying official accuracy, and **expert parallelism had a bug**.

Fits: serving V3.2-Exp with tensor parallelism on **16×H100 / 8×H200 / 8×B200**, `block_size` 64, separate indexer K cache from MLA KV, `ks` / `ke` causal windows. Does not fit: treating this page as the later V4 hybrid-KV stack, or assuming EP was clean on day-0.

Recipe: [DeepSeek-V3_2-Exp](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html). Initial-support PR: [#25869](https://github.com/vllm-project/vllm/pull/25869). Known issues: [#25877](https://github.com/vllm-project/vllm/issues/25877).

DSA: a lightning indexer picks **top-2048**, then sparse attention. Prefill vs decode layouts differ; continuous batching has to mark causal windows. Kernels: DeepGEMM lightning-indexer CUDA, FlashMLA sparse attention. Blackwell with NVIDIA: run on **B200** and **GB200**.

![dsa explained](../../../../assets/vllm/blog/architecture/deepseek-v32/01-dsa-explained.png)

**Figure 1.** DeepSeek Sparse Attention (DSA) (study copy; copyright remains with the original site).

## Usage

Once installed, on 16×H100, 8×H200, or 8×B200, tensor parallelism (EP had a slight bug they were fixing):

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp --tensor-parallel-size 8
```

Scale-out they previewed: one-click Kubernetes via `llm-d` later that week — vLLM with PD disaggregation using **NIXL**, then route to data-parallel ranks per P and D instance. Docs “soon” on the page.

They recommend testing with **long input, or prompts that expect long output**. Compare against **V3.1-Terminus** (continuously pre-trained on the same data mix).

Accuracy: still verifying against official numbers. On a **previous** weight drop they matched expected GSM8K and GPQA-Diamond, similar to V3.1-Terminus.

## Top-K sparse attention in vLLM

### New cache entry and quantization scheme

The lightning indexer caches **K for indexing**. Each token therefore has another K cache besides MLA. vLLM allocates **separate buffers** for indexer K vs MLA K.

![mla indexer block](../../../../assets/vllm/blog/architecture/deepseek-v32/02-mla-indexer-block.png)

**Figure.** MLA KV vs indexer-key block layout (study copy).

FP8 KV, which this model supports. For **MLA**, each token’s KV is **656 bytes**:

- First **512** bytes: quantized NoPE — 512× `float8_e4m3`
- Next **16** bytes: scales — 4× `float32`. First `float32` scales the first 128 `float8_e4m3`, second the next 128, and so on
- Last **128** bytes: RoPE — 64× `bfloat16`. **Not quantized** (accuracy)

Indexer keys are stored **per block**. That is one reason they only support `block_size` **64** for this model; the other is FlashMLA is tailored to it. First `block_size * head_dim` entries are values; the rest are scales:

```
x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
x_fp8[ :, block_size * head_dim :] = scales.view(num_blocks, block_size).view(dtype=torch.uint8)
```

In the indexer, **one token’s cache is not contiguous**.

### New computation with masking

Each new query token goes through the indexer to pick the **top 2048** tokens to attend to. A query is `(h, d)` (`h` query heads, `d` head dim). Context of size `n` is `(n, d)`. Logits (query vs context) are `(n, h)`. Weight by head weights `(h,)` → `(n,)`. Produce a `(2048,)` integer tensor of top-2048 indices, **`-1` padded** if `n < 2048`.

Single-query is obvious; batching is not. DeepGEMM call:

```
logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
```

**Several query tokens from the same request (prefill):** queries `(q, h, d)`, context still `(n, d)`, logits `(q, n, h)`, after head weights `(q, n)`, indices `(q, 2048)`. Causality: each query attends only to tokens before it. `ks` / `ke` are `(q,)` ints marking start and end of context. Here `ks` is all zeros, `ke` is `list(range(n - q, n, 1))`.

**Several requests:** `b` requests, query counts `q1…qb`, context counts `n1…nb`. Queries concatenated `(q1+…+qb, h, d)`, context concatenated `(n1+…+nb, d)`, logits `(q1+…+qb, n1+…+nb, h)`, indices `(q1+…+qb, 2048)`. `ks` and `ke` are length `q1+…+qb`.

As printed: `ks` is `[0] * q1 + [q1] * q2 + … + [q1 + q2 + … + qb] * qb` (`*` = list repeat). `ke` is `list(range(n1 - q1, n1, 1)) + … + list(range(nb - qb, nb, 1))` **plus the `ks` offset**.

After logits: `topk`. Caveat they print: at **high batch × long context**, the full logits tensor is **materialized** before row-wise `topk` — the performance pit.

### Fusion, more kernels, Blackwell

Low-hanging fruit on the page:

- Fused Top-K. TileLang kernel from DeepSeek as reference
- Quantize MLA latent and indexer key vectors **as they are written into vLLM’s page table** — non-trivial because the scheme above is new

Out-of-the-box **Blackwell** support, with NVIDIA. They want Blackwell first-class on later model drops.

## Ongoing work (as of the post)

Barely the surface of DSA / sparse attention:

- Architectures beyond Hopper and Blackwell
- AMD and TPU; with the plugin door, others can add support — [vllm-ascend](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0rc0) and [vllm-mlu](https://github.com/Cambricon/vllm-mlu) **already** had V3.2
- Large-scale wide EP and disaggregation, still under test
- End-to-end RL loop “soon”
- DeepSeek’s **masked MHA mode for short-sequence prefilling**
- **Hadamard transforms removed** in this release — they saw no accuracy effect; further investigation flagged

## Acknowledgements

Community teams named:

- **vLLM:** Chen Zhang, Yongye Zhu, Kaichao You, Simon Mo, Zhuohan Li
- **Red Hat:** Lucas Wilkinson, Matt Bonanni, Wentao Ye, Nicolo Lucchesi, Michael Goin, Robert Shaw, Tyler Michael Smith
- **Meta:** Lucia Fang, Xiaozhu Meng, Lu Fang
- **NVIDIA:** Ray Wang, Barry Kang, Daniel Campora, Julien Demouth, Siyuan Fu, Zeyu Wang, Pen Chun Li

DeepSeek for open-sourcing the model, techniques, and kernels; DeepSeek leadership for trust in vLLM.
