---
source: https://vllm.ai/blog/2026-07-15-inkling
lang: en
fetched: 2026-09-04
---

# TML Inkling on vLLM: Day-0 Support with Optimized Performance

Chinese: [zh/vllm/blog/serving/inkling.md](../../../../zh/vllm/blog/serving/inkling.md)

2026-07-15. **vLLM Team**. Demo: **4× GB200**. Checkpoints: [`thinkingmachines/Inkling-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-NVFP4), [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling) (BF16). Integration: [PR #48768](https://github.com/vllm-project/vllm/pull/48768). Runner: [mrv2.md](../architecture/mrv2.md). Spec: [spec-decode.md](../performance/spec-decode.md). P/D cousin: [large-scale.md](large-scale.md). Study note; **AMD not yet** (needs a relative-attention kernel). **Not a new engine** — sconv cache is a virtual SWA KV layer.

TML Inkling is a **1T** multimodal model from [Thinking Machines Lab](https://thinkingmachines.ai/): **text, image, and audio** in, text out, native **1M** context. Novel pieces — relative attention, short convolution, shared expert sinks — are in vLLM. Demo: up to **380 tok/s/user with MTP**, **140 tok/s/user without**, on 4 GB200 GPUs. Feature parity claimed: LoRA, TP/DP/EP/PP, prefix caching, disaggregated serving. Accuracy and tool parsing verified.

Local figures (copyright remains with the original site; study copies):

![image1](../../../../assets/vllm/blog/serving/inkling/01-image1.png)

![inkling model architecture](../../../../assets/vllm/blog/serving/inkling/02-inkling-model-architecture.png)

![sconv tp sharding](../../../../assets/vllm/blog/serving/inkling/03-sconv-tp-sharding.png)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

vllm serve thinkingmachines/Inkling-NVFP4 \
      --tokenizer-mode inkling \
      --reasoning-parser inkling \
      --tool-call-parser inkling \
      --enable-auto-tool-choice \
      --tensor-parallel-size 8 \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 8}' \
      --kernel-config.enable_flashinfer_autotune=False \
      --trust-remote-code
```

## TL;DR

- **Models:** Inkling-NVFP4 and Inkling (BF16)
- **Hardware:** NVIDIA Blackwell and Hopper. Broader hardware in progress.
- **Modality:** text/image/audio → text
- **Context:** up to 1M natively (Tinker exposes 64K and 256K)
- **Features:** LoRA, MTP, TP/DP/EP/PP, prefix caching, disaggregated serving
- **Optimizations:** sconv-aware TP sharding, low-latency fused collectives, kernel fusion, multi-streaming, PDL
- **Performance:** **380 tok/s/user (MTP)** and **140 tok/s/user (no MTP)** on 4 GB200
- **Accuracy:** MMAU, MMMU-Pro, BFCL, NIAH-1M, HLE vs reference

## Model Architecture

**Figure 1.** TML Inkling architecture (RMSNorm and residuals omitted on the figure).

**Modality.** 1T, natively multimodal. Lightweight image encoder (hMLP) and audio embeddings (dMel); see [TML’s interaction model preview](https://thinkingmachines.ai/blog/interaction-models/). Embeddings go into a decoder-only Transformer backbone.

**Attention.** 66 layers: **11** full-attention + **55** sliding-window. Heavy SWA is what makes **1M** context affordable. All attention is GQA, head size 128.

Positional mechanism is **relative attention**, not RoPE: a learned relative-position term added to pre-softmax logits. Details on TML’s blog.

**Sconv.** Aggressive short convolution, window **4**. Four sconv modules per layer: attention K, attention V, attention output, MoE output. Small local attention at low compute and memory cost.

**MoE.** 256 routed experts, top-6, plus **2 shared experts** — 8 experts per token. **Expert sink:** the two shared experts participate in routing-score computation (absorb probability mass) but are **excluded** from top-6 selection.

`Inkling-NVFP4`: only **routed** experts are NVFP4; shared experts and qkvr linears stay BF16. `Inkling`: MoE weights BF16 as well.

**MTP.** **8 MTP heads**, up to 9 tokens per forward. Heads are **chained**: each consumes hidden states and the sampled draft token from the previous head. Each head is a single-layer Transformer (full or SWA) with a dense MLP. All MTP weights **BF16**.

## vLLM Integration & Optimization

**Managing the sconv cache.** Short convolution keeps hidden states of the last `W-1` tokens. vLLM treats that cache as the KV of a **virtual sliding-window attention layer**. Unified KV cache manager: states outside the window are evictable; prefix caching works with the sconv cache.

**Figure 2.** Sconv-aware TP sharding.

**Sconv-aware TP sharding.** Naive TP: all-reduce (e.g. after `o_proj`) → sconv → residual → RMSNorm. That applies sconv to the **full** hidden on every GPU — duplicated compute and duplicated sconv cache.

Sconv is independent along the channel dim, so vLLM shards sconv **across channels**: reduce-scatter / all-gather on the channel axis instead of all-reduce. Each GPU stores only a shard of the sconv cache and computes only its channel slice. Like sequence parallelism, but the shard axis is **channel**, not token.

**Low-latency fused collectives.** Fused reduce-scatter and all-gather (with surrounding ops) by extending FlashInfer’s low-latency all-reduce **Lamport** protocol. Synchronize via data-value polling instead of explicit barriers. Batch size 1: kernel time **40 µs → 8 µs (5×)**.

**FA4 with sheared bias.** Relative attention complicates memory access and slows the attention pipeline. TML + Colfax Research released [FA4 with sheared-bias](https://github.com/vllm-project/tml-fa4); vLLM integrates it and picks FA4’s `num_splits` per config (batch, TP, KV length).

**Re-computing MTP KV cache.** Each MTP head takes the previous head’s draft token. On rejection, that KV is stale. vLLM caches the base model’s hidden states for the last few tokens and **re-runs** the MTP heads with the accepted tokens after rejection sampling.

Plus kernel fusion, PDL, multi-streaming. Details: [PR #48768](https://github.com/vllm-project/vllm/pull/48768).

### Performance

**4× GB200**, SPEED-Bench prompts **8K** in / **1K** out: **380 tok/s/user** with MTP8 (mean acceptance length **4.5**), **140 tok/s/user** without MTP.

## Accuracy Evals

vLLM matches the reference across modalities. Long context: exact match through **221K**, within ~**1 pp** through **513K**. At 800K+ NIAH, run-to-run variance is higher; they were tightening reproducibility.

| Benchmark / metric | vLLM NVFP4 | Reference NVFP4 | Delta vs Reference |
| --- | ---: | ---: | ---: |
| MMAU overall | 76.10% (761/1,000) | 75.50% | +0.60 pp |
| BFCL exact calls | 78.61% (1,062/1,351) | 78.16% | +0.45 pp |
| BFCL All-Live macro | 75.86% | 73.54% | +2.32 pp |
| MMMU-Pro overall micro | 71.12% (3,691/5,190) | 70.52% (3,660/5,190) | +0.60 pp |
| MMMU-Pro Standard 10-option | 70.23% (1,215/1,730) | 70.00% (1,211/1,730) | +0.23 pp |
| MMMU-Pro Standard 4-option | 76.47% (1,323/1,730) | 76.30% (1,320/1,730) | +0.17 pp |
| MMMU-Pro Vision | 66.65% (1,153/1,730) | 65.26% (1,129/1,730) | +1.39 pp |
| HLE | 29.33% (633/2,158) | 26.65% | +2.68 pp |
| NIAH (2K-221K) | 99.09% (436/440) | 99.09% (436/440) | 0.00 pp |
| NIAH (294K-513K) | 95.68% (421/440) | 96.82% (426/440) | -1.14 pp |
| NIAH (586K-805K) | 81.36% (358/440) | 84.09% (370/440) | -2.73 pp |
| NIAH (878K) | 70.91% (78/110) | 80.91% (89/110) | -10.00 pp |

## Roadmap

- **FP8 for global attention:** currently BF16; compute and KV-capacity bottleneck. Explore FP8 by modifying the new FA4 kernel.
- **CUDA Graphs for image & audio encoders:** currently eager. Usually Prefill-only; graphs would cut CPU overhead.
- **AMD:** not yet — relative attention needs a dedicated kernel. Coming.

## Acknowledgements

Thinking Machines Lab. Model support led by [Inferact](https://inferact.ai/).
