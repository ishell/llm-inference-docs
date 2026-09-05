---
source: https://vllm.ai/blog/2026-07-15-inkling
lang: en
fetched: 2026-09-04
---

# TML Inkling on vLLM: Day-0 Support with Optimized Performance

Chinese: [zh/vllm/blog/serving/inkling.md](../../../../zh/vllm/blog/serving/inkling.md)

2026-07-15. **vLLM Team**. Demo numbers on **4× GB200**. Checkpoints: [`thinkingmachines/Inkling-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-NVFP4), [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling) (BF16). Integration: [PR #48768](https://github.com/vllm-project/vllm/pull/48768). FA4 kernel: [vllm-project/tml-fa4](https://github.com/vllm-project/tml-fa4). TML preview: [interaction models](https://thinkingmachines.ai/blog/interaction-models/). Model-runner flag cousin: [../architecture/mrv2.md](../architecture/mrv2.md). Spec path: [../performance/spec-decode.md](../performance/spec-decode.md). Study note. **Not a new engine** — sconv cache is a virtual SWA KV layer. **AMD not yet** (needs a relative-attn kernel).

1T multimodal (text/image/audio → text), native 1M context (Tinker exposes 64K/256K). 66 layers: 11 full + 55 SWA GQA. Position is **relative attention**, not RoPE. Four window-4 **sconv**s per layer. MoE: 256 routed top-6 + 2 shared **expert sinks**. NVFP4 on routed experts only; 8 MTP heads in BF16.

Local figures (copyright remains with the original site; study copies):

![image1](../../../../assets/vllm/blog/serving/inkling/01-image1.png)

![inkling model architecture](../../../../assets/vllm/blog/serving/inkling/02-inkling-model-architecture.png)

![sconv tp sharding](../../../../assets/vllm/blog/serving/inkling/03-sconv-tp-sharding.png)

**Figure (social / logos).** vLLM × Thinking Machines.

## Quick start

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

`--tensor-parallel-size 8` is the command they print; demo TPS below is **4× GB200**. Do not collapse those two facts.

## TL;DR

- **Models:** NVFP4 and BF16 Hub IDs above; feature parity claimed for both.
- **Hardware:** NVIDIA Blackwell and Hopper. Broader hardware “in progress.” **AMD GPUs not supported** at this writing.
- **Modality:** text/image/audio in → text out.
- **Context:** up to **1M** natively (Tinker 64K / 256K windows).
- **Features:** LoRA, MTP speculative decoding, TP/DP/EP/PP, prefix caching, disaggregated serving.
- **Optimizations:** sconv-aware TP sharding, Lamport fused collectives, kernel fusion, multi-streaming, PDL.
- **Performance (demo):** SPEED-Bench 8K in / 1K out on 4× GB200: MTP8 **380 tok/s/user** (mean accept **4.5**), no MTP **140**.
- **Accuracy:** MMAU, MMMU-Pro, BFCL, NIAH-1M, HLE versus a reference NVFP4 run.

## Model architecture

**Figure 1.** Backbone; RMSNorm / residuals omitted on the diagram.

**Modality.** 1T, natively multimodal. Lightweight image encoder **hMLP**, audio embeddings **dMel** ([TML preview](https://thinkingmachines.ai/blog/interaction-models/)). Embeddings go into a decoder-only Transformer.

**Attention.** 66 layers: **11** full + **55** sliding-window. That SWA-heavy mix is why 1M context is affordable. All layers **GQA**, head size **128**.

Positional mechanism is *relative attention*: a learned relative-position term added to **pre-softmax** logits. **Not RoPE.**

**Sconv.** Short convolution, window **4**. Four modules per layer: attention K, attention V, attention output, MoE output. Local mixing with small compute/memory.

**MoE.** 256 routed experts, **top-6**, plus **2 shared**. Every token hits **8** experts. Shared experts are **expert sinks**: they participate in routing-score (absorb probability mass) but are **excluded** from the top-6 candidates.

`Inkling-NVFP4`: only **routed** experts quantized to NVFP4; shared experts and qkvr linears stay BF16. `Inkling`: MoE weights BF16 too.

**MTP.** **8** MTP heads, up to **9** tokens per forward. Heads are **chained**: each consumes hidden states and the sampled draft token from the previous head. Each head is a single-layer Transformer (full or SWA) + dense MLP. **All MTP weights BF16.**

## vLLM integration and optimization

**Managing the sconv cache.** Short conv needs the last `W-1` hidden states. vLLM treats that cache as the KV of a **virtual sliding-window attention layer**. Unified KV manager: tokens past the window are evictable; prefix caching applies to sconv state the same way. **Not a new cache pool.**

**Figure 2.** Sconv-aware TP sharding.

**Sconv-aware TP sharding.** Naive TP: all-reduce (e.g. after `o_proj`) → sconv → residual → RMSNorm. That runs sconv on the **full** hidden state on every GPU and **replicates** sconv compute + cache.

sconv is independent along channels, so they shard channels: **reduce-scatter / all-gather** instead of all-reduce. Each GPU stores a shard of the sconv cache and computes its channel slice. Same idea as sequence parallelism, but the split axis is **channel**, not token.

**Low-latency fused collectives.** Lamport-protocol reduce-scatter / all-gather (fused with surrounding ops), extending FlashInfer's low-latency all-reduce. Sync by **data-value polling**, not explicit barriers. Batch size 1: kernel time **40 µs → 8 µs (5×)**.

**FA4 with sheared bias.** Relative attention wrecks the attention kernel's memory pattern. TML + Colfax Research released [FA4](https://github.com/vllm-project/tml-fa4) with a **sheared-bias** technique; vLLM uses it directly. vLLM also picks FA4 `num_splits` per config (batch, TP, KV length).

**Re-computing MTP KV.** Each MTP head takes the previous head's draft token, so KV goes stale on rejection. vLLM caches base-model hidden states for the last few tokens and **re-runs MTP heads with accepted tokens** after rejection sampling.

Plus kernel fusion, PDL, multi-streaming. Details in [PR #48768](https://github.com/vllm-project/vllm/pull/48768).

### Performance (demo)

4× GB200, SPEED-Bench prompts **8K** in / **1K** out: **380 tok/s/user** with MTP8 (mean acceptance length **4.5**), **140** without MTP.

## Accuracy evals

Every modality they list, versus a reference NVFP4 implementation. Long context: exact match through **221K**; within ~**1 pp** through **513K**. At **800K+**, NIAH has high run-to-run variance; they say they are tightening that regime.

| Benchmark / metric | vLLM NVFP4 | Reference NVFP4 | Delta vs Reference |
|---|---:|---:|---:|
| MMAU overall | 76.10% (761/1,000) | 75.50% | +0.60 pp |
| BFCL exact calls | 78.61% (1,062/1,351) | 78.16% | +0.45 pp |
| BFCL All-Live macro | 75.86% | 73.54% | +2.32 pp |
| MMMU-Pro overall micro | 71.12% (3,691/5,190) | 70.52% (3,660/5,190) | +0.60 pp |
| MMMU-Pro Standard 10-option | 70.23% (1,215/1,730) | 70.00% (1,211/1,730) | +0.23 pp |
| MMMU-Pro Standard 4-option | 76.47% (1,323/1,730) | 76.30% (1,320/1,730) | +0.17 pp |
| MMMU-Pro Vision | 66.65% (1,153/1,730) | 65.26% (1,129/1,730) | +1.39 pp |
| HLE | 29.33% (633/2,158) | 26.65% | +2.68 pp |
| NIAH (2K-221K) | 99.09% (436/440) | 99.09% (436/440) | 0.00 pp |
| NIAH (294K-513K) | 95.68% (421/440) | 96.82% (426/440) | −1.14 pp |
| NIAH (586K-805K) | 81.36% (358/440) | 84.09% (370/440) | −2.73 pp |
| NIAH (878K) | 70.91% (78/110) | 80.91% (89/110) | −10.00 pp |

Audio = MMAU; vision = MMMU-Pro; tool calling = BFCL; reasoning = HLE; long context = NIAH.

## Roadmap (then)

- **FP8 for global attention:** global attn still BF16; compute and KV capacity bottleneck. Plan: modify the new FA4 kernel.
- **CUDA graphs for image & audio encoders:** those encoders run **eager**. Usually Prefill-only; graphs to kill CPU overhead.
- **AMD GPU support:** **not yet**. Relative attention needs a dedicated kernel. “Coming soon” on the page — no date, no ROCm flag.

## Acknowledgements

Thinking Machines Lab. Model support led by [Inferact](https://inferact.ai/).
