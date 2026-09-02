---
source: https://vllm.ai/blog/2026-06-10-diffusion-gemma
lang: en
fetched: 2026-09-01
---

# DiffusionGemma

2026-06-10. Google DeepMind. First dLLM in vLLM.  Batch=1 demos. Runner: [mrv2.md](mrv2.md). Spec path: [spec-decode.md](../performance/spec-decode.md). Omni: [vllm-omni.md](../serving/vllm-omni.md).

26B Gemma4 backbone. Denoises a **256-token canvas** (compute vs bandwidth at low batch). Parallel inside a block; left-to-right across blocks.

**Encoder:** causal, writes KV — prefill once, commit when a block converges. **Decoder:** bidirectional, reads KV — denoise. Causal writes keep automatic prefix cache. Entropy budget accepts confident positions first. Commit uses clean argmax. Self-conditioning (softmax-weighted embeddings through a gated MLP) only in decoder.

Canvas = a huge draft that is fully accepted or fully rejected. `num_sampled=0` holds the KV cursor. **ModelState** hooks (`prepare_inputs` / `prepare_attn` / `custom_sampler`) so a new block-diffusion model does not fork the runner.

`DiffusionSampler`: Gumbel-max + entropy bound; commit emits 256 tokens. Mixed prefill/denoise/commit in one batch → **per-sequence causality** (`TRITON_ATTN` / `FLASH_ATTN`). Sliding window on the canvas is symmetric `2W+1`.

FP8/NVFP4 on RedHatAI. Demo: H200 FP8 ~**1288 tok/s** (~6× AR, ~3× MTP); H100 ~**1008 tok/s**.

Local figures (copyright remains with the original site; study copies):

![ar vs diffusion](../../../../assets/vllm/blog/architecture/diffusion-gemma/01-ar-vs-diffusion.svg)

![sampling loop horizontal](../../../../assets/vllm/blog/architecture/diffusion-gemma/02-sampling-loop-horizontal.svg)

![denoising grid](../../../../assets/vllm/blog/architecture/diffusion-gemma/03-denoising-grid.svg)

![self conditioning](../../../../assets/vllm/blog/architecture/diffusion-gemma/04-self-conditioning.svg)

![stack](../../../../assets/vllm/blog/architecture/diffusion-gemma/05-stack.svg)

![per seq causal attention](../../../../assets/vllm/blog/architecture/diffusion-gemma/06-per_seq_causal_attention.svg)

![per seq sliding window](../../../../assets/vllm/blog/architecture/diffusion-gemma/07-per_seq_sliding_window.svg)

![perf](../../../../assets/vllm/blog/architecture/diffusion-gemma/08-perf.svg)
