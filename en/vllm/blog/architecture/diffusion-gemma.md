---
source: https://vllm.ai/blog/2026-06-10-diffusion-gemma
lang: en
fetched: 2026-09-01
---

# DiffusionGemma

2026-06-10. Google DeepMind. First dLLM in vLLM. Study note; figures on the original page. Batch=1 demos. Runner: [mrv2.md](mrv2.md). Spec path: [spec-decode.md](../performance/spec-decode.md). Omni: [vllm-omni.md](../serving/vllm-omni.md).

26B Gemma4 backbone. Denoises a **256-token canvas** (compute vs bandwidth at low batch). Parallel inside a block; left-to-right across blocks.

**Encoder:** causal, writes KV — prefill once, commit when a block converges. **Decoder:** bidirectional, reads KV — denoise. Causal writes keep automatic prefix cache. Entropy budget accepts confident positions first. Commit uses clean argmax. Self-conditioning (softmax-weighted embeddings through a gated MLP) only in decoder.

Canvas = a huge draft that is fully accepted or fully rejected. `num_sampled=0` holds the KV cursor. **ModelState** hooks (`prepare_inputs` / `prepare_attn` / `custom_sampler`) so a new block-diffusion model does not fork the runner.

`DiffusionSampler`: Gumbel-max + entropy bound; commit emits 256 tokens. Mixed prefill/denoise/commit in one batch → **per-sequence causality** (`TRITON_ATTN` / `FLASH_ATTN`). Sliding window on the canvas is symmetric `2W+1`.

FP8/NVFP4 on RedHatAI. Demo: H200 FP8 ~**1288 tok/s** (~6× AR, ~3× MTP); H100 ~**1008 tok/s**.
