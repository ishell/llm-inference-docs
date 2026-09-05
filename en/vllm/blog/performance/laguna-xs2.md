---
source: https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor
lang: en
fetched: 2026-09-04
---

# Laguna XS.2: day-0 serve + DFlash draft + LLM Compressor checkpoints

Chinese: [zh/vllm/blog/performance/laguna-xs2.md](../../../../zh/vllm/blog/performance/laguna-xs2.md)

2026-05-28. **Megan Flynn, Dipika Sikka, Alexandre Marques**. Study note. Poolside 33B-A3B MoE for agentic coding / long-horizon software. Parallel-draft math: [parallel-drafting.md](parallel-drafting.md). Speculators library: [speculators-v050.md](speculators-v050.md). Accept math: [spec-decode.md](spec-decode.md). Recipe (not copied): [recipes.vllm.ai/poolside/Laguna-XS.2](https://recipes.vllm.ai/poolside/Laguna-XS.2). Hub: [poolside/laguna-xs2](https://huggingface.co/collections/poolside/laguna-xs2). Page benches, not your SLA.

Red Hat AI × Poolside at launch: first-class vLLM, a DFlash speculator, quantized checkpoints via LLM Compressor.

## First-class vLLM

Integrated at launch. Standard vLLM APIs; no waiting for a third-party plugin.

## DFlash speculative decoding

Red Hat trained [poolside/Laguna-XS.2-speculator.dflash](https://huggingface.co/poolside/Laguna-XS.2-speculator.dflash) with [Speculators](https://github.com/vllm-project/speculators). [DFlash](https://arxiv.org/abs/2602.06036): 5-layer **0.6B** draft, target hidden states in, **8 tokens in one forward**, one target verify. Accept-or-reject keeps the large-model distribution ([lossless argument](https://arxiv.org/abs/2211.17192)). They quote **2–3×** vs autoregressive Laguna XS.2.

Training: 500k samples from [Ultrachat 200k SFT](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) and [Magpie-Align](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered). Prompts sampled, responses regenerated from Laguna with thinking on. 6 epochs, cosine, max LR **6e-4**, seq **8192**, **3072** block positions randomly sampled per sequence.

Page framing: next generation past Eagle-3 — parallel drafting, lower ITL.

![Laguna DFlash](../../../../assets/vllm/blog/performance/laguna-xs2/01-laguna_dflash.png)

**Figure 1.** Laguna XS.2 + DFlash on two datasets (page caption). Full CLI lives in the recipe, not here.

## LLM Compressor checkpoints

Poolside also shipped compressed-tensors variants: [FP8](https://huggingface.co/poolside/Laguna-XS.2-FP8), [NVFP4](https://huggingface.co/poolside/Laguna-XS.2-NVFP4), [INT4/INT8](https://huggingface.co/poolside/Laguna-XS.2-INT4). Pick by hardware / latency / memory. Library: [llm-compressor](https://github.com/vllm-project/llm-compressor).

## Next steps (from the page)

Hub collection above. Roll your own with LLM Compressor and Speculators.
