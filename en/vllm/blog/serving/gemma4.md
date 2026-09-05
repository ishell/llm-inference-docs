---
source: https://vllm.ai/blog/2026-04-02-gemma4
lang: en
fetched: 2026-09-04
---

# Gemma 4 day-0: NVIDIA / AMD / Intel / TPU together, Apache 2.0

Chinese: [zh/vllm/blog/serving/gemma4.md](../../../../zh/vllm/blog/serving/gemma4.md)

2026-04-02. **Google Team**. E2B / E4B / 26B MoE / 31B Dense. TPU day-0 is the hook — [vllm-tpu.md](../architecture/vllm-tpu.md). Recipes on the [model card](https://huggingface.co/collections/google/gemma-4) and GKE/GCE demos. Almost no reproducible TPS; treat as a support matrix, not a benchmark. Diffusion cousin (not this lineup): [diffusion-gemma.md](../architecture/diffusion-gemma.md).

**TL;DR from the page:**

- Immediate support across NVIDIA, AMD, Intel XPU, and first-ever Day-0 on Google TPU.
- Edge **128K**, larger **256K**. All sizes native image/video; E2B/E4B also audio.
- Function calling, structured JSON, system instructions. Apache 2.0.

## Elevating open models

[Gemma 4](https://aistudio.google.com/prompts/new_chat?model=gemma-4-31b-it): Google’s then-most-sophisticated open lineup, commercially [Apache 2.0](https://goo.gle/gemma-4-apache-2). Same research line as Gemini 3. Four sizes: Effective 2B (E2B), Effective 4B (E4B), 26B MoE, 31B Dense.

Local figures (copyright remains with the original site; study copies):

![gemma4 elo score](../../../../assets/vllm/blog/serving/gemma4/01-gemma4-elo-score.png)

**Figure.** Open-model performance vs size on [Arena.ai](http://arena.ai) chat arena as of 2/1. More benches on the [model card](https://ai.google.dev/gemma/docs/core/model_card_4).

## Powerful, accessible, open

Engineered for Android through workstations to high-scale accelerators. Named early uses: INSAIT [BgGPT](https://deepmind.google/models/gemma/gemmaverse/insait/), Yale [Cell2Sentence-Scale](https://blog.google/innovation-and-ai/products/google-gemma-ai-cancer-therapy-discovery/).

Core capabilities on the page:

- **Advanced Reasoning** — multi-step planning; math and logic-heavy instruction following
- **Agentic Workflows** — function-calling, structured JSON, system instructions
- **Code Generation** — local-first workstation
- **Vision and Audio** — native image/video, variable resolution; OCR and charts. E2B/E4B native audio
- **Longer Context** — 128K edge, 256K larger; repository-level analysis
- **140+ Languages**

Google blog: [gemma-4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).

## Hardware support

[NVIDIA, AMD, Intel GPUs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) and [Google TPUs](http://tpu.vllm.ai) — laptop-class cards through datacenter.

## Key capabilities for vLLM users

Same four: native vision (all) + audio (E2B/E4B); agentic (function-calling / JSON / system instructions); 128K / 256K context; 140+ languages.

## Getting started

[Model card](https://huggingface.co/collections/google/gemma-4), [recipes](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).

GKE / GCE vision+text demos: [Trillium](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Gemma4), [Ironwood](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Gemma4), [NVIDIA GPUs](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm).
