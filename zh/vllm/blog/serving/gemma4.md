---
source: https://vllm.ai/blog/2026-04-02-gemma4
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Gemma 4 Day-0：NVIDIA / AMD / Intel / TPU 一起开，Apache 2.0

英文对照：[en/vllm/blog/serving/gemma4.md](../../../../en/vllm/blog/serving/gemma4.md)  
原文：https://vllm.ai/blog/2026-04-02-gemma4  
2026-04-02。署名 **Google Team**。E2B / E4B / 26B MoE / 31B Dense。TPU Day-0 是卖点——接 [vllm-tpu.md](../architecture/vllm-tpu.md)。菜谱在 [model card](https://huggingface.co/collections/google/gemma-4) 和 GKE/GCE demo。几乎没有可复现 TPS；当矩阵看，别当基准。扩散亲戚（不是这套）：[diffusion-gemma.md](../architecture/diffusion-gemma.md)。

**原文 TL;DR：**

- 立刻支持 NVIDIA、AMD、Intel XPU，Google TPU 上第一次 Day-0。
- 边端 **128K**，大号 **256K**。全尺寸原生图/视频；E2B/E4B 另有音频。
- function calling、structured JSON、system instruction。Apache 2.0。

## 把开源模再抬一档

[Gemma 4](https://aistudio.google.com/prompts/new_chat?model=gemma-4-31b-it)：Google 当时最完整的开源线，商用 [Apache 2.0](https://goo.gle/gemma-4-apache-2)。和 Gemini 3 同一条研究线。四号：Effective 2B (E2B)、Effective 4B (E4B)、26B MoE、31B Dense。

本地图（原文版权仍归原站；学习对照用）：

![gemma4 elo score](../../../../assets/vllm/blog/serving/gemma4/01-gemma4-elo-score.png)

**Figure。** 开源模 performance vs size，[Arena.ai](http://arena.ai) chat arena，截至 2/1。更多榜在 [model card](https://ai.google.dev/gemma/docs/core/model_card_4)。

## Powerful, accessible, open

从 Android 到工作站到大加速器。点名的早用：INSAIT [BgGPT](https://deepmind.google/models/gemma/gemmaverse/insait/)、Yale [Cell2Sentence-Scale](https://blog.google/innovation-and-ai/products/google-gemma-ai-cancer-therapy-discovery/)。

页上的核心能力：

- **Advanced Reasoning** — 多步规划；数学和逻辑向 instruction following
- **Agentic Workflows** — function-calling、structured JSON、system instruction
- **Code Generation** — 本地优先的工作站
- **Vision and Audio** — 原生图/视频，分辨率可变；OCR 和图表。E2B/E4B 原生音频
- **Longer Context** — 边端 128K，大号 256K；仓库级分析
- **140+ Languages**

Google 博：[gemma-4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)。

## Hardware support

[NVIDIA、AMD、Intel GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) 和 [Google TPU](http://tpu.vllm.ai)——笔记本卡到机房卡。

## Key capabilities for vLLM users

还是那四条：全尺寸原生视觉 + E2B/E4B 音频；agentic（function-calling / JSON / system instruction）；128K / 256K 上下文；140+ 语言。

## Getting started

[Model card](https://huggingface.co/collections/google/gemma-4)，[recipes](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)。

GKE / GCE 视觉+文本 demo：[Trillium](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Gemma4)、[Ironwood](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Gemma4)、[NVIDIA GPU](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm)。
