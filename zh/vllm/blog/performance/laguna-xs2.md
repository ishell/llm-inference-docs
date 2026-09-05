---
source: https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Laguna XS.2：Day-0 serve + DFlash draft + LLM Compressor 量化盘

英文对照：[en/vllm/blog/performance/laguna-xs2.md](../../../../en/vllm/blog/performance/laguna-xs2.md)  
原文：https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor  
2026-05-28。署名 **Megan Flynn, Dipika Sikka, Alexandre Marques**。学习笔记。Poolside 33B-A3B MoE，agentic coding / 长程软件。并行草稿算法：[parallel-drafting.md](parallel-drafting.md)。Speculators：[speculators-v050.md](speculators-v050.md)。验收数学：[spec-decode.md](spec-decode.md)。菜谱不抄全 CLI：[recipes.vllm.ai/poolside/Laguna-XS.2](https://recipes.vllm.ai/poolside/Laguna-XS.2)。Hub：[poolside/laguna-xs2](https://huggingface.co/collections/poolside/laguna-xs2)。页上的 bench，不是你的 SLA。

Red Hat AI × Poolside 在发布当天：vLLM 一等公民、一只 DFlash speculator、LLM Compressor 量化盘。

## vLLM 一等公民

发布即接入。标准 vLLM API，不必等第三方插件。

## DFlash 投机解码

Red Hat 用 [Speculators](https://github.com/vllm-project/speculators) 训了 [poolside/Laguna-XS.2-speculator.dflash](https://huggingface.co/poolside/Laguna-XS.2-speculator.dflash)。[DFlash](https://arxiv.org/abs/2602.06036)：5 层、**0.6B** draft，吃 target hidden，**一次前向出 8 token**，target 一次 verify。接受才提交，分布跟大模型对齐（[无损论证](https://arxiv.org/abs/2211.17192)）。他们报相对自回归 Laguna XS.2 **2–3×**。

训练：从 [Ultrachat 200k SFT](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) 和 [Magpie-Align](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered) 抽 50 万条。prompt 抽样，回复用 Laguna 重生，thinking 开。6 epoch，cosine，最大学习率 **6e-4**，seq **8192**，每条序列随机采 **3072** 个 block 位置。

页上的说法：越过 Eagle-3 的下一代——并行草稿，压 ITL。

![Laguna DFlash](../../../../assets/vllm/blog/performance/laguna-xs2/01-laguna_dflash.png)

**Figure 1。** Laguna XS.2 + DFlash 在两份数据集上（页上图注）。完整 CLI 在菜谱里，不在这篇抄。

## LLM Compressor 量化盘

Poolside 还出了 compressed-tensors 变体：[FP8](https://huggingface.co/poolside/Laguna-XS.2-FP8)、[NVFP4](https://huggingface.co/poolside/Laguna-XS.2-NVFP4)、[INT4/INT8](https://huggingface.co/poolside/Laguna-XS.2-INT4)。按硬件 / 延迟 / 内存选。库：[llm-compressor](https://github.com/vllm-project/llm-compressor)。

## 下一步（页上）

上面的 Hub collection。自己的模型用 LLM Compressor 和 Speculators 拧。
