---
source: https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Laguna XS.2：Day-0 serve + DFlash draft + LLM Compressor 量化盘

英文对照：`en/vllm/blog/performance/laguna-xs2.md`  
原文：https://vllm.ai/blog/2026-05-28-laguna-xs2-dflash-llm-compressor  
Poolside 33B-A3B MoE，agentic coding。图在原网页。并行草稿算法见 [parallel-drafting](parallel-drafting.md)。

vLLM 一等公民接入。DFlash：5 层、0.6B draft，吃 target hidden state，**一次前向出 8 token**，target 一次 verify。他们报 **2–3×** 更快，质量与自回归对齐（接受才提交）。训在 Ultrachat + Magpie 上对 Laguna 重生回复，thinking 开，6 epoch，seq 8192。LLM Compressor 出 FP8 / NVFP4 / INT4 / INT8 compressed-tensors。菜谱在 vLLM Recipes，不在这篇里抄全 CLI。
