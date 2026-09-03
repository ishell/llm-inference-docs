---
source: https://vllm.ai/blog/2025-08-05-gpt-oss
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# gpt-oss Day-0：MXFP4 MoE + 满/滑窗 1:1 attention + hybrid KV

英文对照：[en/vllm/blog/serving/gpt-oss.md](../../../../en/vllm/blog/serving/gpt-oss.md)  
原文：https://vllm.ai/blog/2025-08-05-gpt-oss  
20B / 120B。Blackwell、Hopper、MI300x/MI355x。当时 `vllm==0.10.1+gptoss` 或 `vllm/vllm-openai:gptoss`。Pareto 后续见 [gpt-oss-optimizations](../performance/gpt-oss-optimizations.md)。

120B：128 expert、每 token 4、无 shared；MXFP4 后约 **63 GB**。20B 约 14 GB。Blackwell：FlashInfer 原生 MXFP4 Tensor Core；Hopper：OpenAI Triton `matmul_ogs`。Attention：GQA 64/8，head dim **64**，满 attention 与 window=128 1:1，每 query head 有 attention sink。hybrid KV allocator 让两类层共享物理页。内置浏览/Python：Responses API 或外部 MCP，不是普通 `/chat/completions` tool parser。
