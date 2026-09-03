---
source: https://vllm.ai/blog/2025-06-30-minimax-m1
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# MiniMax-M1：Lightning Attention + MoE，当时 Docker 示例还钉 V0

英文对照：[en/vllm/blog/serving/minimax-m1.md](../../../../en/vllm/blog/serving/minimax-m1.md)  
原文：https://vllm.ai/blog/2025-06-30-minimax-m1  
456B 总、约 45.9B 激活。M3 见 [minimax-m3](minimax-m3.md)。

他们报 100k 生成相对 DeepSeek R1 约 **25%** FLOPs。`--quantization experts_int8`。文内 Docker 写了 `VLLM_USE_V1=0`——**历史**；后来 hybrid allocator 进 V1。Lightning Attention 走 Triton。PagedAttention 把碎片从传统 60–80% 压到他们说的 <4%。这篇是架构介绍加当时部署，不是 M3 的 MSA。

本地图（原文版权仍归原站；学习对照用）：

![benchmark](../../../../assets/vllm/blog/serving/minimax-m1/01-benchmark.png)

![moe](../../../../assets/vllm/blog/serving/minimax-m1/02-moe.png)

![lightning attention](../../../../assets/vllm/blog/serving/minimax-m1/03-lightning_attention.png)
