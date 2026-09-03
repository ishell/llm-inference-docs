---
source: https://vllm.ai/blog/2025-09-11-qwen3-next
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Qwen3-Next：Gated DeltaNet + 满 attention 交错，hybrid KV 按物理页对齐

英文对照：[en/vllm/blog/serving/qwen3-next.md](../../../../en/vllm/blog/serving/qwen3-next.md)  
原文：https://vllm.ai/blog/2025-09-11-qwen3-next  
80B-A3B，1:50 MoE。当时 nightly。`vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4`。后继 3.5/3.8 见 [qwen35-25k-tps](qwen35-25k-tps.md) / [qwen38](qwen38.md)。

线性 attention（Flash Linear Attention Triton）和满 attention 层交错，目标 65K+。hybrid KV manager 把满 attention 的逻辑 block 调到和线性层状态一样大的物理页，减少碎片。Triton launch 在 decode-only 上 CPU 贵，所以默认 full CUDA graph。MTP 引擎侧原生。当时 roadmap：GDN kernel、hybrid 上的 prefix cache 和 P/D。Qwen3.5 的 GDN+P/D 是这条线的后续，不是同一篇。

本地图（原文版权仍归原站；学习对照用）：

![qwen](../../../../assets/vllm/blog/serving/qwen3-next/01-qwen.png)

![hybrid](../../../../assets/vllm/blog/serving/qwen3-next/02-hybrid.png)
