---
source: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
lang: zh
fetched: 2026-08-30
---

# Mastering LLM Techniques: Inference Optimization（中文导读）

英文全文已保存：`en/nvidia/developer-blog/mastering-llm-techniques-inference-optimization.md`  
原文：https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

这是 NVIDIA 推理优化的概念入门，后面 NIM / TensorRT-LLM 调优文档都假设你读过。本地英文稿约 30KB，尚未逐段全译。核心骨架：

- **Prefill** 偏 compute-bound（一次处理整段 prompt，建 KV cache）
- **Decode** 偏 memory-bound（每步一个 token，反复读 KV）
- **KV cache**、**inflight / continuous batching**、**paged attention**
- 量化、张量/流水线并行、speculative decoding

要精读请直接看英文稿或原网页（有图）。需要的话可以再开一轮把这篇全译。
