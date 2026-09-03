---
source: https://vllm.ai/blog/2025-10-27-semantic-router-modular
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 模块化 LoRA：别为每个分类器跑一整只 BERT

英文对照：[en/vllm/blog/serving/semantic-router-modular.md](../../../../en/vllm/blog/serving/semantic-router-modular.md)  
原文：https://vllm.ai/blog/2025-10-27-semantic-router-modular  
落地见 [Iris](semantic-router-iris.md)。引用数字是演示/文献，不是你集群的 SLA。

意图 + PII + jailbreak 以前三次完整前向。Candle binding 分层；`DualPathUnifiedClassifier` 在全微调与 LoRA 之间选。长上下文：Qwen3-Embedding（32k、号称 100+ 语言）和 EmbeddingGemma-300M（2k；Matryoshka 768/512/256/128）。LoRA：基座一次，适配器通常 **<1%** 参数；`parallel_engine.rs` 用 Rayon。多任务才赚，单任务未必。`OnceLock` 换掉 `lazy_static`。可选 Flash Attention 2（Ampere+）。Rust 分类 + Go FFI 给 Envoy `ext_proc`。

本地图（原文版权仍归原站；学习对照用）：

![modular](../../../../assets/vllm/blog/serving/semantic-router-modular/01-modular.png)

![full params](../../../../assets/vllm/blog/serving/semantic-router-modular/02-full-params.png)

![lora](../../../../assets/vllm/blog/serving/semantic-router-modular/03-lora.png)
