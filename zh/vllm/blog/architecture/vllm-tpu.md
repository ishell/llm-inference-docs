---
source: https://vllm.ai/blog/2025-10-16-vllm-tpu
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# vLLM TPU：PyTorch 和 JAX 走同一条 XLA 路

英文对照：`en/vllm/blog/architecture/vllm-tpu.md`  
原文：https://vllm.ai/blog/2025-10-16-vllm-tpu  
`pip install vllm-tpu`。

第一代 TPU backend 用 PyTorch/XLA + MPMD，Cloud Next 前把 Llama 3.1-8B（v6e-1）吞吐抬到约 **3.6×**、70B（v6e-8）约 **2.1×**。这一代换成 **tpu-inference**：Torchax / JAX 都降成 JAX→XLA。同一份 `llama.py` 只换 lowering，吞吐大约再 **+20%**。默认先找 tpu-inference 里的 TPU 优化实现，没有再 fallback 上游 PyTorch（仍经 Torchax）。重写 JAX 模型通常不是因为框架更快，而是 GPU 逻辑对 TPU 不友好。

**RPA v3**：任意 head dim / 量化 / TP；KV scatter 融进 attention 藏延迟；编译成 prefill / decode / mixed 三只子核；相对 v2 在 Trillium (v6e) 约 **+10%**。默认 **SPMD**（编译器切张量、插通信），不再搬 GPU 那套多 worker。

当时核实：Trillium / v5e；prefix cache、chunked prefill、多模态（tpu-inference 模型）、ngram 投机、权重量化。实验：v5p、Torchax 多模态、multi-LoRA、tree Eagle-3、单机 P/D。大 MoE / MLA / 视觉 encoder 还在路上。和 [hardware plugin](hardware-plugin.md) 一起读：TPU 是插件，不是 fork。

本地图（原文版权仍归原站；学习对照用）：

![vllm tpu](../../../../assets/vllm/blog/architecture/vllm-tpu/01-vllm-tpu.png)

![whats new](../../../../assets/vllm/blog/architecture/vllm-tpu/02-whats-new.png)

![vllm serve model](../../../../assets/vllm/blog/architecture/vllm-tpu/03-vllm-serve-model.png)

![llama3 8b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/04-llama3-8b-throughput-progress.png)

![llama3 70b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/05-llama3-70b-throughput-progress.png)
