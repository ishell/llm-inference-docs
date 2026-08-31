---
source: catalog of NVIDIA + vLLM official URLs
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 那条评论在指什么

那条评论没贴链接。它几乎是在用 NVIDIA 自己的口吻说话：把 **load testing（压测）** 和 **performance benchmarking（性能测试）** 分开，然后再去做 **performance improvement（调优）**。

我不能把每篇官方网页原样搬进这个仓库——篇幅不允许，版权也不允许。这里是学习笔记：公式、CLI、指标名保留英文；中文用一种愿意把「等待第一个字」当成人的事情来写的笔调。不是官方译本。

对照表见仓库根目录 `README.md`。完整度一览可以在 Cursor 里打开画布。

## NVIDIA 自己怎么切开这三件事

| 评论里的词 | 官方在测什么 | 常用工具 |
|---|---|---|
| testing / performance | 模型在给定负载下的 TTFT、ITL/TPOT、TPS、RPS | **AIPerf**（旧名 GenAI-Perf，已停更） |
| load testing | 系统能不能扛住真实流量：容量、伸缩、网络、资源 | **K6 / Locust** |
| performance improvement | batch、KV cache、量化、并行 | TensorRT-LLM Tuning Guide；vLLM 则是 `optimization` 那一页 |

压测问的是：屋子里突然挤进一百个人，门会不会塌。性能测试问的是：同一个人、同一句话，模型吐字有多稳、有多快。两套都要做。只做其中一套，你会爱上一个不会在生产里活下来的数字。

## 最短路径（评论者真正想让你读的四篇）

1. [基本概念](nvidia/benchmarking/blog-01-fundamental-concepts.md) — TTFT / ITL / TPS，压测 vs 性能测试，concurrency 怎么扫。
2. [用 AIPerf 打一轮](nvidia/benchmarking/nim-04-aiperf.md) — 画出 latency–throughput 曲线。
3. [vLLM 调优顺序](vllm/optimization/optimization.md) — CPU 核、`-O*`、`max_num_batched_tokens`、并行、cache。
4. [Anatomy of vLLM](vllm/blog/architecture/anatomy.md) — 把 serving 系统从里翻到外。

北大超算那套更像「推理是什么」的科普。这条评论要的是会算 TTFT/ITL、会压 concurrency、会调 batch/KV/量化 的工程文档。

## TensorRT-LLM 调优手册

评论里的 **performance improvement**，落到 NVIDIA 文档站，就是 [Performance Tuning Guide](nvidia/performance-tuning/trtllm-tuning-guide.md)。六章按官方目录排：先打基线，再拧编译旗标、max batch / max tokens、切卡、FP8、运行时调度。邻居页是 KV cache、IFB 调度、`trtllm-bench`。案例贯穿全书：Llama-3.3-70B、4×H100、2048/2048。数字是演示，质量每一步都要自己测。

## vLLM 博客（必读线）

文档页 `optimization.md` 告诉你旋钮顺序。博客告诉你这些旋钮是怎么长出来的。完整顺序在 [MUST-READ](vllm/blog/MUST-READ.md)。

**架构：** [立项](vllm/blog/architecture/paged-attention.md) → [Anatomy](vllm/blog/architecture/anatomy.md) → [V1](vllm/blog/architecture/v1-alpha.md) → [MRV2](vllm/blog/architecture/mrv2.md)

**性能：** [v0.6 CPU](vllm/blog/performance/v0.6-throughput.md) → [投机解码](vllm/blog/performance/spec-decode.md) → [FP8 KV](vllm/blog/performance/fp8-kvcache.md) → [生产级 CI](vllm/blog/performance/production-quality.md)

**Serving：** [切卡](vllm/blog/serving/distributed-inference.md) → [production-stack](vllm/blog/serving/production-stack.md) / [AIBrix](vllm/blog/serving/aibrix.md) → [Router](vllm/blog/serving/router.md) → [Encoder 分离](vllm/blog/serving/epd.md) → [Wide-EP](vllm/blog/serving/large-scale.md)

EPD 是视觉编码器拆出去；文本 Prefill/Decode 分离在 Router 与大规模两篇。CATALOG 其余不是这条主线。
