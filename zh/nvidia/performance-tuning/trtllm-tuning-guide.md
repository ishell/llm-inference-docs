---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# TensorRT-LLM 性能调优指南

默认就不差。真要为你的负载榨出更多，靠的是这本手册里的旋钮，外加 `trtllm-bench`。它也是一份 LLM-API + bench 工作流的例题。

官方案例贯穿全书：**Llama-3.3-70B**，**4×H100-SXM-80GB**（NVLink），ISL/OSL **2048/2048**。数字是内部测试，用来说明方向，不是你机器上的天气预报。环境、SKU、互联、负载一变，成绩单就会换脸。

先读同目录 `mastering-llm-techniques.md`：prefill / decode、inflight batching、TP / PP、量化。不识这四样，旋钮只是一排没有门牌的开关。

## 怎么读（官方目录）

| 章 | 本地 | 原文 |
|---|---|---|
| 1. 打一条基线 | [trtllm-baseline.md](trtllm-baseline.md) | [benchmarking-default-performance](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html) |
| 2. 编译期旗标 | [trtllm-build-flags.md](trtllm-build-flags.md) | [useful-build-time-flags](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html) |
| 3. max batch / max tokens | [trtllm-max-batch.md](trtllm-max-batch.md) | [tuning-max-batch-size-and-max-num-tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html) |
| 4. 怎么切卡 | [trtllm-sharding.md](trtllm-sharding.md) | [deciding-model-sharding-strategy](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html) |
| 5. FP8 | [trtllm-fp8.md](trtllm-fp8.md) | [fp8-quantization](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) |
| 6. 运行时旗标 | [trtllm-runtime-flags.md](trtllm-runtime-flags.md) | [useful-runtime-flags](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html) |

邻居（不是同一本 TOC，但调的时候会伸手去翻）：

- [trtllm-bench.md](trtllm-bench.md) — `trtllm-bench` 命令页
- [blog-03-tensorrt-llm.md](blog-03-tensorrt-llm.md) — 博客版：bench 完抄到 `trtllm-serve`
- [trtllm-paged-attention-ifb.md](trtllm-paged-attention-ifb.md) / [trtllm-kvcache.md](trtllm-kvcache.md) — 调度与记忆

案例里相对完全没调的 baseline，开齐编译旗标大约 token/s **+31%**、ITL **−54%**；再调 max batch / max tokens 大约再 **+21%** 吞吐；再上 FP8，同一指南里相对调过的 FP16 约 **+144%** token/s。每一步都要自己测质量。地图给出方向，不代替你的秒表。
