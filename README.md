# LLM 推理压测与性能文档（中英对照）

个人学习笔记。原文来自 NVIDIA / vLLM 官方文档与博客，英文稿在 `en/`，中文译文在 `zh/`。

**不是官方译本。** 公式、CLI、指标名尽量保留英文。图表在原文网页里，本地 md 只保留图注。

## 建议阅读顺序

1. [中文](zh/nvidia/developer-blog/01-llm-benchmarking-fundamental-concepts.md) / [英文](en/nvidia/developer-blog/01-llm-benchmarking-fundamental-concepts.md) — 指标与压测 vs 性能测试
2. [中文](zh/nvidia/nim-benchmarking/01-overview.md) → [指标](zh/nvidia/nim-benchmarking/02-metrics.md) → [参数](zh/nvidia/nim-benchmarking/03-parameters.md) → [AIPerf 实操](zh/nvidia/nim-benchmarking/04-quickstart.md)
3. [vLLM 调优](zh/vllm/docs/optimization.md) / [英文](en/vllm/docs/optimization.md)
4. [vLLM 解剖博客](en/vllm/blog/2025-09-05-anatomy-of-vllm.md)（英文已存；中文待译）

## 目录

### NVIDIA NIM Benchmarking Guide

| 文件 | 原文 |
|---|---|
| `nim-benchmarking/01-overview.md` | https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html |
| `nim-benchmarking/02-metrics.md` | https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html |
| `nim-benchmarking/03-parameters.md` | https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html |
| `nim-benchmarking/04-quickstart.md` | https://docs.nvidia.com/nim/benchmarking/llm/latest/quickstart.html |
| `nim-benchmarking/05-benchmarking-lora.md` | https://docs.nvidia.com/nim/benchmarking/llm/latest/benchmarking-lora.html |

### NVIDIA Developer Blog 系列

| 文件 | 原文 |
|---|---|
| `developer-blog/01-llm-benchmarking-fundamental-concepts.md` | https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/ |
| `developer-blog/02-genai-perf-and-nim.md` | https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/ |
| `developer-blog/03-performance-tuning-tensorrt-llm.md` | https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/ |
| `developer-blog/04-how-much-does-inference-cost.md` | https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ |
| `developer-blog/mastering-llm-techniques-inference-optimization.md` | https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ |

### vLLM 文档

| 文件 | 原文 |
|---|---|
| `vllm/docs/quickstart.md` | https://docs.vllm.ai/en/stable/getting_started/quickstart/ |
| `vllm/docs/optimization.md` | https://docs.vllm.ai/en/stable/configuration/optimization/ |
| `vllm/docs/benchmark-cli.md` | https://docs.vllm.ai/en/stable/benchmarking/cli/ |
| `vllm/docs/metrics.md` | https://docs.vllm.ai/en/stable/usage/metrics/ |
| `vllm/docs/automatic-prefix-caching.md` | https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/ |

### vLLM 博客

| 文件 | 原文 |
|---|---|
| `vllm/blog/2025-09-05-anatomy-of-vllm.md` | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |

## 范围说明

**已经落地（第一批）**

- NVIDIA NIM Benchmarking Guide：五章英文 + 五章中文全译
- NVIDIA 博客四篇：英文笔记 + 中文摘译；第 1 篇中文较完整
- vLLM：optimization / prefix cache / metrics / bench CLI / quickstart 中英；optimization 中文接近全文
- `Mastering LLM Techniques` 与 vLLM Anatomy：**英文全文 + 中文导读**（未逐段全译）

**刻意没做**

- vLLM 博客全量（100+ 篇）
- TensorRT-LLM Performance Tuning Guide 逐页
- Triton Perf Analyzer / Model Analyzer 全文

需要再补哪一类直接说。抓取日期：2026-08-30。官方页面会改版。
