---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
lang: zh
fetched: 2026-08-31
---

# TensorRT-LLM 性能调优指南（入口）

默认就不差；真要榨性能靠这套旋钮 + `trtllm-bench`。官方案例：Llama-3.3-70B，4×H100 NVLink，ISL/OSL 2048/2048。

先懂：prefill/decode、inflight batching、TP/PP、量化。见同目录 `mastering-llm-techniques.md`。

本目录子页：

- `trtllm-build-flags.md` — 编译期：multiple profiles、paged context、GEMM plugin
- `trtllm-runtime-flags.md` — 运行时：调度、KV 占比、sliding window
- `trtllm-kvcache.md` — paged KV、跨请求 reuse、卸到 CPU
- `trtllm-bench.md` — `trtllm-bench` CLI（英文全文已抓）
- `blog-03-tensorrt-llm.md` — NVIDIA 博客实操

原文目录：https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
