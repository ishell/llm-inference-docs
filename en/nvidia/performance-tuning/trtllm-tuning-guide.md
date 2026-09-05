---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
lang: en
fetched: 2026-08-31
---

# TensorRT-LLM Performance Tuning Guide

Defaults are solid. Extra performance comes from the knobs in this handbook plus `trtllm-bench`. The official pages also double as an LLM-API + bench walkthrough.

Case study throughout: **Llama-3.3-70B**, **4×H100-SXM-80GB** (NVLink), ISL/OSL **2048/2048**. Numbers are internal demos, not a forecast for your box.

Prereqs (prefill/decode, inflight batching, TP/PP, quantization): `mastering-llm-techniques.md`.

## Official TOC (local notes)

| Ch | Local | Official |
|---|---|---|
| 1. Default baseline | [trtllm-baseline.md](trtllm-baseline.md) | [benchmarking-default-performance](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html) |
| 2. Build-time flags | [trtllm-build-flags.md](trtllm-build-flags.md) | [useful-build-time-flags](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html) |
| 3. Max batch / max tokens | [trtllm-max-batch.md](trtllm-max-batch.md) | [tuning-max-batch-size-and-max-num-tokens](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html) |
| 4. Sharding | [trtllm-sharding.md](trtllm-sharding.md) | [deciding-model-sharding-strategy](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html) |
| 5. FP8 | [trtllm-fp8.md](trtllm-fp8.md) | [fp8-quantization](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) |
| 6. Runtime flags | [trtllm-runtime-flags.md](trtllm-runtime-flags.md) | [useful-runtime-flags](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html) |

Neighbors (not in that TOC, but you will open them):

- [trtllm-bench.md](trtllm-bench.md) — CLI reference (long English fetch)
- [blog-03-tensorrt-llm.md](blog-03-tensorrt-llm.md) — blog: copy bench settings into `trtllm-serve`
- [trtllm-paged-attention-ifb.md](trtllm-paged-attention-ifb.md) / [trtllm-kvcache.md](trtllm-kvcache.md) — scheduler and memory

Case-study ladder vs raw baseline: build flags ~**+31%** token/s, ITL **−54%**; then tuned max batch/tokens ~**+21%** more throughput; then FP8 vs that tuned FP16 ~**+144%** token/s. Measure quality at every step.
