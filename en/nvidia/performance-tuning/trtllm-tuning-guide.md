---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
lang: en
fetched: 2026-08-31
---

# TensorRT-LLM Performance Tuning Guide (index)

Defaults are decent; this guide’s knobs plus `trtllm-bench` are how you extract more. Case study in the docs: Llama-3.3-70B, 4×H100 NVLink, ISL/OSL 2048/2048.

Prereqs: prefill vs decode, inflight batching, TP/PP, quantization. See `mastering-llm-techniques.md` in this folder.

Child pages (local):

- `trtllm-build-flags.md` — multiple profiles, paged context attention, GEMM plugin
- `trtllm-runtime-flags.md` — scheduler, KV fraction, sliding window
- `trtllm-kvcache.md` — paged KV, reuse, offload
- `trtllm-bench.md` — `trtllm-bench` CLI (English full fetch)
- `blog-03-tensorrt-llm.md` — NVIDIA blog walkthrough

Index: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
