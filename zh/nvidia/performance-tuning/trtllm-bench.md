---
source: https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
lang: zh
fetched: 2026-08-31
---

# trtllm-bench（中文导读）

英文全文：`en/nvidia/performance-tuning/trtllm-bench.md`

流程：准备 jsonl 数据集 → `trtllm-bench build`（非 PyTorch flow 才要）→ `throughput`（打满）或 `latency`（低延迟）。

`throughput` 会按数据集 ISL/OSL 启发式调引擎；也可手写 `--max_batch_size` / `--max_num_tokens`（默认 2048 / 8192）。支持 `--tp_size` × `--pp_size`（world size ≤ 8）。

博客实操见 `blog-03-tensorrt-llm.md`。
