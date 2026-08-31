---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
lang: en
fetched: 2026-08-30
---

# Part 3: Performance Tuning with TensorRT-LLM

Source: https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/

Reset GPU clocks/power (`nvidia-smi -rgc/-rmc`, then `-pl` if needed).

## trtllm-bench

Benchmarks the engine without a full HTTP stack:

```bash
trtllm-bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset dataset.jsonl \
  --tp 1 --backend pytorch \
  --report_json results.json --streaming \
  --concurrency $CONCURRENCY
```

Custom jsonl row: `{"task_id": 1, "prompt": "...", "output_tokens": 128}`.

Read PERFORMANCE OVERVIEW (req/s, token/s, TTFT, TPOT, per-user speed) and Max Runtime Batch Size / Max Runtime Tokens.

- Max tokens: cap on tokens per engine iteration (sum of context tokens + 1 per generation request).
- Max batch size: max requests per iteration; can bind before token budget is used.

Sweep `--concurrency` and plot per-GPU throughput vs per-user speed. NVIDIA’s example: target ~50 tok/s/user. FP16 Llama-3.1 8B ~256 concurrent users; FP8 ~512 within the same budget.

Multi-GPU: `--tp --pp --ep`. Extra knobs: `--extra_llm_api_options`.

## trtllm-serve

Does not auto-apply bench settings. Pass `--max_num_tokens`, `--max_batch_size`, CUDA-graph yaml, then verify with AIPerf.

Tuning guide: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/
