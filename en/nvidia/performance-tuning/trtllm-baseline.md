---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html
lang: en
fetched: 2026-08-31
---

# Benchmarking Default Performance

Case study: Llama-3.3-70B, TP=4 on H100 SXM 80GB. Numbers are illustrative.

LLM-API `LLM(model=..., tensor_parallel_size=4)` converts the checkpoint and builds the engine. Guard with `if __name__ == "__main__"` (mpi4py). Some environments need `mpirun -n 1 --oversubscribe --allow-run-as-root python script.py` — `-n 1` is intentional; TRT-LLM spawns the other ranks. Gated HF models need access + login.

`llm.save("baseline")` writes the engine. CLI alternative: `convert_checkpoint.py` then `trtllm-build`.

Dataset (1000 req, ISL/OSL 2048/2048):

```bash
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer /path/to/Llama-3.3-70B-Instruct/ \
  token-norm-dist --input-mean 2048 --output-mean 2048 --input-stdev 0 --output-stdev 0 \
  --num-requests 1000 > synthetic_2048_2048.txt
```

Throughput (all 1000 issued immediately; ~20 min in NVIDIA’s 4×H100 test):

```bash
trtllm-bench --model ... throughput --dataset ... --engine_dir .../baseline
```

Latency forces batch size 1. 100 requests ~1.5 h in the study; 10 is usually enough to iterate.

Baseline used in later pages:

| Metric | Value |
|---|---|
| Token Throughput (tokens/sec) | 1564.3040 |
| Request Throughput (req/sec) | 0.7638 |
| Average TTFT (ms) | 147.6976 |
| Average ITL (ms) | 31.3276 |
