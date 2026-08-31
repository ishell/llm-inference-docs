---
source: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
lang: en
fetched: 2026-08-31
---

# vLLM auto_tune.sh

Sweeps `max-num-seqs` × `max-num-batched-tokens` to maximize throughput, with optional P99 e2e latency and prefix-cache hit-rate constraints.

1. Find highest `gpu-memory-utilization` that does not OOM (from 0.98 down).
2. For each combo: start server → bench `--request-rate inf` → if P99 too high, lower rate until it fits.
3. Keep the best valid throughput; save profiler trace for that run.

Do not put `vllm` in the script path (`pkill -f vllm` would kill the tuner).

Env example: `MODEL=... INPUT_LEN=... OUTPUT_LEN=... MAX_LATENCY_ALLOWED_MS=500 NUM_SEQS_LIST="128 256" NUM_BATCHED_TOKENS_LIST="1024 2048 4096" bash auto_tune.sh`

Results under `$BASE/auto-benchmark/<timestamp>/` (`result.txt` + logs). `best_max_num_seqs: 0` means nothing met the constraint.

`batch_auto_tune.sh` runs many jobs from a JSON array (needs `jq`).
