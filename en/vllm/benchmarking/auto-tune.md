---
source: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
lang: en
fetched: 2026-09-01
---

# vLLM auto_tune.sh

In-tree script: grid-search `max-num-seqs` × `max-num-batched-tokens` for the highest throughput that still lives inside optional constraints (P99 e2e, prefix-cache hit rate). It tunes the batch knobs that `optimization.md` places before TP/DP, not the shard count.

Do **not** put the substring `vllm` in the script path. The script runs `pkill -f vllm` and will kill itself. Use `tmux` / `screen`; this takes a long time.

## Required environment

Edit the script header or override at launch. `BASE` must be the **absolute parent** of the vLLM clone.

```bash
MODEL=meta-llama/Llama-3.3-70B-Instruct \
SYSTEM=GPU TP=8 DOWNLOAD_DIR='' \
INPUT_LEN=128 OUTPUT_LEN=2048 MAX_MODEL_LEN=2300 \
MIN_CACHE_HIT_PCT=0 MAX_LATENCY_ALLOWED_MS=500 \
NUM_SEQS_LIST="128 256" NUM_BATCHED_TOKENS_LIST="1024 2048 4096" \
bash auto_tune.sh
```

| Variable | Meaning |
|---|---|
| `BASE` | Absolute parent of the vLLM repo |
| `MODEL` | Hugging Face id |
| `SYSTEM` | `TPU` or `GPU` (other hardware may not save profiles) |
| `TP` | tensor parallel |
| `DOWNLOAD_DIR` | Weight dir; empty = default download |
| `INPUT_LEN` / `OUTPUT_LEN` / `MAX_MODEL_LEN` | Synthetic request size and window |
| `MIN_CACHE_HIT_PCT` | Prefix-cache hit-rate floor, 0–100; `0` disables |
| `MAX_LATENCY_ALLOWED_MS` | P99 e2e cap. A huge number ≈ ignore latency |
| `NUM_SEQS_LIST` | `max-num-seqs` values to sweep |
| `NUM_BATCHED_TOKENS_LIST` | `max-num-batched-tokens` values to sweep |

Default lists assume medium ISL/OSL. Very short context (e.g. 20/20) often wants larger `max-num-seqs`. Install the matching env first; TPU needs its own conda / torch_xla. Custom models must have configs where the server can see them.

## Three goals

1. **Throughput only**: astronomical `MAX_LATENCY_ALLOWED_MS` (docs example `100000000000`), `MIN_CACHE_HIT_PCT=0`.
2. **Throughput + P99**: e.g. `MAX_LATENCY_ALLOWED_MS=500`.
3. **Plus prefix cache**: `MIN_CACHE_HIT_PCT=60`. Hit rate is a constraint, not a bonus.

## How it walks

1. From `gpu-memory-utilization=0.98` downward, find the highest value that does not OOM. Every later cell shares that cup.
2. Nested loop over seqs × batched-tokens.
3. Per cell: start the server → `vllm bench serve --request-rate inf`. If P99 already fits, that throughput is the cell’s ceiling. If not, **lower request-rate** until latency fits — highest *sustainable* throughput under the SLA, not the lab `inf` curve.
4. Keep the best valid cell; save a profiler only for the winner (JSON trace on GPU, `.xplane.pb` on TPU).

## Artifacts

`$BASE/auto-benchmark/<YYYY_MM_DD_HH_MM>/`:

- `vllm_log_...txt` / `bm_log_...txt` — per-cell server and bench logs
- `result.txt` — one line per cell, then best_*
- `profile/` — winner only

```
max_num_seqs: 128, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 450.5, throughput: 9.8, goodput: 9.8
max_num_seqs: 128, max_num_batched_tokens: 4096 does not meet latency requirement 500
best_max_num_seqs: 256, best_num_batched_tokens: 2048, best_throughput: 12.5, profile saved in: ...
```

No legal cell: `best_max_num_seqs: 0` (server never came up, or the latency cap is cruel).

## Batch: `batch_auto_tune.sh`

JSON array, one `auto_tune.sh` per object. Needs `jq`. Optional second arg uploads to GCS (`gcloud` must already be logged in).

```bash
bash batch_auto_tune.sh runs_config.json [gs://bucket/path]
```

Keys match the variables above (lowercase; the script uppercases them). It **rewrites the JSON in place**: `run_id`, `status` (`SUCCESS` / `FAILURE` / `WARNING_NO_RESULT_FILE`), `results`, optional `gcs_results`.

auto_tune finds a cell for this machine, this ISL/OSL, this SLA. Change the GPU or the context and the grid is void. Treat the winner as a starting point for the two flags in `serve.md`, not as eternal truth.
