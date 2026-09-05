---
source: https://docs.vllm.ai/en/stable/usage/metrics/
lang: en
fetched: 2026-09-05
---

# Production metrics — vLLM

Chinese: [zh/vllm/metrics/production-metrics.md](../../../zh/vllm/metrics/production-metrics.md)  
How the names are computed: `design-metrics.md`. Client stopwatches (AIPerf / `vllm bench serve`) live outside the door; `/metrics` lives inside. The two clocks can disagree — compare formulas, not the three letters “TTFT”.

The OpenAI-compatible API server exposes Prometheus at `/metrics`.

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
curl http://0.0.0.0:8000/metrics
```

Prefix `vllm:`, labelled `model_name`. Histogram buckets will keep evolving — a dashboard that hard-codes old buckets will look “broken” after a few releases.

## Closest to SLOs

| Metric | Type | Meaning |
|---|---|---|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT. Design page: from frontend `arrival_time` (tokenization start); input processing is included |
| `vllm:inter_token_latency_seconds` | Histogram | ITL |
| `vllm:e2e_request_latency_seconds` | Histogram | end-to-end |
| `vllm:request_prefill_time_seconds` | Histogram | time in PREFILL |
| `vllm:request_decode_time_seconds` | Histogram | time in DECODE |
| `vllm:request_queue_time_seconds` | Histogram | WAITING |
| `vllm:kv_cache_usage_perc` | Gauge | KV usage, **1 = 100%** |
| `vllm:num_requests_running` | Gauge | executing |
| `vllm:num_requests_waiting` | Gauge | queued |
| `vllm:prefix_cache_hits` / `queries` | Counter | prefix-cache hit / query **tokens** (not requests) |
| `vllm:num_preemptions` | Counter | cumulative preemptions. When this climbs, e2e and ITL usually shake |
| `vllm:generation_tokens` / `prompt_tokens` | Counter | decode / prefill tokens processed |

The repo ships a Grafana example. The subset it plots is the official “important” list: e2e, TTFT, ITL, KV%, running/waiting, token histograms, queue / prefill / decode time. Wire that subset first; collect the full table later.

The full table (spec-decode counters, LoRA, parallelism, tokenizer, …) lives on the official page and changes by release. Copying the generated table into git becomes a lie in three months.

## Deprecation Policy

A metric deprecated in `X.Y` is **hidden** in `X.Y+1` (re-enable with `--show-hidden-metrics-for-version=X.Y`) and **removed** in `X.Y+2`. Why the long fuse: [design-metrics.md](design-metrics.md) (`vllm:avg_prompt_throughput_toks_per_s` was gone before a user noticed).

## Nearby families

- **Speculative decoding**: acceptance length, draft rate, and similar counters. The ITL/TPOT fork in `vllm bench serve` has an engine-side counterpart here.
- **NIXL KV transfer**: histograms when P/D or cross-instance KV is on. Ops face of the Mooncake / connector posts.
- **MFU**: only with `--enable-mfu-metrics`. Off by default; it costs to compute.
- **HTTP**: `prometheus_fastapi_instrumentator` (`http_requests_total`, …). Door counts, not token gaps.

`--api-server-count > 1` uses Prometheus multiprocess. Process-level `python_gc_*` / `process_*` then vanish. Not broken — bookkeeping moved rooms.

## The five-second log line

You can see the weather without scraping Prometheus. About every 5s: running/waiting, GPU cache %, prompt/gen tokens per second, prefix-cache hit rate over the last 1k blocks. `vllm:cache_config_info` nails startup config (block size, prefix cache on/off, `gpu_memory_utilization`, …) as labels; a config change is a new timeline.

Optional `--kv-cache-metrics-sample`: block lifetime, idle-before-evict, reuse gap. If the question is “is KV sitting idle”, turn this on; occupancy percent alone will not say.
