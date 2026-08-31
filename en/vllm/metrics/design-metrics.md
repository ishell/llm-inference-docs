---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: en
fetched: 2026-08-31
---

# Metrics design (vLLM V1)

How `/metrics` is built. The **name → meaning** table for operators is `production-metrics.md`.

## Split

- **Server-level** (gauges/counters): engine state — running/waiting counts, KV usage, token totals. These *explain* request SLOs.
- **Request-level** (histograms): TTFT, ITL, e2e, prompt/decode sizes — typically the SLOs.

Prefix: `vllm:`. Grafana example dashboard exists in the repo; the subset it plots is the “important” list (e2e, TTFT, ITL, KV%, running/waiting, token histograms, queue/prefill/decode time).

## Where the work runs

V1: **EngineCore** is the inner GPU loop — keep it thin. Bookkeeping lives in the frontend (`AsyncLLM.output_handler_loop`) using `EngineCoreOutputs`. Intervals use **`time.monotonic()` in one process** (monotonic clocks are not comparable across processes).

Engine-core events: `QUEUED` → `SCHEDULED` → `NEW_TOKENS` (plus `PREEMPTED`). Frontend derives queue / prefill / decode / inference / inter-token intervals from those. **TTFT** is measured from frontend `arrival_time` (tokenization start) so input processing is included.

Preempt during decode stretches ITL/decode/inference. Preempt during prefill stretches TTFT/prefill.

`--api-server-count > 1` uses prometheus multiprocess mode; process-level python_gc_* / process_* metrics are then missing.

## Publishing

- Log every 5s: running/waiting, GPU cache %, prompt/gen tokens per second, prefix-cache hit rate over last 1k block queries.
- Prometheus `/metrics`: Counter / Gauge / Histogram, labelled with `model_name`. Histogram buckets will keep evolving.
- `vllm:cache_config_info`: startup config as labels (block size, prefix cache on/off, gpu_memory_utilization, …).
- Optional KV residency histograms (`--kv-cache-metrics-sample`): block lifetime, idle-before-evict, reuse gap.

HTTP request metrics come from `prometheus_fastapi_instrumentator` (`http_requests_total`, durations, …).
