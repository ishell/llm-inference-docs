---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: en
fetched: 2026-09-01
---

# Metrics design — vLLM V1

Operator table: `production-metrics.md`. This page is which room, and which clock, produces those histograms.

## Split

- **Server-level** (gauge / counter): weather inside the engine — running, waiting, KV occupancy, token totals. These *explain* why a request SLO broke.
- **Request-level** (histogram): TTFT, ITL, e2e, prompt/decode sizes. Those are usually the SLO itself.

Prefix `vllm:`. The Grafana example dashboard plots the “wire these first” subset.

## Where the work runs

V1 keeps **EngineCore** as the inner GPU loop — thin on purpose. Bookkeeping lives in the frontend: `AsyncLLM.output_handler_loop` consumes `EngineCoreOutputs`.

Intervals must use **`time.monotonic()` in one process**. Monotonic clocks are not comparable across processes — each answers “how long since this process started,” and the difference is meaningless. That is why TTFT is not subtracted from `arrival_time` inside the GPU process.

Engine-core events: `QUEUED` → `SCHEDULED` → `NEW_TOKENS`, plus `PREEMPTED`. The frontend builds queue / prefill / decode / inference / ITL from those.

**TTFT** starts at frontend `arrival_time` (tokenization start), so input processing is in the first wait. Client AIPerf TTFT starts at “HTTP went out.” Inside you pay tokenize; outside you pay the wire. When they disagree, ask which segment is missing from the other clock.

Preempt during decode stretches ITL / decode / inference. Preempt during prefill stretches TTFT / prefill. When `vllm:num_preemptions` climbs, ask which phase is being shown the door.

`--api-server-count > 1` uses Prometheus multiprocess. Process-level `python_gc_*` / `process_*` then vanish from `/metrics` — not a scraper bug; multiprocess mode does not expose the single-process set.

## Publishing

- Log about every 5s: running/waiting, GPU cache %, prompt/gen tokens per second, prefix-cache hit rate over the last 1k blocks. Heartbeat without Grafana.
- Prometheus `/metrics`: Counter / Gauge / Histogram, labelled `model_name`. **Buckets will keep evolving.**
- `vllm:cache_config_info`: startup config as labels (block size, prefix cache on/off, `gpu_memory_utilization`, …). A config change is a new series; do not overlay two curves on the same labels and argue.
- Optional KV residency histograms (`--kv-cache-metrics-sample`): block lifetime, idle-before-evict, reuse gap. Occupancy says the room is full; residency says whether people live there or just pass through.

HTTP metrics come from `prometheus_fastapi_instrumentator` (`http_requests_total`, durations, …). Door-open counts, not gaps between tokens. Using them as a stand-in for TTFT is a polite lie.
