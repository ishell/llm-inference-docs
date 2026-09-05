---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: en
fetched: 2026-09-04
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

`--api-server-count > 1` uses Prometheus multiprocess. Process-level `python_gc_*` / `process_*` then vanish from `/metrics` — not a scraper bug; multiprocess mode does not expose the single-process set. Collection historically lived in the engine-core process ([#7279](https://github.com/vllm-project/vllm/pull/7279)); now it lives in the API server, and multiprocess is **only** for `--api-server-count > 1` ([#17546](https://github.com/vllm-project/vllm/pull/17546)).

Library path: `aioprometheus` → `prometheus_client`. HTTP metrics vanished briefly in that migration, then came back via `prometheus_fastapi_instrumentator` ([#15657](https://github.com/vllm-project/vllm/pull/15657)): `http_requests_total`, request/response size, duration — door counts, not token gaps.

## Intervals (engine-core events)

Events recorded in the **engine-core process** (frontend cannot see `QUEUED` / `SCHEDULED` timing):

| Event | Meaning |
|---|---|
| `QUEUED` | Received by engine core, in scheduler queue |
| `SCHEDULED` | First scheduled for execution |
| `PREEMPTED` | Back to waiting; will re-schedule and **re-start prefill** |
| `NEW_TOKENS` | Tokens in this `EngineCoreOutput` (one timestamp on the whole `EngineCoreOutputs`) |

Derived:

- Queue: `QUEUED` → most recent `SCHEDULED`
- Prefill: most recent `SCHEDULED` → first following `NEW_TOKENS`
- Decode: that first `NEW_TOKENS` → last `NEW_TOKENS`
- Inference: most recent `SCHEDULED` → last `NEW_TOKENS`
- ITL: successive `NEW_TOKENS`

TTFT is **not** that prefill interval: frontend measures from `arrival_time` (tokenization start) so input processing is inside the first wait. E2E is frontend `arrival_time` → last token received at the frontend.

Docs-site figures: common case; preempt-during-decode (ITL / decode / inference stretch); preempt-during-prefill (TTFT / prefill stretch).

Frontend, per engine-core iteration: new tokens, prompt tokens from prefills that finished, queue intervals for newly scheduled requests, prefill intervals / TTFT for prefills that completed, ITL for everyone in the iteration.

## KV residency

`--kv-cache-metrics-sample` keeps overhead tiny. Sampled block: lifetime (alloc → evict), idle-before-evict (last touch → evict), reuse gaps. Prometheus: `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds`. Engine core ships raw eviction events in `SchedulerStats`; frontend observes Prometheus **and** `LLM.get_metrics()` when logging is on.

## Publishing

- `LoggingStatLogger`: INFO about every 5s — running/waiting, GPU cache %, prompt/gen tok/s over 5s, prefix-cache hit rate over last **1k** block queries.
- `PrometheusStatLogger`: `/metrics`. Counter never decreases until restart; Gauge goes both ways; Histogram is bucketed samples. Every series labelled `model_name`. **Buckets will keep evolving.** Histogram example on the page: TTFT `le="0.02"` already has 13 of 140 requests, etc. — do not copy buckets into a dashboard as law.
- `vllm:cache_config_info`: Info-metric idea (Gauge stuck at 1) with startup labels (`block_size`, `cache_dtype`, `cpu_offload_gb`, `enable_prefix_caching`, `gpu_memory_utilization`, …). `prometheus_client` never supported Info in multiprocess mode, so this is a Gauge with `multiprocess_mode="mostrecent"`.
- LoRA: `vllm:lora_requests_info` Gauge whose **value is wall-clock time**, updated every iteration. Labels: `running_lora_adapters` / `waiting_lora_adapters` as comma-separated `adapter=count` strings, plus `max_lora`. The page itself calls packing counts into a CSV string “quite misguided”. `multiprocess_mode="livemostrecent"`. Added [#9477](https://github.com/vllm-project/vllm/pull/9477); at least one known downstream (Gateway API Inference Extension). Coordinate before removing.
- Prefix cache: every query records tokens queried vs tokens hit. Logs expose a **1k-query** hit rate. Prometheus should keep **counters** so PromQL can pick the window: `rate(cache_query_hit[5m]) / rate(cache_query_total[5m])` — not a Gauge of hit rate.

## Deprecation

Do not drop a metric lightly (`vllm:avg_prompt_throughput_toks_per_s` was deprecated, removed, then a user noticed). Policy sketched on the page: caution; deprecation in the `/metrics` HELP string; user docs + release notes; hide behind a CLI escape hatch for a while. Project-wide: contributing deprecation policy. Production page: hide in `X.Y+1` via `--show-hidden-metrics-for-version=X.Y`, delete in `X.Y+2`.

Named leftovers: unimplemented `vllm:tokens_total` (just remove); duplicated queue-time series; prefix-cache hit **rate** Gauge vs counters; KV-offload metrics still evolving.

## Future work (named, not promised)

Parallel sampling metrics; speculative-decoding counters matching the feature page; autoscaling / load-balancing signals; naming consistency (`_total` suffixes, `vllm:` prefix); “adding more metrics” should prefer engine-core events + frontend intervals over new work in the inner loop.

## Tracing

OpenTelemetry is a **separate** path from Prometheus histograms. Model-forward vs execute time are distinguished in that section — do not treat a span duration as TTFT.

HTTP metrics remain door-open counts. Using them as a stand-in for TTFT is a polite lie.


- Log about every 5s: running/waiting, GPU cache %, prompt/gen tokens per second, prefix-cache hit rate over the last 1k blocks. Heartbeat without Grafana.
- Prometheus `/metrics`: Counter / Gauge / Histogram, labelled `model_name`. **Buckets will keep evolving.**
- `vllm:cache_config_info`: startup config as labels (block size, prefix cache on/off, `gpu_memory_utilization`, …). A config change is a new series; do not overlay two curves on the same labels and argue.
- Optional KV residency histograms (`--kv-cache-metrics-sample`): block lifetime, idle-before-evict, reuse gap. Occupancy says the room is full; residency says whether people live there or just pass through.

HTTP metrics come from `prometheus_fastapi_instrumentator` (`http_requests_total`, durations, …). Door-open counts, not gaps between tokens. Using them as a stand-in for TTFT is a polite lie.
