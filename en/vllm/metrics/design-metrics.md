---
source: https://docs.vllm.ai/en/stable/design/metrics/
lang: en
fetched: 2026-09-05
---

# Metrics design — vLLM V1

Chinese: [zh/vllm/metrics/design-metrics.md](../../../zh/vllm/metrics/design-metrics.md)  
Operator table: [production-metrics.md](production-metrics.md). Source: https://docs.vllm.ai/en/stable/design/metrics/  
Interval figures live on the docs site (`assets/design/metrics/intervals-*.png`); not copied here.

Which room, and which clock, produces those histograms. Much of the design was planned under ["Even Better Observability"](https://github.com/vllm-project/vllm/issues/3616).

## Objectives

- Cover engine-level and request-level metrics for production monitoring.
- Prefer Prometheus — that is what they expect in production.
- Keep logging (INFO metrics) for ad-hoc tests, debugging, and exploration.

## Split

1. **Server-level** (Gauge / Counter): weather inside the engine — running, waiting, KV occupancy, token totals. These *explain* why a request SLO broke.
2. **Request-level** (Histogram): TTFT, ITL, e2e, prompt/decode sizes. Those are usually the SLO itself.

Mental model: server-level explains request-level.

### Named V1 `/metrics` examples

Prefix `vllm:`. Full tables live on the production page; the design page names:

- Gauges: `vllm:num_requests_running`, `vllm:kv_cache_usage_perc` (0–1)
- Counters: `vllm:prefix_cache_queries` / `hits`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_success_total` (by finish reason)
- Histograms: `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`, `vllm:time_to_first_token_seconds` (TTFT), `vllm:inter_token_latency_seconds` (ITL), `vllm:e2e_request_latency_seconds`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`

### Grafana subset

Repo example: [examples/observability/prometheus_grafana](https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana). [#2316](https://github.com/vllm-project/vllm/pull/2316) is the background for this subset:

`e2e_request_latency_seconds_bucket`, `prompt_tokens`, `generation_tokens`, `inter_token_latency_seconds` (the page also calls ITL TPOT here), `time_to_first_token_seconds`, `num_requests_running` (also `_swapped` / `_waiting`), `kv_cache_usage_perc`, `request_prompt_tokens`, `request_generation_tokens`, `request_success`, `request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `request_max_num_generation_tokens`.

`_swapped` is a V1 leftover — see deprecation below.

### Prometheus client and HTTP layer

First [aioprometheus](https://github.com/vllm-project/vllm/pull/1890), then quickly [prometheus_client](https://github.com/vllm-project/vllm/pull/2730). HTTP `MetricsMiddleware` vanished during that migration and came back via `prometheus_fastapi_instrumentator` ([#15657](https://github.com/vllm-project/vllm/pull/15657)):

```bash
curl http://0.0.0.0:8000/metrics | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
# http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
# http_request_size_bytes_count / http_response_size_bytes_count / http_request_duration_*
```

Door-open counts, not token gaps. Using them as a stand-in for TTFT is a polite lie.

### Multi-process mode

Historically collected in the engine-core process and published into the API server via multiprocess ([#7279](https://github.com/vllm-project/vllm/pull/7279)). Now collected in the API server; multiprocess is **only** for `--api-server-count > 1` ([#17546](https://github.com/vllm-project/vllm/pull/17546)).

`prometheus_client` also ships `python_gc_*`, `python_info`, `process_virtual_memory_bytes`, `process_resident_memory_bytes`, `process_start_time_seconds`, `process_cpu_seconds_total`, `process_open_fds` / `process_max_fds`. Multiprocess mode does not expose them, so they vanish from `/metrics` when `--api-server-count > 1`. They also do not aggregate every process in a vLLM instance, so relevance was already questionable.

## Where the work runs

V1 keeps **EngineCore** as the inner GPU loop — thin on purpose. `AsyncLLM` is the outer loop, ideally overlapped with the GPU. Bookkeeping lives in the frontend: `AsyncLLM.output_handler_loop` consumes `EngineCoreOutputs`. Implementation PRs hang off [#10582](https://github.com/vllm-project/vllm/issues/10582) ([#11962](https://github.com/vllm-project/vllm/pull/11962) and the rest of that cluster). Legacy PRs: #1890, #2316, #2730, #4464, #7279.

Intervals must use **`time.monotonic()` in one process**, not `time.time()` (NTP moves the wall clock). Monotonic clocks are not comparable across processes. That is why TTFT is not subtracted from `arrival_time` inside the GPU process.

The scheduler ships scheduled / waiting counts in `EngineCoreOutputs`.

## Intervals (engine-core events)

Timestamps recorded in the **engine-core process** (the frontend cannot see `QUEUED` / `SCHEDULED` timing):

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

Docs-site figures: common case; preempt-during-decode (ITL / decode / inference stretch; already-generated tokens are reused); preempt-during-prefill (TTFT / prefill stretch).

Frontend, per engine-core iteration: new tokens, prompt tokens from prefills that finished, queue intervals for newly scheduled requests, prefill intervals / TTFT for prefills that completed, ITL for everyone in the iteration (the page also writes TPOT here). Completed requests also record inference / decode and e2e.

## KV residency

`--kv-cache-metrics-sample` keeps overhead tiny. Sampled block: lifetime (alloc → evict), idle-before-evict (last touch → evict), reuse gaps. Prometheus: `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds`. Engine core ships raw eviction events in `SchedulerStats`; frontend observes Prometheus **and** `LLM.get_metrics()` when logging is on. Lifetime vs idle on one chart spots stranded cache or long decode pinning a prompt.

## Publishing

- `LoggingStatLogger`: INFO about every 5s — running/waiting, GPU cache %, prompt/gen tok/s over 5s, prefix-cache hit rate over last **1k** block queries.
- `PrometheusStatLogger`: `/metrics` for Prometheus to scrape (the page says e.g. every second). Counter never decreases until restart; Gauge goes both ways; Histogram is bucketed samples. Every series labelled `model_name`. **Buckets will keep evolving** — “useful for everyone” is not a one-shot choice. Histogram example: TTFT `le="0.02"` already has 13 of 140 requests; `le="0.1"` reaches 140. Do not copy buckets into a dashboard as law. `request_success_total` is labelled `finished_reason`: `stop` / `length` / `abort`.
- `vllm:cache_config_info`: Info-metric idea (Gauge stuck at 1) with startup labels (`block_size`, `cache_dtype`, `cpu_offload_gb`, `enable_prefix_caching`, `gpu_memory_utilization`, …). `prometheus_client` never supported Info in multiprocess mode, so this is a Gauge with `multiprocess_mode="mostrecent"`.
- LoRA: `vllm:lora_requests_info` Gauge whose **value is wall-clock time**, updated every iteration. Labels: `running_lora_adapters` / `waiting_lora_adapters` as comma-separated `adapter=count` strings, plus `max_lora`. The page itself calls packing counts into a CSV string “quite misguided”; labels per adapter would be better. `multiprocess_mode="livemostrecent"`. Added [#9477](https://github.com/vllm-project/vllm/pull/9477); at least one known downstream (Gateway API Inference Extension). Coordinate before removing.
- Prefix cache: every query records tokens queried vs tokens hit. Logs expose a **1k-query** hit rate. Prometheus should keep **counters** so PromQL can pick the window: `rate(cache_query_hit[5m]) / rate(cache_query_total[5m])` — not a Gauge of hit rate. Discussion in [#10582](https://github.com/vllm-project/vllm/issues/10582).

## Deprecation

Do not drop a metric lightly. `vllm:avg_prompt_throughput_toks_per_s` was [deprecated](https://github.com/vllm-project/vllm/pull/2764), [removed](https://github.com/vllm-project/vllm/pull/12383), then [noticed by a user](https://github.com/vllm-project/vllm/issues/13218). Policy on the page:

1. Be cautious; user impact is hard to predict.
2. Put a prominent deprecation in the `/metrics` HELP string.
3. List deprecated metrics in user docs and release notes.
4. Hide behind a CLI escape hatch for a while (they cite Kubernetes [show-hidden-metrics](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)). Production page: deprecated in `X.Y` → hidden in `X.Y+1`, re-enable with `--show-hidden-metrics-for-version=X.Y` → removed in `X.Y+2`. Project-wide policy is a separate contributing page.

Named leftovers:

- **Unimplemented:** `vllm:tokens_total` ([#4464](https://github.com/vllm-project/vllm/pull/4464)) — just remove.
- **Duplicated queue time:** `vllm:time_in_queue_requests` ([#9659](https://github.com/vllm-project/vllm/pull/9659), `now - arrival_time`) vs later `vllm:request_queue_time_seconds` (Grafana uses the latter). Deprecate the former.
- **Prefix cache hit-rate Gauge:** replaced by queries / hits counters.
- **KV swapped leftovers:** `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc`. V1 does not GPU↔CPU-swap; `--swap-space` is gone. History: beam-search SequenceGroup shared prompt KV with copy-on-write branching; prefix caching later won (near-zero overhead in V1, on by default) and preemption became recompute. SequenceGroup was removed in V1; parallel sampling (`n>1`) still needs a stand-in. Beam search left the core.

## Future work (named, not promised)

**Parallel sampling** ([#10980](https://github.com/vllm-project/vllm/pull/10980)): `vllm:request_params_n` (the `n` on every finished request); `vllm:request_max_num_generation_tokens` (max output in a sequence group; without parallel sampling this equals `vllm:request_generation_tokens`).

**Speculative decoding:** `vllm:spec_decode_draft_acceptance_rate`, `vllm:spec_decode_efficiency` (Gauges), `vllm:spec_decode_num_accepted_tokens` / `num_draft_tokens` / `num_emitted_tokens` (Counters). Acceptance rate should probably be accepted / draft counters, like prefix cache. They named [#12193](https://github.com/vllm-project/vllm/pull/12193) for ngram on V1.

**Autoscaling / load-balancing:** Kubernetes Serving WG notes, Inference Perf, [#5041](https://github.com/vllm-project/vllm/issues/5041) / [#12726](https://github.com/vllm-project/vllm/pull/12726). Need a saturation signal: raising request rate no longer raises throughput, latency starts to pile. Rob: estimate the max concurrency where average request length > QPS — that is what saturates the server.

**Naming:** colons vs Prometheus “colons are for recording rules”; most names end in units, not all; `_total` is stripped/re-added between OpenMetrics and Prometheus text format.

**Adding more:** easy to add, hard to remove; only useful if on by default, and default-on has a perf tax; maintenance cost grows with the set. Prefer engine-core events + frontend intervals over new work in the inner loop. Inspiration elsewhere: TGI, K8s autoscaling, [OTel Gen AI semantic conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai).

## Tracing

Metrics aggregate over time; tracing follows one request through components. Both are observability; the design page treats them as separate topics.

OpenTelemetry: [added](https://github.com/vllm-project/vllm/pull/4687), [reinstated](https://github.com/vllm-project/vllm/pull/20372). Flags: `--oltp-traces-endpoint`, `--collect-detailed-traces`. User docs: `examples/observability/opentelemetry`.

These Histograms exist only when detailed tracing is on ([#7089](https://github.com/vllm-project/vllm/pull/7089)):

- `vllm:model_forward_time_milliseconds` — time in model forward while this request was in the batch
- `vllm:model_execute_time_milliseconds` — execute: forward + worker block/sync + CPU–GPU sync + sampling

`--collect-detailed-traces=all/model/worker`. The docs say this may be costly or blocking. Spans look like:

```text
gen_ai.latency.time_in_scheduler
gen_ai.latency.time_in_model_forward
gen_ai.latency.time_in_model_execute
```

`inference_time` / `decode_time` already exist. Whether the higher-resolution timings justify the overhead is an open question. Do not treat a span duration as TTFT.
