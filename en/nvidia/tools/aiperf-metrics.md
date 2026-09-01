---
source: https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference
lang: en
fetched: 2026-09-01
---

# AIPerf Metrics

The official Metrics Reference is grouped by **computation phase**, not by “what the user felt.” NIM guide chapter 2 (`../benchmarking/nim-02-metrics.md`) is the same ruler in prose. This note adds formulas, preconditions, and why the two throughputs must not be compared.

Streaming metrics require `--streaming`, a token-producing endpoint, and at least one **non-empty** chunk. An empty first packet is not the first word.

## Three computation kinds

| Kind | When | Shape |
|---|---|---|
| Record | Per request (and its response chunks) | Distribution: avg / min / max / p50 / p90 / p99 |
| Aggregate | Across the whole run | Single value: `request_count`, timestamp bounds |
| Derived | After other metrics exist | Single value or another distribution: TPS, RPS, goodput |

Record metrics never see the whole run. Derived metrics never see a single HTTP exchange. System TPS and “how fast this person felt” are different stopwatches.

## Streaming: wait and gaps

Internal timestamps are nanoseconds; the table shows milliseconds; rates use seconds.

**TTFT** — send until the first non-empty chunk:

```
ttft_ns = request.content_responses[0].perf_ns - request.start_perf_ns
```

Includes network, queue, prefill, and generating that first chunk. Longer prompts usually mean larger TTFT.

**TTST** — first chunk to second chunk. Splits startup tax from steady decode. Needs two non-empty chunks.

**TTFO** — time to the first **non-reasoning** output token. For models that think before they speak, TTFT may already be reasoning; TTFO is when the user sees body text. Without reasoning, TTFO = TTFT.

**ITL** — steady token gap, **excluding TTFT**:

```
ITL = (request_latency − TTFT) / (output_sequence_length − 1)
```

Needs two non-empty chunks and valid TTFT, request latency, and OSL.

**ICL** — the full distribution of gaps between consecutive **chunks**, not an average per token. A chunk may hold several tokens. Jitter, batching, and the network show up here more honestly than in a single mean ITL.

## Throughput: system vs this user

**Output Token Throughput Per User** = `1 / ITL` (seconds). Excludes TTFT. As concurrency rises this number usually falls — the system gets faster while each person gets slower.

**Prefill Throughput Per User** = `ISL / TTFT_seconds`. Prompt-reading speed only.

**E2E Output Token Throughput** (per request) = `OSL / request_latency`. Denominator includes TTFT and queueing, so it sits below per-user; works without streaming.

**Output Token Throughput** (system, Derived) = `total_osl / benchmark_duration`. Wall-clock for the whole run, **including TTFT**. Do not rank it against per-user.

**Total Token Throughput** = `(total_isl + total_osl) / benchmark_duration`.

The NIM guide’s system TPS uses `Ty − Tx` (first send to last response). The AIPerf reference uses `benchmark_duration`. Align the denominator before comparing. Warmup is excluded from scored traffic.

## How tokens are counted

Client tokenizer with `add_special_tokens=False`.

- **ISL**: prompt tokens
- **output_token_count**: user-visible output, excluding a separate `reasoning_content` field; reasoning inside ordinary `content` (for example think tags) is counted unless you filter
- **OSL** = output_token_count + reasoning_token_count. Without reasoning, OSL equals output tokens

If the server returns `usage`, AIPerf can emit Usage* metrics and Diff% — if client and server token counts disagree, fix the counting before arguing about TPS.

## End-to-end and counts

**Request latency** = last chunk `perf_ns − start_perf_ns`. Streaming counts through the final chunk.

**Request count** = successful valid records.  
**Error request count** = failures (network / HTTP / timeout / other).  
**Request throughput (RPS)** = `request_count / benchmark_duration`. Requests are fat and thin; RPS alone rarely explains anything.

## Goodput

Set SLOs with `--goodput`, e.g. `"time_to_first_token:370 request_latency:648"`. A request counts as good only if it meets **all** thresholds.

```
goodput = good_request_count / benchmark_duration
good_request_fraction = good_request_count / (request_count + error_request_count)
```

Errors enter the denominator. A backend that drops traffic under load cannot look compliant just because survivors were fast. Goodput is always ≤ request throughput. The fraction may be hidden from the console (`NO_CONSOLE`) but is in JSON/CSV.

Throughput is how many people you dismiss per second. Goodput is how many of them would come back. Plan capacity on the latter.

## Other families (not copied line-by-line)

The same official page also covers image count / throughput / latency; video inference time and peak memory; audio duration and RTFx; reasoning token counts; the full usage suite (cache read/write/miss, tools, accepted/rejected predictions); OSL mismatch; ISL of failed requests. Filter by tags: `STREAMING_ONLY`, `GOODPUT`, `HTTP_TRACE`, `USAGE_DIFF_ONLY`, and so on.

HTTP timing (`--show-trace-timing` / `--export-http-trace`) splits sending / waiting (time to first **body** byte) / receiving. That is the wire, not a synonym for TTFT. Tutorial: https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/http-trace-metrics-guide
