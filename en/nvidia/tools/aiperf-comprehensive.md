---
source: https://docs.nvidia.com/aiperf/getting-started/ai-perf-comprehensive-llm-benchmarking
lang: en
fetched: 2026-09-01
---

# AIPerf: five real workloads

Official comprehensive guide (labeled AIPerf v0.5.0; demo 2025-11-13, page updated 2026-02-02). The demo cluster is gone. Numbers below are **their case study**, not your hardware. Start from `aiperf.md`.

Their target: Qwen3-0.6B, vLLM v0.11.0, 8-way data parallel (8×H200, one GPU per replica). The small model exists so the stopwatch has something to write, not as a model to worship.

Follow along with one replica:

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
export ENDPOINT_URL=localhost:8000
```

## 1. Fixed ISL/OSL, then a Pareto sweep

```bash
aiperf profile \
  --model qwen3-0.6b \
  --url $ENDPOINT_URL \
  --endpoint-type chat --streaming \
  --concurrency 100 --request-count 1000 \
  --isl 1000 --osl 500 \
  --tokenizer Qwen/Qwen3-0.6B
```

One hundred virtual users, about 1000→500 tokens each. Their 8-replica table: TTFT avg 347 ms, request latency 2.1 s, ITL 3.57 ms, ~22.5K output tok/s, 45.7 RPS, 1000/1000 OK.

Sweep concurrency 10 / 50 / 100 / 200 / 500 (separate `--artifact-dir` per cell):

| Concurrency | Total TPS | TPS/GPU | TPS/User | TTFT avg |
|---|---|---|---|---|
| 10 | 3,045 | 1,522 | 365 | ~250 ms |
| 50 | 12,890 | 6,445 | 326 | ~270 ms |
| 100 | 22,521 | 11,261 | 285 | ~347 ms |
| 200 | 35,999 | 18,000 | 239 | ~420 ms |
| 500 | 29,836 | 14,918 | 129 | ~1,129 ms |

TPS/GPU = total TPS / 8. c=200 is the GPU-efficiency peak on this table; c=500 collapses both axes — queues eat throughput and experience. You cannot maximize tok/s per GPU and tok/s per user at once. Seat choice: experience at 10–50; balance at 100–200; cost at the peak, not past the collapse.

## 2. When default percentiles are the wrong SLA, read jsonl

The console prints P50 / P90 / P99. If the org wants P75, do not rerun — read `profile_export.jsonl`:

```python
import json
import numpy as np

ttft = []
with open("./artifacts/profile_export.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        ttft.append(rec["metrics"]["time_to_first_token"]["value"])
print(f"P75 TTFT: {np.percentile(ttft, 75):.2f} ms")
```

On their 1000 requests, P75 TTFT was 422.87 ms. Each line also has ISL/OSL, ITL, and `benchmark_phase` (`warmup` or `profiling`). Pydantic loaders: Working with Profile Export Files.

## 3. Mooncake traces: what uniform ISL cannot see

Mooncake published production traces from an arXiv QA service: arrival times, ISL/OSL, and **hash_ids** per 512-token block. The same document reuses the same hashes across turns, so you can talk about KV hits without leaking user text.

Their 23,608 requests over 60 minutes: median ~6,402 tokens, P99 ~61k, max >125k; about 393 requests/minute, not laboratory-smooth.

```bash
curl -o mooncake_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl

# Original timestamps: can the lobby keep up with that day
aiperf profile ... --input-file mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace --fixed-schedule --streaming

# Drop --fixed-schedule: fire as fast as possible for capacity
```

They replayed the first 5 minutes (1,765 requests) at 5× in about a minute: ISL 890–32,236, 96% success — 75 requests hit Qwen3-0.6B’s 32K window. A synthetic 1000→500 run never tells you that. Traces expose the roof; uniform ISL only exposes the lab floor.

## 4. Goodput: how much of throughput still meets the SLA

Same sped-up trace, plus:

```bash
--goodput "time_to_first_token:370 request_latency:648"
```

Their demo: 26.67 RPS vs 7.43 goodput — about 28% of requests met **both** SLOs. Mean TTFT already sat above 370 ms; median latency above 648 ms. Sizing 38 machines from raw throughput becomes ~135 from goodput. Ignoring goodput is buying GPUs against a number users are already suffering under.

Change thresholds by product tier. Formulas: `aiperf-metrics.md`.

## 5. Time slices: averages hide cold start

```bash
--slice-duration 10
```

Two extra files: `profile_export_aiperf_timeslices.csv` / `.json`. Their 10s windows: slice 0 TTFT 545 ms, slice 1 381 ms, then 344–388 ms. The run-wide mean (~386 ms) washes out a +41% tax on the first slice. Do not sign an SLA on an average that includes cold start.

Slices <5s jitter; >60s smear. Typical 10–30s. To hunt leaks: `--benchmark-duration 3600 --slice-duration 300` and watch TTFT trend up.

## Same page, shorter notes

- **In-cluster load**: put the client in the same Kubernetes cluster, use ClusterIP. At extreme concurrency the client’s ports often collapse first.
- **Cancellation**: `--request-cancellation-rate 20 --request-cancellation-delay 0.5`.
- **Prometheus**: auto-discovered from `--url`, or `--server-metrics`.
- **Plots**: `aiperf plot`; `--dashboard` on 8050.
- **Trace synthesis**: `--synthesis-speedup-ratio`, `--synthesis-prefix-len-multiplier`, … to torture KV on purpose.
- **User-centric**: `--user-centric-rate` + `--num-users` + `--shared-system-prompt-length` — see `aiperf-load-generator.md`.

The guide’s own close: use case 1 is baseline capacity; production still needs traces, goodput, and time slices. Miss one and you will fall in love with a lab curve that does not survive the lobby.
