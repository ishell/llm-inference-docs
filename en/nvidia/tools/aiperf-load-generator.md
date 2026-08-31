---
source: https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference
lang: en
fetched: 2026-08-31
---

# AIPerf Load Generator Options

How AIPerf schedules requests (priority if several flags are set):

1. `--fixed-schedule` / mooncake_trace → replay timestamps
2. `--user-centric-rate` → per-user turn gap (needs `--num-users`)
3. `--request-rate` → target QPS (`constant` / `poisson` / `gamma`)
4. `--concurrency` only → burst / saturation (as fast as possible within N)

| Mode | Use |
|---|---|
| `--concurrency` alone | Max throughput / saturation |
| `--request-rate` | Controlled QPS load test |
| `--fixed-schedule` | Trace replay |
| `--user-centric-rate` + `--num-users` | Multi-turn KV-cache benchmarking (`turn_gap = num_users / user_centric_rate`) |

`--concurrency` with a rate is a **ceiling**, not the load driver. Unset concurrency = unlimited in-flight sessions.

Stop with `--request-count`, `--num-sessions`, or `--benchmark-duration` (+ optional grace period).

Warmup is independent (`--warmup-request-count` / `--warmup-duration`). Prefill cap `--prefill-concurrency` requires `--streaming`.

Examples:

```bash
# saturation
aiperf profile --url localhost:8000 --model llama --concurrency 10 --request-count 100

# 10 QPS poisson
aiperf profile --url localhost:8000 --model llama --request-rate 10 --arrival-pattern poisson --request-count 100

# KV / multi-turn
aiperf profile --url localhost:8000 --model llama \
  --user-centric-rate 1.0 --num-users 15 --session-turns-mean 20 --streaming --benchmark-duration 300
```

Repo: https://github.com/ai-dynamo/aiperf
