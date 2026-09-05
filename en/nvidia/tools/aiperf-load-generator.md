---
source: https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference
lang: en
fetched: 2026-09-01
---

# AIPerf Load Generator

The official page is a compatibility matrix: which flags work together, which raise immediately. Scheduling is the personality of the run. Pick the wrong mode and you measure your invented arrival process, not the model.

Install and architecture: `aiperf.md`. Formulas: `aiperf-metrics.md`.

## Four ways to issue requests

If several scheduling flags are set, priority is top-down; lower ones are ignored or error:

1. `--fixed-schedule` / `mooncake_trace` → replay timestamps from the dataset
2. `--user-centric-rate` → per-user turn gap (requires `--num-users`)
3. `--request-rate` → target QPS (`constant` / `poisson` / `gamma`)
4. `--concurrency` only → burst / saturation (as fast as possible within N)

| Question | Flags |
|---|---|
| What did this trace look like in wall time? | `--fixed-schedule` + mooncake_trace |
| Is KV still alive across turns? | `--user-centric-rate` + `--num-users` |
| Latency at a controlled QPS? | `--request-rate` (optional `--arrival-pattern`) |
| How hard can we feed the GPU? | `--concurrency` alone, no rate |

With a rate, `--concurrency` is a **ceiling**: credits fire on schedule, then wait if in-flight is full. Unset concurrency = unlimited in-flight sessions. Docs: in user-centric mode set concurrency ≥ `--num-users`, or some “users” never get a seat.

## Arrival patterns

Only with `--request-rate`:

- `constant` — interval = `1/rate`
- `poisson` — exponential intervals (default with `--request-rate`)
- `gamma` — smoothness via `--arrival-smoothness` (errors on any other pattern)
- `concurrency_burst` — auto-selected when no rate is set

`--request-rate-ramp-duration` belongs to request-rate only (not fixed-schedule or user-centric). `--concurrency-ramp-duration` works with all three.

## How the run stops

At least one stop condition:

- `--request-count` and `--num-sessions` are **mutually exclusive**
- `--benchmark-grace-period` requires `--benchmark-duration` (default 30s; user-centric + duration defaults grace to ∞)

For synthetic data with no request-count, the docs auto-pick `max(10, concurrency * 2)`.

## Prefill cap

`--prefill-concurrency` requires `--streaming` and must be ≤ `--concurrency`. Too many simultaneous prefills on long context exhaust memory first. This reserves decode seats; it is not another QPS knob.

## Warmup

Warmup **always uses rate-based scheduling internally**, regardless of the main timing mode. Stop with `--warmup-request-count` / `--warmup-duration` / `--num-warmup-sessions` (the two count-style flags are exclusive). Unset warmup concurrency/rate/pattern fall back to the main flags. `--warmup-grace-period` defaults to ∞ but warmup must be enabled first.

## Other combinations that bite

| Flag | Note |
|---|---|
| `--num-users` | Only with `--user-centric-rate` |
| `--session-turns-mean` | User-centric requires ≥ 2 (single-turn → `--request-rate`) |
| `--dataset-sampling-strategy` | Incompatible with `--fixed-schedule` |
| `--fixed-schedule-auto-offset` vs `--fixed-schedule-start-offset` | Exclusive; both need `--fixed-schedule` |
| `--request-cancellation-delay` | Needs `--request-cancellation-rate` (0–100 %) |
| Repeated `--url` | `--url-strategy` defaults to `round_robin` |

User-centric formula:

```
turn_gap = num_users / user_centric_rate
```

15 users at rate 1.0 → 15 seconds between that user’s turns. This tests KV lifetime, not peak TPS. Full KV setups also want `--shared-system-prompt-length` and friends — see the User-Centric Timing tutorial.

## Examples

Saturation:

```bash
aiperf profile --url localhost:8000 --model llama \
  --concurrency 10 --request-count 100
```

10 QPS Poisson:

```bash
aiperf profile --url localhost:8000 --model llama \
  --request-rate 10 --arrival-pattern poisson --request-count 100
```

Constant arrivals + concurrency ceiling + duration:

```bash
aiperf profile --url localhost:8000 --model llama \
  --request-rate 20 --arrival-pattern constant \
  --concurrency 5 --benchmark-duration 60
```

Prefill-limited long context:

```bash
aiperf profile --url localhost:8000 --model llama \
  --concurrency 20 --prefill-concurrency 5 --streaming \
  --benchmark-duration 60
```

Mooncake replay with a millisecond window:

```bash
aiperf profile --url localhost:8000 --model llama \
  --input-file trace.jsonl --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --fixed-schedule-start-offset 60000 \
  --fixed-schedule-end-offset 120000
```

Multi-turn KV:

```bash
aiperf profile --url localhost:8000 --model llama \
  --user-centric-rate 1.0 --num-users 15 \
  --session-turns-mean 20 --streaming --benchmark-duration 300
```

## Validation errors (direction of the official messages)

- user-centric cannot combine with `--request-rate` / `--arrival-pattern`
- user-centric needs `--num-users` and `--session-turns-mean >= 2`
- grace period only with duration
- warmup grace only after warmup is enabled
- prefill-concurrency needs streaming
- arrival-smoothness only with gamma
- request-count and num-sessions cannot both be set
- `--num-users` cannot appear alone
- `--request-rate-ramp-duration` cannot pair with user-centric or fixed-schedule

Type/default tables live on the official “Full Options Reference”. This note covers the flags that change what the measurement *means*.
