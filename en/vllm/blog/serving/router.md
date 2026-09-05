---
source: https://vllm.ai/blog/2025-12-13-vllm-router-release
lang: en
fetched: 2026-09-04
---

# vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer

Chinese: [zh/vllm/blog/serving/router.md](../../../../zh/vllm/blog/serving/router.md)

2025-12-13. Repo: [vllm-project/router](https://github.com/vllm-project/router). Rust load balancer between clients and a vLLM fleet (Kubernetes or bare metal). Forked from [SGLang model gateway](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway), then simplified for vLLM. They floated merging into the vLLM main repo; large-scale gateway features might later realign with SGLang.

Vanilla load balancers treat LLM inference as stateless HTTP. KV is stateful (next turn wants the worker that still holds the cache). Prefill/Decode disaggregation is not “every pod looks the same.” This gateway is the Helm-chart intuition from [production-stack](production-stack.md), extracted.

Local figures (copyright remains with the original site; study copies):

![llama benchmark](../../../../assets/vllm/blog/serving/router/01-llama-benchmark.png)

![deepseek benchmark](../../../../assets/vllm/blog/serving/router/02-deepseek-benchmark.png)

## Two jobs

Intelligent load balancing, and orchestration for **text** P/D disaggregation.

### Load-balancing policies

Stick a conversational session to the worker that still holds its KV, or prefix cache is off.

- **Consistent hashing** on a routing key (session ID / user ID) — sticky, the performance default.
- **Power of Two (PoT)** — cheap random choice, still decent spread.
- **Round-robin and random** — stateless fallback; the benchmark villain, and the honest choice when there is no shared prefix.

Local stickiness is layer one. When the next turn lands elsewhere, [mooncake.md](mooncake.md) is layer two. Cache-aware routing co-designed with the pool was still future work in that post.

### Native Prefill/Decode disaggregation

Compute-bound Prefill and memory-bound Decode run as **separate worker groups**. The router:

1. Sends new requests to the Prefill group.
2. After Prefill, directs request state to a Decode worker for generation.
3. Discovers / routes for **NIXL** and **NCCL + ZMQ discovery** backends.

This is **not** the vision-encoder split in [epd.md](epd.md). DistServe (Hao AI Lab, 2024) named the text path; [large-scale.md](large-scale.md) explains why Wide-EP needs it (one fat Prefill can stall the whole EP combine).

## Resiliency and observability

- **Kubernetes service discovery** via label selectors; also works on bare metal.
- **Retries** (exponential backoff + jitter) and **circuit breakers**. Failed health checks eject the worker immediately so one dead replica does not cascade.
- Prometheus **`/metrics`**: volume, latency, errors, per-worker health.

The control plane (AIBrix / production-stack / llm-d) can change; the gateway cares about worker state.

## Benchmark snapshot (demo)

Comparators:

- **[llm-d](https://github.com/llm-d/llm-d)** — K8s-native, default **queue-aware** balancing.
- **vLLM-native / K8s Service** — round-robin, **not** P/D-aware (all pods look identical).

**Excluded:** vLLM's built-in DP/EP coordinator (the docs' [external load-balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html#external-load-balancing) path). Then ~**1/8** the others' throughput — known issue [#24461](https://github.com/vllm-project/vllm/issues/24461).

**Llama 3.1 8B**, 8 Prefill pods + 8 Decode pods:

- Router req/s ~**25%** above llm-d, ~**100%** above K8s-native.
- TTFT close to K8s-native, ~**1200 ms** faster than llm-d.

**DeepSeek V3**, 1 Prefill TP8 + 1 Decode TP8:

- Router req/s close to llm-d, ~**100%** above K8s-native.
- TTFT ~**2000 ms** faster than both.

K8s RR does not understand P/D — a fair villain for this topology, not a 2026 llm-d scorecard.

## Acknowledgements

Phi and AWS (clusters). Naman Lalit (perf / correctness benches). SGLang Model Gateway team (API + service framework fork). Tyler Michael Smith and Robert Shaw (llm-d expertise that unblocked the benches).
