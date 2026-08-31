---
source: https://vllm.ai/blog/2025-12-13-vllm-router-release
lang: en
fetched: 2026-08-31
---

# vLLM Router

2025-12-13. GitHub: vllm-router. Rust load balancer between clients and a vLLM fleet. Forked from SGLang model gateway, then simplified; they floated merging into vLLM and realigning large-scale gateway features with SGLang later. K8s or bare metal. Study note; figures on the original page.

Vanilla LBs treat the LLM as stateless HTTP. KV is stateful. Prefill/Decode disaggregation is not “every pod looks the same.” This gateway is the Helm-chart intuition from production-stack, extracted.

**Policies.** Stick a session/user to the worker that still holds their KV, or prefix cache is off.

- Consistent hashing on a routing key — the performance default.
- Power of Two — cheap randomness, still decent spread.
- Round-robin / random — stateless fallback; the benchmark villain, and the honest choice when there is no shared prefix.

Local stickiness is layer one. When the next turn lands elsewhere, [mooncake.md](mooncake.md) is layer two. Cache-aware routing co-designed with the pool was still future work in that post.

**Text P/D.** New request → prefill group, then hand state to decode. Discovery: **NIXL** and **NCCL+ZMQ**. This is not the encoder split in [epd.md](epd.md). DistServe (2024) named the text path; [large-scale.md](large-scale.md) explains why Wide-EP needs it (one fat prefill can stall the whole EP combine).

**Ops.** K8s label discovery; retries with jitter; circuit breaker; eject on failed health; Prometheus `/metrics`. The control plane (AIBrix / production-stack / llm-d) can change; the gateway cares about worker state.

**Bench snapshot (demo).** Excluded vLLM’s own DP/EP coordinator (then ~1/8 the others’ throughput — known bug). Llama 3.1 8B, 8P+8D: ~**+25%** req/s vs llm-d, ~**2×** vs K8s RR; TTFT ~1200 ms better than llm-d. DeepSeek V3 1P+1D TP8: ~2× vs K8s RR; TTFT ~2000 ms better than both. K8s RR does not understand P/D — a fair villain, not a 2026 llm-d score.
