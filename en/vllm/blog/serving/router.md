---
source: https://vllm.ai/blog/2025-12-13-vllm-router-release
lang: en
fetched: 2026-08-31
---

# vLLM Router

2025-12-13. Rust load balancer between clients and a vLLM fleet. Forked from SGLang model gateway. Stateful: KV affinity + prefill/decode disaggregation. K8s or bare metal.

**Policies:** consistent hashing (sticky session/user → KV reuse); power-of-two; round-robin/random.

**P/D:** new requests → prefill group, then decode group. NIXL and NCCL+ZMQ discovery.

**Ops:** K8s label discovery; retries + circuit breaker; Prometheus `/metrics`.

**Bench snapshot:** Llama 3.1 8B 8P+8D: ~**+25%** req/s vs llm-d, ~**2×** vs K8s RR; TTFT ~1200 ms better than llm-d. DeepSeek V3 1P+1D TP8: ~2× vs K8s RR; TTFT ~2000 ms better than both. Excluded vLLM DP/EP coordinator (then 1/8 throughput, known bug).
