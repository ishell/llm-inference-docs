---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: en
fetched: 2026-08-31
---

# vLLM production-stack

2025-01-21. https://github.com/vllm-project/production-stack (LMCache + vLLM). 

vLLM was already the busy single-node engine. This post is the cluster layer on top of it, not a replacement for PagedAttention. Claimed **3–10×** lower delay and **2–5×** throughput vs “bare vLLM + other racks” on their multi-round QA bench (the title also says 10× — trust the figures). Demo numbers, not your SLA.

Four patches:

- **KV sharing / off-engine storage** (LMCache) when context is reused. Their stated strength: long-context, prefill-heavy work.
- **Prefix-aware routing** to the instance that already holds that KV. Stateless round-robin smashes prefix hit rate. Later [router.md](router.md) makes this a Rust gateway; [mooncake.md](mooncake.md) adds a pool for when the local instance does not hold it.
- **Observability:** engine health plus query-level TTFT / TBT / throughput.
- **Autoscaling and faults.**

Path: app → router looks for a cached context in the pool → forward; cluster manager watches load and starts nodes; dashboards watch TTFT/TBT/throughput and KV hit rate.

```bash
helm repo add llmstack-repo https://lmcache.github.io/helm/
helm install llmstack llmstack-repo/vllm-stack
```

Comparison table vs neighbouring stacks is on the webpage. See the [AIBrix](aibrix.md) FAQ for how this differs from ByteDance’s control plane: production-stack starts from community building blocks and KV-centric tricks; it planned to reuse AIBrix parts.
