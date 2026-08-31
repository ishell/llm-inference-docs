---
source: https://vllm.ai/blog/2025-01-21-stack-release
lang: en
fetched: 2026-08-31
---

# vLLM production-stack

2025-01-21. https://github.com/vllm-project/production-stack (LMCache + vLLM). Cluster reference on top of the single-node engine: KV sharing (LMCache), prefix-aware routing, observability (TTFT/TBT/throughput), autoscaling / fault tolerance. Claimed 3–10× lower delay and 2–5× throughput vs alternatives in their multi-round QA bench (figures on the page).

```bash
helm repo add llmstack-repo https://lmcache.github.io/helm/
helm install llmstack llmstack-repo/vllm-stack
```

See the AIBrix FAQ for how this differs from ByteDance’s control plane.
