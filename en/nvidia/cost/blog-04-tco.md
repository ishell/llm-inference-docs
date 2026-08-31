---
source: https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/
lang: en
fetched: 2026-08-30
---

# Part 4: How Much Does LLM Inference Cost?

Source: https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/

Turn latency/throughput into TCO.

1. Benchmark each serving unit (AIPerf).
2. Plot latency–throughput; take the Pareto front (best throughput at a given latency). Normalize req/s per GPU if instance sizes differ.
3. Constraints: e.g. chat TTFT ≤ 250 ms; planned peak requests/s (not concurrent users).
4. Drop points that miss the latency cap; pick highest throughput among the rest → achievable RPS per instance and GPUs per instance.
5. Min instances = planned peak RPS / RPS per instance.
6. Servers = instances × GPUs/instance / GPUs/server.
7. Yearly cost/server = purchase / depreciation years + hosting + software license.
8. Total = servers × yearly cost.

Then cost per 1000 prompts and per 1M tokens (using that use case’s ISL/OSL). Split input vs output using a price ratio (example in the post: $1 / 1M in vs $3 / 1M out).

Illustrative hardware numbers in the post are for the formula only, not a quote.
