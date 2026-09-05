---
source: https://docs.vllm.ai/en/stable/
lang: en
fetched: 2026-09-04
---

# Welcome to vLLM

Chinese: [zh/vllm/getting-started/index.md](../../../zh/vllm/getting-started/index.md)  
Hub: https://docs.vllm.ai/en/stable/  ·  rolling: https://docs.vllm.ai/en/latest/

Fast LLM serving: PagedAttention, continuous batching, chunked prefill, prefix cache, CUDA/HIP graphs, speculative decoding, disaggregated P/D, many quant formats, OpenAI + Anthropic + gRPC APIs. Study notes, not official docs.

Start here locally:

- Install / offline / `vllm serve`: [quickstart.md](quickstart.md)
- Perf-related server flags (not the generated CLI page): [serve.md](serve.md)
- Tuning order: [optimization.md](../optimization/optimization.md)
- Client ruler: [cli.md](../benchmarking/cli.md) · batch grid: [auto-tune.md](../benchmarking/auto-tune.md)
- Clock indoors: [/metrics](../metrics/production-metrics.md) · how they are computed: [design-metrics.md](../metrics/design-metrics.md)
- Features: [APC](../features/prefix-caching.md) · [ledger](../features/prefix-caching-design.md) · [spec decode](../features/speculative-decoding.md) · [V1](../features/v1-guide.md)
