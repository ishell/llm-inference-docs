---
source: https://docs.vllm.ai/en/stable/configuration/optimization/
lang: en
fetched: 2026-08-31
---

# Knob → blog map

The optimization page is the polite order of knobs. The blogs are how those knobs grew. Full reading order: [MUST-READ.md](MUST-READ.md). Docs: [optimization.md](../optimization/optimization.md), [serve.md](../getting-started/serve.md).

| Knob | Docs section | Blog first | One line |
|---|---|---|---|
| KV blocks / PagedAttention | preemption, cache size | [launch](architecture/paged-attention.md) → [Anatomy](architecture/anatomy.md) | The house is KV, not weights. |
| `-O0`…`-O3`, CUDA graphs, `torch.compile` | optimization level; `--enforce-eager` | [v0.6](performance/v0.6-throughput.md), [V1](architecture/v1-alpha.md), [MRV2](architecture/mrv2.md) | Startup vs steady-state decode. |
| `max_num_batched_tokens`, chunked prefill | Chunked Prefill | [vs DeepSpeed](architecture/vs-deepspeed.md), Anatomy | 2023 name: SplitFuse. Small value guards ITL; large guards TTFT; throughput often wants >8192. |
| prefix cache | features | Anatomy, [production-stack](serving/production-stack.md), [Router](serving/router.md), [Mooncake](serving/mooncake.md) | Local hit → sticky session; miss → distributed pool. |
| speculative decoding | features | [spec-decode](performance/spec-decode.md) | Draft/verify. “Not yet supported” is historical. |
| FP8 KV / attention quant | memory / precision | [FP8 KV](performance/fp8-kvcache.md) | Hopper/Blackwell validation, not a free lunch. |
| `gpu_memory_utilization`, preemption | Preemption | Anatomy, launch | V1 default RECOMPUTE. Frequent preempt → give KV rooms. |
| TP / PP | parallelism | [distributed](serving/distributed-inference.md); TRT-LLM sharding chapter | TP in-node, PP between; do not naive-TP MLA. |
| EP / DP / `--enable-expert-parallel` | Expert / Data Parallelism | [Wide-EP](serving/large-scale.md) | Dense → DP Attention; sparse → EP. |
| `--enable-dbo`, EPLB | (blog-first) | Wide-EP | Overlap microbatches when EP comm is fat; reshuffle hot experts. |
| `--enable-elastic-ep` | (blog) | [Elastic EP](serving/elastic-ep.md) | Resize DP at runtime. Then: TP=1, no DBO, Ray only. |
| Text P/D | deploy | [Router](serving/router.md), Wide-EP | One fat prefill can stall the EP combine. |
| `mm_encoder_tp_mode="data"`, MM caches | encoder DP; multimodal cache | [EPD](serving/epd.md) | Single-node batch-split the ViT; cluster moves it to another building. |
| KVConnector / external KV | (blog) | Mooncake, production-stack | One plugin door: LMCache, Mooncake Store, NIXL. |
| `--api-server-count`, CPU cores | API scale-out; CPU | v0.6, Anatomy | V1 is multiprocess; starved CPU looks like idle GPU. |
| Ship quality | CI | [production CI](performance/production-quality.md) | Nightly benches, many accelerators, two-week trains. |
| Routing / control plane | (not in optimization) | production-stack / [AIBrix](serving/aibrix.md) / Router | The rack on top of the engine can change; KV affinity cannot. |

NVIDIA sibling map: TensorRT-LLM sharding chapter. Official CLI once wrote `--tp_size` twice; PP is `--pp_size`.
