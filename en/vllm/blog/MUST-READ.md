---
source: https://vllm.ai/llms.txt
lang: en
fetched: 2026-08-31
---

# vLLM blogs to read first

Full machine list: [`CATALOG.md`](CATALOG.md). Hub: https://vllm.ai/blog · RSS: https://vllm.ai/blog/rss.xml

Knob map: [FLAG-MAP.md](FLAG-MAP.md) (`optimization.md` flags → these posts).

**Best order:** launch → Anatomy → V1 → MRV2 → (optional DeepSpeed notes) → v0.6 CPU → spec decode → FP8 KV → production CI → distributed sharding → production-stack → AIBrix → Router → encoder disagg → wide-EP → Mooncake → Elastic EP.

“Not yet supported” in V1 / spec-decode is historical. The EPD post is **encoder** disaggregation; text P/D lives in Router + large-scale. Mooncake is the cross-instance KV pool (agent prefixes). Elastic EP resizes DP at runtime.

## Architecture

| Post | Local | URL |
|---|---|---|
| PagedAttention launch | [paged-attention.md](architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm |
| Anatomy of vLLM | [anatomy.md](architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |
| V1 alpha | [v1-alpha.md](architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| Model Runner V2 | [mrv2.md](architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 |
| vLLM vs DeepSpeed | [vs-deepspeed.md](architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed |

## Performance

| Post | Local | URL |
|---|---|---|
| v0.6 throughput | [v0.6-throughput.md](performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update |
| Speculative decoding | [spec-decode.md](performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode |
| FP8 KV cache | [fp8-kvcache.md](performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| Production-quality CI | [production-quality.md](performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality |

## Serving

| Post | Local | URL |
|---|---|---|
| Distributed inference | [distributed-inference.md](serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference |
| Production stack | [production-stack.md](serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release |
| AIBrix | [aibrix.md](serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release |
| Router | [router.md](serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| Encoder disagg (EPD) | [epd.md](serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd |
| Large-scale serving | [large-scale.md](serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving |
| Mooncake Store | [mooncake.md](serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store |
| Elastic EP | [elastic-ep.md](serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism |

## Second wave (mechanics, after the main line)

Not day-0 model posts. Order: torch.compile → Sleep Mode → structured decoding → DCP → KV offload → single-node P/D (MORI-IO) → Hybrid SSM → AFD.

| Post | Local | URL |
|---|---|---|
| torch.compile | [torch-compile.md](architecture/torch-compile.md) | https://vllm.ai/blog/2025-08-20-torch-compile |
| Sleep Mode | [sleep-mode.md](architecture/sleep-mode.md) | https://vllm.ai/blog/2025-10-26-sleep-mode |
| Structured decoding | [struct-decode.md](performance/struct-decode.md) | https://vllm.ai/blog/2025-01-14-struct-decode-intro |
| Decode Context Parallelism | [dcp.md](performance/dcp.md) | https://vllm.ai/blog/2026-08-07-decode-context-parallelism |
| KV offloading connector | [kv-offload.md](serving/kv-offload.md) | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| Single-node P/D (MORI-IO) | [moriio.md](serving/moriio.md) | https://vllm.ai/blog/2026-04-07-moriio-kv-connector |
| Hybrid SSM disagg | [hybrid-ssm.md](serving/hybrid-ssm.md) | https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg |
| AFD Plugin | [afd.md](serving/afd.md) | https://vllm.ai/blog/2026-07-23-vllm-afd-plugin |
