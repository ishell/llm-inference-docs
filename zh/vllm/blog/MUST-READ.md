---
source: https://vllm.ai/llms.txt
lang: zh
fetched: 2026-08-31
---

# 建议先读的 vLLM 博客

完整机器表：英文 [`CATALOG.md`](../../en/vllm/blog/CATALOG.md)。入口 https://vllm.ai/blog ，RSS https://vllm.ai/blog/rss.xml

旋钮对照：[FLAG-MAP.md](FLAG-MAP.md)（`optimization.md` 里的旗标 → 这篇博客）。

**阅读顺序（最佳）：** 立项 → Anatomy → V1 → MRV2 →（可插 DeepSpeed 笔记）→ v0.6 CPU → 投机解码 → FP8 KV → 生产级 CI → 分布式切卡 → production-stack → AIBrix → Router → Encoder 分离 → 大规模 Wide-EP → Mooncake → Elastic EP。

V1 / spec-decode 文中的「还不支持」是当时的缺口。EPD 那篇是 **视觉编码器分离**，文本 Prefill/Decode 分离在 Router 与大规模两篇。Mooncake 是跨实例 KV 池（agent 前缀）；Elastic EP 是运行时改 DP 宽度。

## architecture

| 文 | 本地 | URL |
|---|---|---|
| PagedAttention 立项 | [paged-attention.md](architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm |
| Anatomy of vLLM | [anatomy.md](architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |
| V1 alpha | [v1-alpha.md](architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| Model Runner V2 | [mrv2.md](architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 |
| vLLM vs DeepSpeed | [vs-deepspeed.md](architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed |

## performance

| 文 | 本地 | URL |
|---|---|---|
| v0.6 吞吐 | [v0.6-throughput.md](performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update |
| Speculative decoding | [spec-decode.md](performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode |
| FP8 KV cache | [fp8-kvcache.md](performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| 生产级 CI | [production-quality.md](performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality |

## serving

| 文 | 本地 | URL |
|---|---|---|
| 分布式推理 | [distributed-inference.md](serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference |
| Production stack | [production-stack.md](serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release |
| AIBrix | [aibrix.md](serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release |
| Router | [router.md](serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| Encoder 分离 (EPD) | [epd.md](serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd |
| 大规模 serving | [large-scale.md](serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving |
| Mooncake Store | [mooncake.md](serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store |
| Elastic EP | [elastic-ep.md](serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism |

## 第二波（机制，必读之后）

主线走完再读。不是 day-0 模型文。顺序：torch.compile → Sleep Mode → structured decoding → DCP → KV offload → 单机 P/D（MORI-IO）→ Hybrid SSM → AFD。

| 文 | 本地 | URL |
|---|---|---|
| torch.compile | [torch-compile.md](architecture/torch-compile.md) | https://vllm.ai/blog/2025-08-20-torch-compile |
| Sleep Mode | [sleep-mode.md](architecture/sleep-mode.md) | https://vllm.ai/blog/2025-10-26-sleep-mode |
| Structured decoding | [struct-decode.md](performance/struct-decode.md) | https://vllm.ai/blog/2025-01-14-struct-decode-intro |
| Decode Context Parallelism | [dcp.md](performance/dcp.md) | https://vllm.ai/blog/2026-08-07-decode-context-parallelism |
| KV offloading connector | [kv-offload.md](serving/kv-offload.md) | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| 单机 P/D（MORI-IO） | [moriio.md](serving/moriio.md) | https://vllm.ai/blog/2026-04-07-moriio-kv-connector |
| Hybrid SSM 分离 | [hybrid-ssm.md](serving/hybrid-ssm.md) | https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg |
| AFD Plugin | [afd.md](serving/afd.md) | https://vllm.ai/blog/2026-07-23-vllm-afd-plugin |
