---
source: https://vllm.ai/blog/2024-07-25-lfai-perf
lang: en
fetched: 2026-09-04
---

# vLLM’s Open Governance and Performance Roadmap

Chinese: [zh/vllm/blog/architecture/lfai-roadmap.md](../../../../zh/vllm/blog/architecture/lfai-roadmap.md)

2024-07-25. **vLLM Team**. Historical: V1, Wide-EP write-ups, Mooncake-as-post, and Elastic EP did not exist yet. Do not let this page’s “in progress / not yet” override 2025–2026 feature notes. Chronology: [paged-attention.md](paged-attention.md), [v0.6-throughput.md](../performance/v0.6-throughput.md). Words that later became posts: async scheduling, API-frontend isolation, FA3, Flux, [torch-compile.md](torch-compile.md), disagg prefill ([mooncake.md](../serving/mooncake.md) / [moriio.md](../serving/moriio.md) / [large-scale.md](../serving/large-scale.md)).

## Future of vLLM is open

Local figures (copyright remains with the original site; study copies):

![vllm lfai light](../../../../assets/vllm/blog/architecture/lfai-roadmap/01-vllm-lfai-light.png)

The post treats vLLM as becoming the default LLM serving engine. In Meta’s [Llama 3.1 announcement](https://ai.meta.com/blog/meta-llama-3-1/), **8 of 10** official real-time-inference partners ran vLLM for Llama 3.1. Anecdotes: it also showed up in everyday AI features.

Credit goes to the open-source community. Maintainers named then: UC Berkeley, Anyscale, AWS, CentML, Databricks, IBM, Neural Magic, Roblox, Snowflake, and others. Ownership and governance should be equally open.

Announcement: vLLM had [started LF AI & Data Foundation incubation](https://lfaidata.foundation/blog/2024/07/17/lf-ai-data-foundation-mid-year-review-significant-growth-in-the-first-half-of-2024/?hss_channel=tw-976478457881247745). No single party gets exclusive control. License and trademark stay irrevocably open. The claim is: the project stays, and it will keep being maintained.

## Performance is top priority

Six objectives: wide model coverage, broad hardware, top performance, production-ready, thriving community, extensible architecture. Performance progress then:

**Public benchmarks**

- Per-commit tracker: [perf.vllm.ai](https://perf.vllm.ai) — enhancements and regressions.
- Reproducible comparison ([docs](https://docs.vllm.ai/en/latest/performance_benchmark/benchmarks.html)): vLLM vs LMDeploy, TGI, TensorRT-LLM, to find gaps and close them.

**Kernels**

- FlashAttention2 wired to PagedAttention, plus [FlashInfer](https://github.com/flashinfer-ai/flashinfer). Plan: [FlashAttention3](https://github.com/vllm-project/vllm/issues/6348) (still an issue then).
- Integrating [Flux](https://arxiv.org/abs/2406.06858v1) to overlap compute and collectives.
- Quant kernels: INT8 / FP8 activation (cutlass); GPTQ / AWQ INT4, INT8, FP8 weight-only (marlin).

**Critical-path tax**

- Synchronous, blocking scheduler is a bottleneck on fast GPUs (H100s). Work: async scheduling, plan steps ahead.
- OpenAI-compatible API frontend overhead higher than wanted. [Isolate it from scheduler / model inference](https://github.com/vllm-project/vllm/issues/6797).
- Input prep and output processing scale suboptimally with data size; vectorize or move off the hot path.

Overall tracker then: [issue #6801](https://github.com/vllm-project/vllm/issues/6801).

## More resources

RFCs in flight:

- [SPMD Worker Control Plane](https://github.com/vllm-project/vllm/issues/6556) — simpler, faster TP.
- [Graph optimization via torch.compile](https://github.com/vllm-project/vllm/issues/6378). Later note: [torch-compile.md](torch-compile.md).
- [Disaggregated prefilling via KV cache transfer](https://github.com/vllm-project/vllm/issues/5557) — long inputs, lower ITL variance. The later P/D family grows from here.

Research they named and wanted to collaborate with (not exhaustive):

- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [Mooncake](https://arxiv.org/abs/2407.00079) (later note: [mooncake.md](../serving/mooncake.md))
- [Llumnix](https://arxiv.org/abs/2406.03243)
- [CacheGen](https://arxiv.org/abs/2310.07240)
- [vAttention](https://arxiv.org/abs/2405.04437)
- [Andes](https://arxiv.org/abs/2404.16283)
- [SGLang](https://arxiv.org/abs/2312.07104)

Read this as a **July 2024 time capsule**: incubation plus the performance debt they thought they still owed. Today’s kernels, scheduler, and P/D shape live in the 2025–2026 notes.
