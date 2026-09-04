---
source: https://vllm.ai/blog/2024-07-25-lfai-perf
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# LF AI 孵化与 2024 年中的路线图

英文对照：[en/vllm/blog/architecture/lfai-roadmap.md](../../../../en/vllm/blog/architecture/lfai-roadmap.md)  
原文：https://vllm.ai/blog/2024-07-25-lfai-perf  
2024-07-25。署名 **vLLM Team**。历史文献：当时 V1、Wide-EP、Mooncake 成文、Elastic EP 都还没写出来。不要用这篇的「正在做 / 尚未」覆盖 2025–2026 的 feature 页。编年一起读：[立项](paged-attention.md)、[v0.6](../performance/v0.6-throughput.md)。后来单独成篇的词：异步调度、API frontend 隔离、FA3、Flux、[torch.compile](torch-compile.md)、disagg prefill（[Mooncake](../serving/mooncake.md) / [MORI-IO](../serving/moriio.md) / [large-scale](../serving/large-scale.md)）。

## 未来要开着

本地图（原文版权仍归原站；学习对照用）：

![vllm lfai light](../../../../assets/vllm/blog/architecture/lfai-roadmap/01-vllm-lfai-light.png)

他们写 vLLM 正在变成 LLM 推理的默认件。Meta [Llama 3.1 发布](https://ai.meta.com/blog/meta-llama-3-1/) 里，实时推理的官方伙伴大约 **10 家里有 8 家** 用 vLLM 侍候 Llama 3.1。日常 AI 功能里也有不少口头反馈在用。

成功被归到开源社区。当时点名的维护方：UC Berkeley、Anyscale、AWS、CentML、Databricks、IBM、Neural Magic、Roblox、Snowflake 等。所有权和治理也要公开、可核验。

宣布：vLLM **已进入 [LF AI & Data Foundation 孵化](https://lfaidata.foundation/blog/2024/07/17/lf-ai-data-foundation-mid-year-review-significant-growth-in-the-first-half-of-2024/?hss_channel=tw-976478457881247745)**。没有单方独占未来；许可证和商标不可撤回地开着。原文的承诺是：项目会留下来，也会被继续维护。

## 性能是第一优先

六个目标：宽模型覆盖、宽硬件、顶尖性能、生产可用、活的开源社区、可扩展架构。性能这一条当时的进度：

**公开基准**

- 每 commit 的性能追踪：[perf.vllm.ai](https://perf.vllm.ai)，用来看提升和回退。
- 可复现对照（[文档](https://docs.vllm.ai/en/latest/performance_benchmark/benchmarks.html)）：vLLM vs LMDeploy、TGI、TensorRT-LLM。目的是找差距再补上。

**核**

- FlashAttention2 接到 PagedAttention，以及 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。计划接 [FlashAttention3](https://github.com/vllm-project/vllm/issues/6348)（当时还是 issue）。
- 正在接 [Flux](https://arxiv.org/abs/2406.06858v1)：计算和集合通信重叠。
- 量化核：INT8 / FP8 activation（cutlass）；GPTQ / AWQ 的 INT4、INT8、FP8 weight-only（marlin）。

**关键路径上的税**

- 同步、阻塞的 scheduler 在快卡（H100）上是瓶颈。计划改成异步、提前把 step 排好。
- OpenAI 兼容 API frontend 开销偏高。[打算从 scheduler / 模型推理的热路径上拆出去](https://github.com/vllm-project/vllm/issues/6797)。
- 输入准备和输出处理随数据量次线性变差；许多操作可以向量化，或挪出热路径。

总进度当时挂在 [issue #6801](https://github.com/vllm-project/vllm/issues/6801)。

## 更多资源

当时在写的 RFC：

- [SPMD Worker Control Plane](https://github.com/vllm-project/vllm/issues/6556)：降低复杂度，抬 TP。
- [用 torch.compile 做图优化](https://github.com/vllm-project/vllm/issues/6378)：PyTorch 原生编译、核融合。落地后见 [torch-compile](torch-compile.md)。
- [disaggregated prefilling via KV cache transfer](https://github.com/vllm-project/vllm/issues/5557)：长输入、压 ITL 方差。后来的 P/D 族从这里长出来。

原文点名、愿意合作的研究（非完整名单）：

- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)（吞吐–延迟）
- [Mooncake](https://arxiv.org/abs/2407.00079)（KV-centric 分离；后来的笔记：[mooncake](../serving/mooncake.md)）
- [Llumnix](https://arxiv.org/abs/2406.03243)（动态调度）
- [CacheGen](https://arxiv.org/abs/2310.07240)（KV 压缩与流式）
- [vAttention](https://arxiv.org/abs/2405.04437)（不用 PagedAttention 的动态显存）
- [Andes](https://arxiv.org/abs/2404.16283)（文本流式的 QoE）
- [SGLang](https://arxiv.org/abs/2312.07104)（结构化语言模型程序）

读这篇只当 **2024-07 的时间胶囊**：孵化公告 + 当时认为该补的性能债。核、调度、P/D 的今日形状以 2025–2026 各篇为准。
