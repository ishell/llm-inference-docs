---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 总览 — NVIDIA NIM LLM 压测指南

NVIDIA NIM LLMs Benchmarking 2.0.0

## 摘要

这本指南是给要把 NIM 当真用的人写的：应用开发者、系统负责人，以及任何一个曾经对着「官网上那张性能表」发呆的人。它讲指标、讲参数，并带你用 AIPerf 走完一轮实测。

读完你应能回答：

- LLM 推理的延迟和吞吐里，哪些数字真正值得失眠？
- Locust、K6、AIPerf 看起来都在报「延迟」，它们量的是不是同一件外套？
- 怎样用 NVIDIA AIPerf 给一个 OpenAI 兼容服务量一次体温？

章节：

- **指标** — TTFT、ITL、TPS、RPS
- **参数与实践** — 并发、序列长度、怎样测才不是自欺
- **用 AIPerf 实测** — 搭配 NVIDIA NIM 的端到端流程
- **LoRA** — 多 adapter 时，流量会像一副洗过的牌

## LLM 推理基准测试是做什么的

企业里 LLM 铺开之后，需要一种可复现的方法来比较 serving 方案。部署成本取决于：系统在保持响应的前提下，单位时间能处理多少请求。本指南只谈**性能测量**。精度评估不在范围内——准不准是另一场审判。

你可以用通用压测工具（Locust、K6），也可以用面向 LLM 的客户端（NVIDIA AIPerf）。这些工具暴露的指标看起来重叠，像几本封面相似的字典；翻开以后，同一个词的释义经常不一样。本指南把差异讲清楚，并走一遍 NVIDIA 推荐的生成式基准工具 AIPerf。

**性能测试**和**负载测试（压测）**回答的问题不同：

- **负载测试 / Load testing**（例如 K6）：模拟并发流量，看扩缩容、弹性伸缩、网络行为和资源上限。门会不会被挤垮。
- **性能基准 / Performance benchmarking**（例如 AIPerf）：在受控条件下测模型级吞吐、延迟和 token 级行为。刀快不快。

本指南聚焦性能基准——模型效率、优化和配置。两端都做，才能看清端到端部署。只做刀法、不做门厅，你会在第一个真实星期一死去。

> **说明：** NVIDIA NIM 也有服务端指标，本文不覆盖。见 Logging and Observability。客户端秒表和服务端仪表盘，请不要假装它们是同一只手表。

## LLM 推理怎么工作

解读指标之前，先看流水线。典型请求经过：

1. **提交 Prompt** — 用户给出查询。
2. **排队 Queuing** — 等待空闲推理槽位。这里消耗的时间，模型完全不知情。
3. **Prefill** — 模型处理完整输入 prompt，把过去写进 KV。
4. **Decode（生成）** — 一次吐出一个 token。未来在这里一寸寸发生。

Token 是 LLM 处理文本的基本单位。每个模型有自己的 tokenizer，像每人有自己的口音。粗略估计：很多主流模型上，1 token ≈ 0.75 个英文词。

序列长度决定显存和延迟：

- **ISL（Input Sequence Length）** — prompt 里的 token 数，包含 system 指令、聊天历史、思维链、RAG 上下文。你以为用户只问了一句；模型看见的是一整座阁楼。
- **OSL（Output Sequence Length）** — 模型生成的 token 数。
- **Context length** — 每一步生成时模型能看见的总 token 数（已输入 + 已生成），受最大上下文窗口限制。

更深入的背景见 [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)。

**Streaming** 会在生成过程中把部分输出推给用户，聊天场景体感更快。非流式则等全部生成完再一次性返回。一种是说书，一种是交卷。
