---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
lang: zh
fetched: 2026-08-30
---

# 总览 — NVIDIA NIM LLM 压测指南

NVIDIA NIM LLMs Benchmarking 2.0.0

## 摘要

本指南面向 LLM 应用开发者和系统负责人，帮助你对 NVIDIA NIM 部署做推理延迟与吞吐的基准测试。内容包括关键指标、测试参数，以及用 AIPerf 做一遍完整实测。

读完后你应能回答：

- LLM 推理延迟和吞吐里，哪些指标最重要？
- 有哪些压测/基准工具，它们的测量方式有何不同？
- 怎样用 NVIDIA AIPerf 给 LLM 应用做基准测试？

章节结构：

- **指标（Metrics）** — TTFT、ITL、TPS、RPS 等定义
- **参数与实践（Parameters）** — 并发、序列长度及其他测试设置
- **用 AIPerf 实测** — 搭配 NVIDIA NIM 的端到端流程
- **LoRA 模型压测** — 多 adapter 部署怎么测

## LLM 推理基准测试是做什么的

企业里 LLM 应用铺开之后，需要可复现的方法来比较不同 serving 方案。部署成本取决于：系统在保持响应的前提下，单位时间能处理多少请求。本指南只谈**性能测量**；精度评估不在范围内，请按业务单独验证。

可以用通用压测工具（Locust、K6），也可以用面向 LLM 的客户端（NVIDIA AIPerf）。这些工具暴露的指标看起来重叠，但定义和计算公式经常不一样。本指南会把差异讲清楚，并走一遍 NVIDIA 推荐的生成式 AI 基准工具 AIPerf。

**性能测试**和**负载测试（压测）**回答的问题不同：

- **负载测试 / Load testing**（例如 K6）：模拟并发流量，看扩缩容、弹性伸缩、网络行为和资源上限。
- **性能基准 / Performance benchmarking**（例如 AIPerf）：在受控条件下测模型级吞吐、延迟和 token 级行为。

本指南聚焦性能基准——模型效率、优化和配置。两端都做，才能看清端到端部署表现。

> **说明：** NVIDIA NIM 也有服务端指标，本文不覆盖。见 Logging and Observability 文档。

## LLM 推理怎么工作

解读指标之前，先看推理流水线。典型请求会经过：

1. **提交 Prompt** — 用户给出查询。
2. **排队 Queuing** — 等待空闲推理槽位。
3. **Prefill** — 模型处理完整输入 prompt。
4. **Decode（生成）** — 一次吐出一个 token。

Token 是 LLM 处理文本的基本单位。每个模型有自己的 tokenizer。粗略估计：很多主流模型上，1 token ≈ 0.75 个英文词。

序列长度决定显存和延迟：

- **ISL（Input Sequence Length）** — prompt 里的 token 数，包含 system 指令、聊天历史、思维链、RAG 上下文。
- **OSL（Output Sequence Length）** — 模型生成的 token 数。
- **Context length** — 每一步生成时模型能看见的总 token 数（已输入 + 已生成），受模型最大上下文窗口限制。

更深入的背景见 [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)。

**Streaming** 会在生成过程中把部分输出推给用户，聊天场景体感更快。非流式则等全部生成完再一次性返回。
