---
source: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
lang: zh
fetched: 2026-08-30
---

# LLM 推理基准测试：基本概念

系列第 1 篇。原文：https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/  
英文对照：`en/nvidia/developer-blog/01-llm-benchmarking-fundamental-concepts.md`

第 2 篇：[用 GenAI-Perf 和 NIM](https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/)

企业铺 LLM 应用时，需要比较不同 serving 方案的成本效率。成本取决于：在用户可接受的响应速度和精度下，每秒能处理多少请求。本文只谈吞吐和延迟。

NVIDIA 推理栈包括 Dynamo、TensorRT-LLM、NIM。基准工具曾主推 **GenAI-Perf**（现已逐步被 **AIPerf** 替代，概念完全通用）。

不同客户端工具对同一指标的定义、测量、计算公式常不一致，数字不能直接横比。本文把常用指标和关键测试参数说清楚。

## 压测 vs 性能基准

- **Load testing（压测）**：模拟大量并发，看真实流量下的容量、弹性伸缩、网络延迟、资源占用。
- **Performance benchmarking（性能基准）**：在受控条件下测模型本身的吞吐、延迟、token 级指标（效率、优化、配置问题）。

两者都要做。

## 推理怎么走

1. Prompt：用户提问  
2. Queuing：排队  
3. Prefill：模型处理 prompt，建 KV cache  
4. Generation / Decode：逐 token 生成  

Token 是 LLM 的最小处理单位。很多主流模型大约 1 token ≈ 0.75 个英文词。

- **ISL**：模型吃进去的 token（用户问题、system、历史、CoT、RAG 文档）
- **OSL**：模型吐出来的 token
- **Context length**：每一步生成时能看见的总 token（已输入 + 已生成），受最大窗口限制

Streaming：边生成边把 token 块推给用户，聊天体感快。非流式：整段生成完再返回。

更深背景：[Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

## 指标

### TTFT

处理 prompt 并生成第一个 token 的时间，即用户要等多久才看到输出。GenAI-Perf 和 LLMPerf 都会丢掉空内容的初始响应。

TTFT ≈ 排队 + prefill + 网络。Prompt 越长越大，因为 attention 要用完整输入建 KV cache。多请求并行时，一个请求的 prefill 可以和另一个的 generation 重叠。

### e2e latency

从发出请求到收到最后一个 token。流式模式下 detokenize 可能多次。

```
e2e_latency = TTFT + generation_time
```

`generation_time` 是第一个 token 到最后一个 token。GenAI-Perf 会去掉最后的 done/空响应。

### ITL / TPOT

连续 token 之间的平均时间。**GenAI-Perf 不含 TTFT；LLMPerf 含 TTFT。**

GenAI-Perf：

```
ITL = (e2e_latency - TTFT) / (output_tokens - 1)
```

只刻画 decode。输出变长时 KV 变大，attention 对已有长度近似线性，但通常不是 compute-bound。ITL 稳定说明显存管理和带宽健康。

### TPS

**系统 TPS**：所有并发请求合计的输出 token/秒。并发升高时上升，直到 GPU 饱和，再往上可能下降。

- GenAI-Perf：总输出 token /（第一个请求发出 → 最后一个请求的最后响应）
- LLMPerf：总输出 token / 整个测试墙钟。会把造 prompt、准备请求、存响应算进去。单并发时这些开销有时能占到 **33%**。

GenAI-Perf 用滑动窗口取稳态，warmup / cooldown 不计入。

**单用户 TPS** = OSL / e2e_latency，输出足够长时趋近 `1/ITL`。系统并发升高时，系统 TPS 升、单用户 TPS 降。

### RPS

平均每秒成功完成的请求数。

## 参数与实践

### 业务决定 ISL/OSL

- ISL 长 → prefill 更吃显存 → TTFT 大
- OSL 长 → 生成更吃带宽/容量 → ITL 大

常见量级：

| 场景 | ISL | OSL |
|---|---|---|
| 翻译（语言/代码） | ~500–2000 | ~500–2000 |
| 生成（代码/故事/邮件） | ~100 | ~1000 |
| 摘要 / RAG / 多轮 | ~1000 | ~100 |
| 推理模型（显式 CoT） | ~100 | ~1000–10000 |

### 负载控制

**Concurrency N**：始终保持 N 个在途请求。请求完成后立刻补发。这是控负载最常用的方式。

注意：LLMPerf 按批发 N 个，然后等整批结束再发下一批，批末并发会掉到 0。**GenAI-Perf / AIPerf 全程维持 N 个活跃请求。** 两种工具的并发含义不一样。

**Max batch size**：引擎同时真正在算的请求数，可以小于并发。`concurrency > max_batch × 副本` 时排队，TTFT 会涨。

**Request rate**：按到达率发。到达超过吞吐时在途请求会无限堆积。GenAI-Perf 两种都支持，**官方建议用 concurrency**。

扫描并发：从 1 扫到略大于 max batch。超过 max batch 后吞吐饱和、延迟继续涨。

### 其他

- 基准测试设 **`ignore_eos=True`**，生成到 `max_tokens`，OSL 才可控。
- 采样（greedy / top_p / top_k / temperature）会影响速度。greedy 不用归一化排序。同一套测试采样必须固定。

## 小结

先对齐指标定义，再用 concurrency 扫出延迟–吞吐曲线，才谈得上成本和 SLA。后续见系列第 2–4 篇（NIM 实测、TensorRT-LLM 调优、TCO）。
