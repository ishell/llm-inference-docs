---
source: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
lang: zh
voice: book-zh
fetched: 2026-09-06
---

# LLM 推理基准测试：基本概念

系列第 1 篇。英文对照：[en/nvidia/benchmarking/blog-01-fundamental-concepts.md](../../../en/nvidia/benchmarking/blog-01-fundamental-concepts.md)

LLM 推理的大部分工作发生在等待里：请求排队，把整段 prompt 读进 KV cache，然后一个 token 一个 token 地生成。用户看到的，是从提交到第一个非空 token、再到整段输出结束的这段时间。

企业部署 LLM 应用时，真正贵的往往不是「模型在论文里有多强」，而是：在延迟和精度都还能接受的前提下，每秒能完成多少请求。本文只谈吞吐和延迟。精度应单独评估，不要和秒表混在一起。

NVIDIA 的推理栈里有 Dynamo、TensorRT-LLM、NIM。他们一度主推 **GenAI-Perf**，现在请改用 **AIPerf**——换了名字，指标的定义还在。

不同客户端对同一指标的定义、测量、除法常常对不齐。数字不能直接横比。拿两张表对照之前，先确认：两边说的 TTFT，是不是同一种等待。

## 压测 vs 性能基准

- **Load testing（压测）**：模拟大量并发，看真实流量下的容量、弹性伸缩、网络延迟、资源占用。问的是系统。
- **Performance benchmarking（性能基准）**：在受控条件下测模型本身的吞吐、延迟、token 级指标。问的是模型在给定负载下有多快、配置有没有走偏。

两者都要做。只压系统，我们看不出模型本身够不够快；只测模型，真实高峰到来时系统可能先撑不住。

## 推理怎么走

一次请求大致经过四个阶段：

1. **Prompt**：用户输入。
2. **Queuing**：排队。没有空闲的推理槽位时，时间就在这里变长，而模型还什么都没算。
3. **Prefill**：模型把整段输入读完，建起 KV cache。这是开始生成之前的那一步。
4. **Generation / Decode**：一次生成一个 token。

Token 是 LLM 的最小处理单位。很多主流模型大约 1 token ≈ 0.75 个英文词。中文没有这么整齐的换算——不要用词数去跟别人的 token 数对赌。

- **ISL**：输入 token（用户问题、system、历史、CoT、RAG 文档）。长输入让 prefill 更重、TTFT 更大。
- **OSL**：输出 token。长输出让 decode 走得更久，ITL 更容易被 KV 的增长拖累。
- **Context length**：每一步生成时能看见的总 token（已输入 + 已生成），被最大窗口拦住。窗口是上限，不是下限。

**Streaming** 边生成边把 token 块推给用户，聊天体感更快。非流式则整段生成完再返回。

更深的背景：[Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（本地导读在 `../performance-tuning/mastering-llm-techniques.md`）。

## 指标

下面几张是按笔记重画的学习图，不是官方原图。TTFT / ITL / Prefill / Decode 仍用英文。英文页保留原站附图。

![一次请求上的三把尺子](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/01-ttft-itl-generation.png)

### TTFT

从提交查询到收到**第一个非空** token。这就是用户要等多久才看见输出开始出现。GenAI-Perf 和 LLMPerf 都会丢掉空内容的初始响应——没有字的「第一包」不算第一个 token。

![走到第一个输出 token](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/02-first-token.png)

TTFT ≈ 排队 + prefill + 网络。Prompt 越长，attention 越要在完整输入上建 KV cache，TTFT 越大。多请求并行时，一个请求的 prefill 可以和另一个的 generation 重叠：你的等待里，可能藏着别人的生成。

### e2e latency

从发出请求到收到最后一个 token。流式模式下 detokenize 可能发生很多次。

![e2e_latency](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/03-e2e.png)

```
e2e_latency = TTFT + generation_time
```

`generation_time` 是第一个 token 到最后一个 token。GenAI-Perf 会去掉最后的 done / 空响应，免得把结束标记也算进生成时间。

### ITL / TPOT

连续输出 token 之间的平均时间。**GenAI-Perf / AIPerf 不含 TTFT；LLMPerf 常常把 TTFT 算进去。** 这是两把尺子最容易对不齐的地方。

![ITL / TPOT](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/04-itl.png)

GenAI-Perf：

```
ITL = (e2e_latency - TTFT) / (output_tokens - 1)
```

只刻画 decode。输出变长时 KV 变大，attention 对已有长度近似线性，但 decode 通常不是 compute-bound，而是受显存带宽限制。ITL 稳定，说明显存和带宽还健康。

### TPS

**系统 TPS**：所有并发请求合计的输出 token/秒。并发升高时上升，直到 GPU 饱和；再往上，有时会往下掉。

![一场基准的时间轴](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/05-bench-timeline.png)

- GenAI-Perf：总输出 token /（第一个请求发出 → 最后一个请求的最后响应）
- LLMPerf：总输出 token / 整个测试墙钟。会把造 prompt、准备请求、存响应算进去。单并发时这些开销有时能占到 **33%**。一种算法在给生成计时，另一种在给整场测试计时。

GenAI-Perf 用滑动窗口取稳态，warmup / cooldown 不计入。

**单用户 TPS** = OSL / e2e_latency，输出足够长时趋近 `1/ITL`。系统并发升高时，系统 TPS 升、单用户 TPS 降：整体更忙，每一个请求更慢。这不是 bug，这是物理。

### RPS

平均每秒成功完成的请求数。请求有大有小，RPS 自己很少能讲完故事；它要和 ISL/OSL 一起看。

## 参数与实践

### 业务决定 ISL/OSL

| 场景 | ISL | OSL |
|---|---|---|
| 翻译（语言/代码） | ~500–2000 | ~500–2000 |
| 生成（代码/故事/邮件） | ~100 | ~1000 |
| 摘要 / RAG / 多轮 | ~1000 | ~100 |
| 推理模型（显式 CoT） | ~100 | ~1000–10000 |

有生产流量，就用真实 prompt。合成数据很干净，真实用户很脏——脏才是上线以后要面对的分布。

### 负载控制

**Concurrency N**：始终保持 N 个在途请求。一个走了，立刻补上一个。这是控负载最常用的方式。

注意：LLMPerf 按批发 N 个，等整批结束再发下一批，批末并发会掉到 0。**GenAI-Perf / AIPerf 全程维持 N 个活跃请求。** 两种工具说的「并发」不是同一种调度。

**Max batch size**：引擎同时真正在算的请求数，可以小于并发。`concurrency > max_batch × 副本` 时，多出来的请求在排队，TTFT 会涨——涨的不是算力，是排队。

**Request rate**：按到达率发。到达超过吞吐时，在途请求会无限堆积。官方建议 benchmark 用 **concurrency**。QPS 压测留给真正想模拟泊松到达的时候。

扫描并发：从 1 扫到略大于 max batch。超过 max batch 后吞吐饱和、延迟继续涨。那条弯下去的曲线，就是以后谈 SLA 时要拿出来的图。

### 其他

- 基准测试设 **`ignore_eos=True`**，生成到 `max_tokens`，OSL 才可控。生产里请尊重 EOS。测试里让它生成到上限，是为了让尺子公平。
- 采样（greedy / top_p / top_k / temperature）会影响速度。greedy 不用归一化排序。同一套测试采样必须固定。换温度等于换了条件再比成绩。

## 小结

先对齐指标定义，再用 concurrency 扫出延迟–吞吐曲线，才谈得上成本和 SLA。后续：系列第 2 篇 NIM 实测、第 3 篇 TensorRT-LLM 调优、第 4 篇 TCO。
