---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
lang: zh
fetched: 2026-08-30
---

# 指标 — NVIDIA NIM LLM 压测指南

本节定义常见 LLM 推理指标。不同工具的实现不一样，**定义对齐之前不要直接比数字**。如何用 AIPerf 采集这些指标，见「用 AIPerf 实测」。

图 1：常用 LLM 推理性能指标总览。

## TTFT（Time to First Token，首 token 时间）

TTFT 衡量用户要等多久才能看到模型开始输出：从提交查询到收到**第一个非空 token** 的时间。

图 2：TTFT 包含第一个输出 token 的 tokenize 和 detokenize。

> NVIDIA AIPerf 会丢掉没有内容或空字符串的初始响应。第一个响应里没有 token 时，TTFT 没有意义。

TTFT 一般包含：**排队时间 + prefill 时间 + 网络延迟**。Prompt 越长，TTFT 越大，因为 attention 要用完整输入序列建好 KV cache，生成循环才能开始。生产环境里多请求并行时，一个请求的 prefill 可以和另一个请求的 generation 重叠。

> 传统 Web 压测工具（如 K6）也可以用 HTTP 时序事件给出某种 TTFT。

## 端到端请求延迟（e2e latency）

从提交查询到收到完整响应的时间，包含排队、组 batch、网络。

> 流式模式下，部分结果返回时 detokenize 可能发生多次。

单请求：

```
e2e_latency = TTFT + Generation_time
```

`Generation_time` 是从收到第一个 token 到收到最后一个 token 的时长。AIPerf 会去掉最后的 `[done]` 信号或空响应，不计入 e2e。

## ITL（Inter-token Latency / TPOT）

连续两个 token 之间的平均时间，也叫 TPOT（time per output token）。

不同工具是否把 TTFT 算进平均值，并不统一。**AIPerf 不含 TTFT。**

AIPerf 定义：

```
ITL = (e2e_latency − TTFT) / (Total_output_tokens − 1)
```

分母减 1，是为了让 ITL 只刻画 **decode 阶段**。

输出越长，KV cache 和显存越大；每多一个新 token，attention 计算对「已有序列长度」近似线性增长。ITL 稳定，通常说明显存管理、带宽和 attention 计算比较健康。

## TPS（Tokens Per Second）

**系统级 TPS**：所有同时进行的请求合计的输出 token 吞吐。负载升高时系统 TPS 上升，直到 GPU 算力饱和，再往上可能下降。

一次基准测试的时间线（n 个请求）：

- `L_i`：第 i 个请求的 e2e 延迟
- `T_start`：测试开始
- `Tx`：第一个请求发出的时刻
- `Ty`：最后一个请求的最后一次响应时刻
- `T_end`：测试结束

AIPerf 定义：

```
TPS = Total_output_tokens / (Ty − Tx)
```

这是整批算出来的，不是实时仪表盘。若配置了 warmup，warmup 不计入。

**单用户 TPS** = 该请求的 OSL / e2e_latency。输出足够长时，它趋近于 `1/ITL`。并发升高时，系统 TPS 通常上升，单用户 TPS 下降。

## RPS（Requests Per Second）

平均每秒成功完成的请求数：

```
RPS = total_completed_requests / (Ty − Tx)
```
