---
source: https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AIPerf 指标

官方 Metrics Reference 按**计算阶段**排，不是按「你关心什么用户体验」排。NIM 手册第 2 章（`../benchmarking/nim-02-metrics.md`）是同一把尺子的人话版。这里补公式、前提、以及两套吞吐为什么不能横比。

流式指标一律要求：`--streaming`、会吐 token 的 endpoint、至少一包**非空**内容。空首包不算第一句话——那是清嗓。

## 三种计算

| 类型 | 何时算 | 结果长什么样 |
|---|---|---|
| Record | 每条请求（及其若干 response chunk） | 分布：avg / min / max / p50 / p90 / p99 |
| Aggregate | 整场基准上累加 | 单个值：`request_count`、时间戳边界 |
| Derived | 其它指标算完再套公式 | 单个值或再分布：TPS、RPS、goodput |

Record 不知道整场；Derived 看不见单条 HTTP。所以「系统 TPS」和「这个人觉得有多快」不是同一只秒表。

## 流式：等待与缝

时间戳在内部是纳秒，表上是毫秒，除法用秒。

**TTFT** — 发出请求到第一包非空内容：

```
ttft_ns = request.content_responses[0].perf_ns - request.start_perf_ns
```

里面有网络、排队、prefill、以及第一个 chunk 的生成。更长的 prompt 通常更大的 TTFT。

**TTST** — 第一包到第二包。用来把「启动税」从稳态 decode 里拆出来。至少两包非空。

**TTFO** — 到第一个**非 reasoning** 输出 token。会想很久才开口的模型：TTFT 可能已经在思考，TTFO 才是用户看见正文。没有 reasoning 时 TTFO = TTFT。

**ITL** — 稳态字缝，**不含 TTFT**：

```
ITL = (request_latency − TTFT) / (output_sequence_length − 1)
```

至少两包非空，且 TTFT、request latency、OSL 都合法。这是用户听你说话时，字与字之间的缝。

**ICL** — 相邻 **chunk** 到达间隔的整条分布，不是平均到 token。chunk 里可能挤了好几个 token。抖动、组 batch、网线性格，看 ICL 比看一个平均 ITL 更诚实。

## 吞吐：系统 vs 这个人

**Output Token Throughput Per User** = `1 / ITL`（秒）。不含 TTFT。并发升高时，这个数字通常往下掉——整体更快，每一个人更慢。

**Prefill Throughput Per User** = `ISL / TTFT_seconds`。只描述读 prompt 有多快。

**E2E Output Token Throughput**（每条请求）= `OSL / request_latency`。分母含 TTFT 和排队，比 per-user 更矮，非流式也能算。

**Output Token Throughput**（系统，Derived）= `total_osl / benchmark_duration`。分母是整场墙钟，**含 TTFT**。不要拿它和 per-user 比高低。

**Total Token Throughput** = `(total_isl + total_osl) / benchmark_duration`。

NIM 手册里的系统 TPS 用 `Ty − Tx`（第一条发出到最后一条收完）。AIPerf 参考页用 `benchmark_duration`。对表的时候先问分母是哪一段墙钟。warmup 不进成绩。

## Token 怎么数

客户端用 tokenizer，`add_special_tokens=False`。

- **ISL**：prompt token
- **output_token_count**：给用户看的输出，不含独立 `reasoning_content` 字段；若思考写在普通 `content` 里（例如 think 标签），会算进去，除非你另外滤
- **OSL** = output_token_count + reasoning_token_count。没有 reasoning 时 OSL = 输出 token 数

服务若在 JSON 里带回 `usage`，AIPerf 还能打一组 Usage* 和 Diff%——客户端计数和服务端计数对不齐时，先别吵 TPS，先问谁在数 token。

## 端到端与个数

**Request latency** = 最后一包的 `perf_ns − start_perf_ns`。流式算到最后一 chunk。

**Request count** = 成功且合法的条数。  
**Error request count** = 失败（网络 / HTTP / 超时 / 其它）。  
**Request throughput (RPS)** = `request_count / benchmark_duration`。请求有胖有瘦，RPS 单独很少能说明问题。

## Goodput

先用 `--goodput` 设 SLO，例如 `"time_to_first_token:370 request_latency:648"`。一条请求必须**同时**满足全部阈值才算 good。

```
goodput = good_request_count / benchmark_duration
good_request_fraction = good_request_count / (request_count + error_request_count)
```

失败请求进分母。服务在负载下丢掉流量，不能靠「活下来的都很快」装成合规。Goodput 永远 ≤ Request Throughput。控制台可能不显示 fraction（`NO_CONSOLE`），JSON/CSV 里有。

吞吐告诉你每秒打发走多少人。Goodput 告诉你每秒有多少人还愿意再来。容量规划用后者，否则你会按一个用户正在受苦的 TPS 去买卡。

## 其它族（不逐条抄）

官方同一页还有：图像张数 / 吞吐 / 延迟；视频推理时间与峰值显存；音频时长与 RTFx；reasoning token 计数；usage 全套（含 cache read/write/miss、tool、accepted/rejected prediction）；OSL mismatch；错误请求的 ISL。要用哪一族，回原页按 tag 滤：`STREAMING_ONLY`、`GOODPUT`、`HTTP_TRACE`、`USAGE_DIFF_ONLY` 等。

HTTP 时序（`--show-trace-timing` / `--export-http-trace`）把一次请求拆成 sending / waiting（到第一字节 body）/ receiving。那是网线和握手，不是 TTFT 的同义词。教程：https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/http-trace-metrics-guide
