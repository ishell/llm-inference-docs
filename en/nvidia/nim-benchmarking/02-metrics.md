---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
lang: en
fetched: 2026-08-30
---

# Metrics — NVIDIA NIM LLMs Benchmarking

This section defines common LLM inference metrics. Tool implementations vary, so compare results only when definitions align. Refer to Using AIPerf to Benchmark for collecting these metrics with AIPerf.

Figure 1. Overview of popular LLM inference performance metrics.

## Time to First Token

Time to first token (TTFT) measures how long you wait before seeing the model’s output. It is the time from query submission to the first received token, if the response is not empty.

Figure 2: TTFT includes both tokenization and de-tokenization for the first output token.

> NVIDIA AIPerf disregards initial responses with no content or an empty string. TTFT is meaningless when the first response contains no token.

Time to first token generally includes request queuing time, prefill time, and network latency. Longer prompts increase TTFT because the attention mechanism uses the full input sequence to create the KV cache before generation begins. In a production application, several requests can be in progress at the same time, so one request’s prefill phase can overlap with another request’s generation phase.

> Traditional web service benchmarking tools such as K6 can also provide TTFT by using timing events in the HTTP request.

## End-to-End Request Latency

End-to-end request latency (`e2e_latency`) measures how long it takes from submitting a query to receiving the full response. This includes queueing, batching, and network latency.

> In streaming mode, de-tokenization can run multiple times as partial results are returned.

For an individual request:

```
e2e_latency = TTFT + Generation_time
```

`Generation_time` is the duration from the first token received to the final token received. AIPerf also removes the final `[done]` signal or empty response so it is not included in e2e latency.

## Inter-token Latency

Inter-token latency (ITL) is the average time between consecutive tokens. Also known as time per output token (TPOT).

Tools differ on whether TTFT is included in the average. **AIPerf excludes TTFT.**

AIPerf defines ITL as:

```
ITL = (e2e_latency − TTFT) / (Total_output_tokens − 1)
```

The first token is excluded so ITL characterizes only the decoding part of request processing.

With longer output sequences, the KV cache and its memory cost grow. Attention cost also grows linearly with the sequence generated so far. Consistent ITL indicates efficient memory management, memory bandwidth, and attention computation.

## Tokens Per Second

Total tokens per second (TPS) per system is total output-token throughput across all simultaneous requests. As load increases, system TPS rises until GPU compute saturates; beyond that it can decrease.

Timeline of a benchmark with n requests:

- `L_i`: end-to-end latency of request i
- `T_start`: start of benchmark
- `Tx`: timestamp of the first request
- `Ty`: timestamp of the last response of the last request
- `T_end`: end of benchmark

AIPerf defines TPS as:

```
TPS = Total_output_tokens / (Ty − Tx)
```

This is batch-oriented, not a live running metric. AIPerf excludes warmup if you configure a warmup phase.

**TPS per user** is OSL / e2e_latency for each request, which asymptotically approaches `1/ITL` as output length increases. As concurrency rises, system TPS usually increases while TPS per user decreases.

## Requests Per Second

Requests per second (RPS) is the average number of successfully completed requests per second:

```
RPS = total_completed_requests / (Ty − Tx)
```
