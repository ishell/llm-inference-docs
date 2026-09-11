---
source: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
lang: en
fetched: 2026-08-30
---

# LLM Inference Benchmarking: Fundamental Concepts

Source: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/

This is part 1 of NVIDIA’s LLM latency–throughput series. Part 2: GenAI-Perf and NIM.

The cost of an LLM deployment depends on how many queries it can process per second while remaining responsive and accurate enough. This post is about throughput and latency, not accuracy.

NVIDIA’s inference stack includes Dynamo, TensorRT-LLM, and NIM. For benchmarking they published GenAI-Perf (now being succeeded by AIPerf; the concepts still apply).

Client-side tools disagree on how they define and compute metrics. Do not compare numbers across tools until definitions match.


Local figures (copyright remains with the original site; study copies):

![llm inference performance metrics](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/01-llm-inference-performance-metrics.png)

![time to first token process](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/02-time-to-first-token-process.png)

![end to end request latency](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/03-end-to-end-request-latency.png)

![itl average time between consecutive token generations](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/04-itl-average-time-between-consecutive-token-generations.png)

![event timeline benchmarking run](../../../assets/nvidia/benchmarking/blog-01-fundamental-concepts/05-event-timeline-benchmarking-run.png)

## Load testing vs performance benchmarking

- **Load testing:** simulate lots of concurrent traffic; find capacity, autoscaling, network, resource issues.
- **Performance benchmarking:** measure model throughput, latency, and token-level metrics under controlled load (efficiency, optimization, configuration).

Do both.

## How LLM inference works

Prompt → Queue → Prefill → Generation (one token at a time).

A token is the LLM’s unit of text. Roughly 1 token ≈ 0.75 English words on many popular models.

- **ISL:** tokens into the model (query, system, history, CoT, RAG).
- **OSL:** tokens the model generates.
- **Context length:** tokens visible at each step (input + generated so far), capped by the model window.

Streaming returns chunks as they are generated. Non-streaming waits for the full answer.

See also: Mastering LLM Techniques: Inference Optimization.

## Metrics

**TTFT:** time to first (non-empty) token. Includes queue + prefill + network. Longer prompts → larger TTFT because KV cache is built over the full input. Prefill of one request can overlap generation of another.

**e2e_latency = TTFT + generation_time.** GenAI-Perf drops the final done/empty signal.

**ITL / TPOT:** average time between consecutive output tokens. GenAI-Perf excludes TTFT; LLMPerf includes it.

```
ITL_GenAI-Perf = (e2e_latency − TTFT) / (output_tokens − 1)
```

Stable ITL means healthy memory bandwidth and attention.

**TPS (system):** total output tokens / (first request sent → last response of last request) in GenAI-Perf. LLMPerf divides by the whole benchmark wall clock (prompt construction, request prep, storing responses). NVIDIA observed those overheads can be ~33% at concurrency 1. GenAI-Perf uses a sliding window and drops warmup/cooldown.

**TPS per user** ≈ OSL / e2e_latency → 1/ITL as OSL grows. More concurrency: system TPS up, per-user TPS down.

**RPS:** completed requests / (Ty − Tx).

## Parameters

Use-case ISL/OSL (order of magnitude):

- Translation: 500–2000 / 500–2000
- Generation: ~100 / ~1000
- Summarization / RAG / multi-turn: ~1000 / ~100
- Reasoning models: ~100 / 1000–10000

**Concurrency N:** keep exactly N in-flight requests. LLMPerf sends batches of N then drains (concurrency falls to 0 at batch end). GenAI-Perf keeps N active the whole time.

**Max batch size:** engine’s simultaneous requests. If concurrency > max_batch × replicas, requests queue and TTFT rises.

**Request rate:** can unbounded-queue if arrival > throughput. Prefer concurrency.

Sweep concurrency from 1 to a bit above max batch: throughput saturates near max batch; latency keeps climbing.

For benchmarks set **`ignore_eos=True`**. Keep sampling (greedy / top_p / top_k / temperature) fixed; greedy is cheaper.

## Get started

Align metric definitions, then sweep concurrency and plot latency–throughput before talking cost or SLA.
