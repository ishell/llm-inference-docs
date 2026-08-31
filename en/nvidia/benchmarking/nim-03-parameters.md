---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html
lang: en
fetched: 2026-08-30
---

# Parameters and Best Practices

After reviewing Metrics, configure test parameters and sweep ranges that reflect your deployment. The settings below help produce meaningful, comparable benchmark results.

## Use Cases

Your application’s use cases influence input sequence length (ISL) and output sequence length (OSL). Sequence lengths affect how fast a system processes input, builds the KV cache, and generates output tokens. Longer input sequences increase prefill memory requirements and TTFT. Longer output sequences increase generation memory requirements and ITL. Understand the distribution of inputs and outputs in your LLM deployment to optimize hardware utilization.

Common use cases and likely ISL/OSL pairs:

- **Translation** (language or code): similar ISL and OSL, typically 500 to 2,000 tokens each.
- **Generation** (code, story, email, search-based content): OSL near 1,000 tokens, ISL near 100 tokens.
- **Summarization** (retrieval, chain-of-thought, multi-turn chat): ISL near 1,000 tokens, OSL near 100 tokens.

If you have production traffic, you can use real prompts as inputs.

## Load Control

**Concurrency N** is the number of concurrent clients, each with one active request. Equivalently, it is the number of requests the LLM service handles concurrently. After each request receives a complete response, another request is sent so the system always has N active requests.

Concurrency is the most common way to describe and control load on an inference system.

**Max batch size:** the group of simultaneous requests the inference engine processes. This can be a subset of the concurrent requests. If concurrency exceeds `max_batch_size × active replicas`, some requests wait in a queue and TTFT can rise due to queueing delay.

**Request rate** controls load by the rate at which new requests are sent. A constant rate `r` means one request every `1/r` seconds. A Poisson rate sets the average inter-arrival time.

AIPerf supports both concurrency and request rate. **Prefer concurrency for most benchmarks:** with request rate, outstanding requests can grow without bound when arrival exceeds throughput.

Sweep concurrency from 1 to a value slightly greater than max batch size. When concurrency exceeds max batch size, requests queue. Throughput generally saturates near max batch size while latency keeps rising.

## Other Parameters

**`ignore_eos`:** Most LLMs emit an end-of-sequence (EOS) token when generation should stop. In real usage, inference should honor EOS. **For benchmarking, set `ignore_eos` to `true`** so the model generates until `max_tokens`, reaching the intended OSL and giving consistent measurements.

**Sampling vs greedy decoding:** Sampling strategy can change generation speed. Greedy decoding picks the highest logit and skips normalizing/sorting the distribution. Keep the sampling method consistent within one benchmarking setup. See Hugging Face generation strategies for details.
