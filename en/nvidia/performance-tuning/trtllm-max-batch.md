---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html
lang: en
fetched: 2026-08-31
---

# Tuning Max Batch Size and Max Num Tokens

In-flight batching mixes context (prefill) and generation in one iteration. Two build-time caps decide who gets scheduled:

- **`max_batch_size`**: max in-flight requests. Default **2048**. Too small → new requests stall. Sweep powers of 2.
- **`max_num_tokens`**: max packed tokens per iteration (padding removed). Default **8192**. Too small → cannot schedule large prompts; too large → KV cache starved (or OOM) on long context.

Scheduler prefers **generation tokens first**, then fills leftover token budget with new prefills. A request that would exceed the token budget cannot start unless **context chunking** is on (needs paged context FMHA).

Toy walkthrough in the official page: `max_batch_size=4`, `max_num_tokens=12`. Two 5-token prompts consume 10 tokens → leftover 2 cannot start a longer prompt without chunking. After they enter generation they only cost 1 token each, so more prefills fit — until the batch-size cap blocks the 5th request. When one request hits EOS it is evicted and the 5th can start.

## How to set

Python:

```python
build_config = BuildConfig(max_batch_size=512, max_num_tokens=2048)
```

CLI: `trtllm-build --max_batch_size … --max_num_tokens …`

Grid-search both if you can. Good `max_num_tokens` candidates: powers of 2 ≥ 1024.

## Case study (Llama-3.3-70B, 4×H100, after other build flags)

Max batch size (default vs 64 vs 512): **512** was the sweet spot (~20% more token/request throughput vs 2048; 64 bottlenecked). Latency barely moved.

Max num tokens at batch 512: **2048 / 8192 / 16384** were almost tied; 2048 slightly best on that workload. Always measure — the delta can be tiny or huge.

## Why always enable paged context attention

Chunked prefill lets a long prompt be split across iterations, so:

1. A long prompt is not blocked forever behind in-flight work (better worst-case TTFT).
2. `max_num_tokens` no longer needs to be ≥ longest prompt, which **frees GPU memory for KV cache**.

NVIDIA: enable it even if a given run shows little gain.

Tuned max batch + max num tokens vs previous page in that guide: **~+21% throughput**, latency within noise. Vs untuned baseline: **~+58% throughput**, ITL **~−53%**.
