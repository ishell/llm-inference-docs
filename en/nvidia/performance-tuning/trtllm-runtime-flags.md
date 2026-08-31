---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
lang: en
fetched: 2026-08-31
---

# TRT-LLM Useful Runtime Options

No rebuild. Apply when calling LLM-API / serve.

**Capacity scheduler**

- `GUARANTEED_NO_EVICT` (default): a started request is never paused.
- `MAX_UTILIZATION`: pack as many requests as possible; may pause if KV fills — better throughput, riskier tail latency.
- `STATIC_BATCH`: legacy, skip for production.

**Context chunking policy:** `FIRST_COME_FIRST_SERVED` (default, usually better overall) vs `EQUAL_PROGRESS` (more even TTFT).

**KV memory**

- `kv_cache_free_gpu_mem_fraction` default 0.90. If the GPU is dedicated, try **0.95**. Cannot be 1.0 (need room for I/O).
- `max_tokens_in_paged_kv_cache`: leave unset unless you know the cap; engine uses the min of this and the fraction-derived size.
- More KV memory → usually more throughput.

**`max_attention_window_size`:** sliding-window attention. Smaller than `max_seq_len` saves compute/memory, may drop accuracy.
