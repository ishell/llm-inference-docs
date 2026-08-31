---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
lang: zh
fetched: 2026-08-31
---

# TRT-LLM 运行时选项

不用重建引擎。

**容量调度**

- `GUARANTEED_NO_EVICT`（默认）：已开始的请求不会被暂停。
- `MAX_UTILIZATION`：尽量塞满；KV 不够可能暂停——吞吐更好，尾延迟有风险。
- `STATIC_BATCH`：遗留，生产别用。

**Context chunking：** 默认 `FIRST_COME_FIRST_SERVED`（整体通常更好）；`EQUAL_PROGRESS` 让各请求 TTFT 更齐。

**KV 显存**

- `kv_cache_free_gpu_mem_fraction` 默认 0.90。GPU 独占可试 **0.95**。不能 1.0（要留输入输出）。
- `max_tokens_in_paged_kv_cache`：不清楚就别设；最终取它和按比例算出的较小值。
- KV 越大，吞吐通常越高。

**`max_attention_window_size`：** sliding window。小于 `max_seq_len` 省算力和显存，可能掉精度。
