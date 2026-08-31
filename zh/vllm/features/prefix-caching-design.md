---
source: https://docs.vllm.ai/en/stable/design/prefix_caching/
lang: zh
fetched: 2026-08-31
---

# vLLM Prefix Caching（实现）

对外说明见 `prefix-caching.md`。这里是机制：

调度：对 prompt token 做 hash → `get_computed_blocks()` 命中 → `allocate_slots()` 从空闲队列头取块（可能驱逐缓存块）→ 填满的块立刻入缓存，同一 batch 也能复用。请求结束且引用计数为 0 时，块按反序接到空闲队列**尾**（近似 LRU）。

用户侧：`enable_prefix_caching=True`。Hash：`--prefix-caching-hash-algo`（默认 `sha256`；`xxhash` 更快、隔离更弱）。
