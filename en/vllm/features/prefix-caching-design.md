---
source: https://docs.vllm.ai/en/stable/design/prefix_caching/
lang: en
fetched: 2026-08-31
---

# vLLM Prefix Caching (design)

User-facing: `features/prefix-caching.md`. This is the mechanism.

Scheduler: hash prompt tokens → `get_computed_blocks()` hits → `allocate_slots()` pops the free-queue head (that may evict a cached block) → full blocks are inserted so the same batch can reuse them. Finished requests free blocks (refcount 0) onto the **tail** of the free queue in reverse order (LRU-ish).

User API: `enable_prefix_caching=True`. Hash algo: `--prefix-caching-hash-algo` (`sha256` default; `xxhash` faster, weaker isolation).
