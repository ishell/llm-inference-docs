---
source: https://docs.vllm.ai/en/stable/design/prefix_caching/
lang: en
fetched: 2026-09-04
---

# Automatic Prefix Caching (design)

Chinese: [zh/vllm/features/prefix-caching-design.md](../../../zh/vllm/features/prefix-caching-design.md)  
User-facing: [prefix-caching.md](prefix-caching.md). Anatomy scheduler is the same picture. Official figures (overview / free-queue / example times) live on the docs site; not copied here.

Prefix caching of KV blocks is a free lunch that does **not** change model outputs. Public endpoints and most open-source engines do it. vLLM uses a **hash-based** approach: each KV block is hashed from the tokens **in** the block plus the tokens **before** it.

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

Block hash is `hash(tuple[components])`:

- **Parent hash** of the previous block.
- **Block tokens** (exact IDs — collision resistance).
- **Extra hashes**: LoRA IDs, multimodal input hashes, `cache_salt` for multi-tenant isolation.

Only **full** blocks are cached.

As of **v0.11**, default hash is `sha256` (collision risk of older keys). `--prefix-caching-hash-algo`:

| Algo | Serialization | Notes |
|---|---|---|
| `sha256` (default) | Python `pickle` | Hashes may **not** be reproducible across Python / vLLM versions |
| `sha256_cbor` | `cbor2` | Reproducible, cross-language; recommended for deterministic cache across environments |
| `xxhash` | pickle + xxHash 128-bit | Faster, non-crypto. Optional `xxhash` package. Collision theoretically can leak private data in multi-tenant setups |
| `xxhash_cbor` | canonical CBOR + xxHash | Reproducible xxHash. Optional `xxhash` package |

## Multimodal hashing

`[IMG]` becomes placeholder tokens, then image embeddings at prefill. Placeholders alone would collide across images, so the **frontend image-processor hash** is an extra hash on every block that contains those placeholders. Example on the page: block size 16, 41 placeholders → four blocks, each carrying `<image hash>`.

## Cache isolation (`cache_salt`)

Optional per-request salt is injected into the **first** block’s hash. Only matching salts reuse KV. Stops timing attacks that infer cached content from latency. Example JSON on the page: `"cache_salt": "your-cache-salt"` on a chat request.

## Data structure

Implemented in the KV cache manager. Simplified `KVCacheBlock`: `block_id` (immutable), `block_hash` (set when full, reset on eviction), `ref_cnt`, plus `prev_free_block` / `next_free_block` for an intrusive free queue.

Design points:

1. All blocks allocated at manager init — a pool; no Python object churn; every block is always trackable.
2. Doubly-linked pointers **on the block** → O(1) move-to-tail, no extra `deque` wrapper.

At init: **block pool**, **free-queue head/tail**, **cache blocks** (`hash → block IDs`), **request blocks** (`request ID → allocated IDs`). **Figure: Component Overview** on the docs page.

## Operations

### New request

1. Scheduler: `kv_cache_manager.get_computed_blocks()` — hash prompt, look up cache.
2. `allocate_slots()`:
   1. Count new blocks needed; return if not enough.
   2. **Touch** computed blocks: `ref_cnt += 1`; pull them off the free queue if unused by others (so they cannot be evicted).
   3. Allocate by popping **heads** of the free queue. A cached head is **evicted** — nobody else may reuse it from now on.
   4. A newly filled block is inserted into the cache **immediately**, so later requests in the **same batch** can hit it.

### Running request

`allocate_slots()` again: count → pop heads (evict if cached) → append token IDs → cache a block when it fills.

### Duplicated blocks (V1 append-only table)

Block size 4. Request 1 prompt `ABCDEF`, decode length 3 → over time cache blocks 0 (`ABCD`) then 1 (`EFGH`). Request 2 identical greedy: Time 0 reuses 0 but allocates **new** block 3 for `EFG`; Time 1 fills 3 as `EFGH` and caches it — **duplicate** of block 1. V0 would free 3 and rewrite the table to `[0, 1]`. V1 block tables are **append-only**, so `[0, 3]` stays; duplication is cleaned when the request is freed.

### Free

Finished request, `ref_cnt = 0`: blocks go to the **tail** of the free queue in **reverse** order. The last block hashed more tokens and is less reusable → evict first. **Figure: Free queue after a request is freed.**

### Eviction (LRU)

When the free-queue **head** is still cached:

1. Pop head (LRU).
2. Remove its ID from the cache map.
3. Clear its hash.

## Walk-through (block size 4, 10 blocks)

- **Time 1:** empty cache; new request allocates 4 blocks; 3 full and cached; 4th partial (3/4).
- **Time 2:** request 0 fills block 3, asks for another; cache 3, allocate 4.
- **Time 3:** request 1, 14 prompt tokens, first 10 same as request 0 → only first **2** blocks (8 tokens) hit; 3rd block matches 2 of 4 tokens.
- **Time 4:** request 0 finishes. Blocks 2, 3, 4 enter the free queue reverse (2 and 3 still cached). 0 and 1 stay off the queue (still used by request 1).
- **Time 5:** request 1 finishes and frees.
- **Time 6:** request 2, 29 prompt tokens, first 12 same as request 0. Free queue was `7-8-9-4-3-2-6-5-1-0`; cache-hit blocks 0, 1, 2 are **touched and removed** before allocation → queue `7-8-9-4-3-6-5`. Allocated: 0, 1, 2 (cached), then 7, 8, 9, 4, 3 (**evicted**).
