---
source: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
lang: en
fetched: 2026-08-31
---

# KV Cache System

Stores past K/V so decode does not recompute them. Also: cross-request reuse, host offload, prioritized eviction, variable windows, MQA/GQA.

Runtime page (`trtllm-runtime-flags.md`) only sets how much memory and sliding window. This page is the structure.

## Blocks

A pool of blocks, each holding a fixed token count. **Tokens per block = power of two > 1**, set at engine build. Layers with the same head count and window share a pool; extra pools for GQA/MQA / mixed windows. Multi-pool split of free memory is static after init (NVIDIA calls this suboptimal).

Filled blocks enter a **radix tree**. A later request with a matching prefix reuses compute and memory.

## Eviction and offload

Need a blank block → evict. Prioritized LRU, priority 0–100 (100 = keep). Drain the lowest priority fully before the next. Same priority: least recently used.

GPU eviction can copy KV to **host** blocks; those stay searchable until evicted from secondary. Same scheme on both tiers.

Caveat: **only leaves are evictable** (no descendants in the radix tree). Fine for full attention; bad for limited-window layers. To be fixed.

`host_cache_size` (bytes, default **0**). Blocks below `secondary_offload_min_priority` (default **35**) skip offload and drop from GPU.

## Retention

Per-request `TokenRangeRetentionConfig` (prompt ranges + optional `duration_ms`; expires to priority 35). Decode tokens: `decode_retention_policy` / `decode_duration_ms`. `None` duration = never expires. `transfer_mode` is debug — do not use.

## Reuse switches

`enable_block_reuse` default True.

`scheduler_config.enable_prefix_aware_scheduling` (default True) only uses **estimated** reusable tokens for admission / token-budget math. Set False to disable those estimates; **actual reuse still follows `enable_block_reuse`**.

Partial reuse: `enable_partial_reuse` default on. `copy_on_partial_reuse` copies matched tokens into a new block so several requests can share a partial hit.

## Memory and dtype

`free_gpu_memory_fraction` default **0.9**, in (0, 1). If `max_tokens` is also set, allocate the lesser. `dtype` default `auto`.

`max_attention_window` is a per-layer int list, repeated if shorter than layer count: `[4096, 256]` = full, limited, full, limited, …

## `cache_salt`

Mixed into the block-key hash. Only matching salt can reuse. Isolation is digest equality only (no token re-compare). Hash must be cryptographic with a 256-bit digest (SHA-256). Do **not** substitute a non-crypto hash.

## Multimodal UUIDs

Optional `multi_modal_uuids` on `TextPrompt`. Cache key is `BLAKE3(UUID || Content)`. Original UUID is emitted in KV events. `None` falls back to content-only hash.

## Which manager

`use_kv_cache_manager_v2` default `auto` (model preference, else V1 C++). Override with true/false.

Default V2: Hybrid Mamba (NemotronH, Qwen3-Next); DeepSeek-V4; GPT-OSS (VSWA); Gemma3/4. Gemma4 hybrid/sparse is **unconditionally** V2.

Two-model spec decode (e.g. Eagle3 with `eagle3_one_model=False`) cannot use V2: `auto` falls back to V1; explicit `true` errors.

Hybrid Mamba snapshots live under `kv_cache_config.mamba_state_config`; set `avg_seq_len` or V2 warns and uses `max_seq_len / 2`.

Deprecated: `use_uvm`; `sink_token_length` ignored on PyTorch backend.
