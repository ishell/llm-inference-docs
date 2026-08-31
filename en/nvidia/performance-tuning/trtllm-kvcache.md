---
source: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
lang: en
fetched: 2026-08-31
---

# TRT-LLM KV Cache System

Pool of blocks (token count per block = power of two > 1). Layers sharing head count / window share a pool. Extra pools for GQA/MQA and mixed window sizes.

**Reuse:** filled blocks go into a radix tree; later requests with the same prefix skip compute and share memory. Eviction: prioritized LRU (priority 0–100). Leaves only (limitation for limited-attention layers). Optional **host offload** (`host_cache_size`); default 0. Priority below `secondary_offload_min_priority` (default 35) skips offload.

`enable_block_reuse` default True. Scheduler `enable_prefix_aware_scheduling` only affects admission/token-budget estimates, not actual reuse.

`free_gpu_memory_fraction` (default 0.9) vs `max_tokens`: allocate the lesser. `dtype` default auto from the model.

`cache_salt` isolates reuse (hashed into block keys; needs a real cryptographic hash). Multimodal: optional `multi_modal_uuids`.

V2 manager is default for some hybrid/sparse models (NemotronH, DeepSeek-V4, GPT-OSS VSWA, Gemma3/4). Two-model spec decode cannot use V2.
