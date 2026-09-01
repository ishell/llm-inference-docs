---
source: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
lang: en
fetched: 2026-09-01
---

# Seven ROCm attention backends

Chinese: `../../zh/vllm/blog/architecture/rocm-attention.md`  
Default Triton attention: [triton-attn](triton-attn.md). This post is **AITER FA’s three paths**.

`ROCM_AITER_FA` splits a batch into Prefill / Extend / Decode: new sequences use `flash_attn_varlen_func` (CDNA matrix cores); continuing long context uses chunked attention + LSE merge (~32K tokens per iteration); decode uses AITER `pa_fwd_asm`. The model runner reorders to `[decode:extend:prefill]` (`reorder_batch_threshold=1`) so each kernel sees contiguous memory.

Preshuffled KV:

```
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

Decode has zero layout conversion — about **15–20%** decode TPS vs a standard layout. Extend must gather shuffled cache back to standard layout before MHA. Official vs other ROCm backends: about **1.2–4.4×** system TPS — hardware and model vary; not a promise. Explicit software routing instead of one mega-kernel: easier to debug; same router from MI300X to MI355X.
