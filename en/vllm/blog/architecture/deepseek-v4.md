---
source: https://vllm.ai/blog/2026-04-24-deepseek-v4
lang: en
fetched: 2026-09-01
---

# Serving DeepSeek V4 in vLLM

Chinese: [zh/vllm/blog/architecture/deepseek-v4.md](../../../../zh/vllm/blog/architecture/deepseek-v4.md)  
V4-Pro 1.6T / V4-Flash 285B, 1M context. Image `vllm/vllm-openai:deepseekv4-cu130`.

V4 attention compresses KV and compute together: shared K/V (then inverse RoPE), `c4a` / `c128a` across tokens, DSA over top compressed slots, 128-token window for locality. At 1M, bf16 ≈ 9.62 GiB/sequence vs ~83.9 GiB for a 61-layer V3.2-style stack (~8.7×). Production uses fp4 indexer cache and fp8 attention cache, roughly another 2×.


Local figures (copyright remains with the original site; study copies):

![c4a animation](../../../../assets/vllm/blog/architecture/deepseek-v4/01-c4a_animation.gif)

![kv cache comparison](../../../../assets/vllm/blog/architecture/deepseek-v4/02-kv-cache-comparison.svg)

![decode path](../../../../assets/vllm/blog/architecture/deepseek-v4/03-decode-path.svg)

## Three allocator moves

Heterogeneous compress ratios would shred paged KV. vLLM:

1. **One logical block = 256 native positions.** `c4a` holds 64 compressed entries, `c128a` holds 2. Slot maps, scheduling, prefix-hit all use that unit.
2. **Compressor residual as sliding-window KV.** Window 8 for C4, 128 for C128. Prefix cache and disagg reuse SWA instead of a side residual path.
3. **Five cache kinds into three page-size buckets** so pools do not fragment each other.

Prefill still bf16 KV; decode is partly token-wise fp8. CUDA graphs, MTP, and P/D follow the SWA abstraction.

## Hot path

Three fusions: compressor+RMSNorm+RoPE+insert (~1.4–3×); inverse RoPE+fp8 quant (~2–3×); horizontal Q-norm + KV RoPE + K insert (~10–20× vs naive). Indexer / main compress / SWA insert overlap on CUDA streams. Suggested flags: `--kv-cache-dtype fp8`, `--block-size 256`, EP+DP, FP4 indexer. Manifold-Constrained Hyper-Connections and MoE deltas are omitted in the post — easier to port than attention.

Read with [FP8 KV](../performance/fp8-kvcache.md) and [Wide-EP](../serving/large-scale.md).
