---
source: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
lang: en
fetched: 2026-09-01
---

# DeepSeek-V3.2-Exp: sparse attention in continuous batching

2025-09-29. Day-0.  Later compression stack: [deepseek-v4.md](deepseek-v4.md). GB300 V3.2 vs R1: [gb300-deepseek.md](../serving/gb300-deepseek.md). Wide-EP: [gb200-wideep.md](../serving/gb200-wideep.md).

DSA: lightning indexer picks top-2048, then sparse attention. Separate indexer K cache from MLA KV. Prefill vs decode layouts; batching marks causal windows with `ks` / `ke`.

MLA KV per token: **656 B** = 512× `float8_e4m3` NoPE + 4× `float32` scales + 64× `bfloat16` RoPE (unquantized). Indexer keys stored **per block**: values then scales — hence `block_size` 64 (FlashMLA too). One token’s indexer cache is not contiguous. FP8 KV: [fp8-kvcache.md](../performance/fp8-kvcache.md).

`deep_gemm.fp8_mqa_logits(...)`. Prefill: `ks=0`, `ke=range(n-q,n)`. Multi-request: concatenate Q and context, offset `ks`/`ke`. High batch × long context materializes full logits before row-wise topk. Fused topk (TileLang); quantize MLA latent and indexer keys as they hit the page table.

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp --tensor-parallel-size 8
```

Then: 16×H100 / 8×H200 / 8×B200. EP had a bug. Roadmap: llm-d + NIXL P/D, AMD/TPU (`vllm-ascend` / `vllm-mlu` already had V3.2), masked MHA for short prefill. Indexer still young; GB300 post: one DSA layer ~**2.7×** MLA kernel time.

Local figures (copyright remains with the original site; study copies):

![dsa explained](../../../../assets/vllm/blog/architecture/deepseek-v32/01-dsa-explained.png)

![mla indexer block](../../../../assets/vllm/blog/architecture/deepseek-v32/02-mla-indexer-block.png)
