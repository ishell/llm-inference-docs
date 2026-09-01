---
source: https://vllm.ai/blog/2026-07-15-inkling
lang: en
fetched: 2026-09-01
---

# TML Inkling day-0

Chinese: `../../zh/vllm/blog/serving/inkling.md`  
2026-07-15. Demo: 4× GB200. Figures on the original page.

1T multimodal (text/image/audio → text), native 1M context. 66 layers: 11 full + 55 SWA GQA. Position is **relative attention**, not RoPE. Four window-4 **sconv**s per layer. MoE: 256 routed top-6 + 2 shared **expert sinks**. NVFP4 on routed experts only; 8 MTP heads in BF16. **AMD not yet** (needs a relative-attn kernel). **Not a new engine** — sconv cache is a virtual SWA KV layer.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
vllm serve thinkingmachines/Inkling-NVFP4 \
  --tokenizer-mode inkling --reasoning-parser inkling --tool-call-parser inkling \
  --enable-auto-tool-choice --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 8}' \
  --trust-remote-code
```

Channel-sharded sconv (reduce-scatter / all-gather). Lamport fused collectives: bs=1 **40 µs → 8 µs**. FA4 sheared-bias. MTP KV recomputed after rejection.

Demo SPEED-Bench 8K/1K: MTP8 **380 tok/s/user** (mean accept 4.5), no MTP **140**. MMAU/MMMU-Pro/BFCL within ~1 pp of reference; NIAH matches through 221K.
