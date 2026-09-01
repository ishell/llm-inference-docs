---
source: https://vllm.ai/blog/2026-05-28-speculators-v050
lang: en
fetched: 2026-09-01
---

# Speculators v0.5.0: DFlash and online hidden

Chinese: `../../zh/vllm/blog/performance/speculators-v050.md`  
vLLM ≥0.20.0 (PR#38300).

EAGLE-3 is multi-step autoregressive; DFlash emits a **block in one forward**, non-causal inside the block. Starting a block at every position explodes the attention mask. They sample **anchors** only on loss positions so block count is independent of sequence length. Train flags: `--speculator-type dflash`, `--block-size`, `--max-anchors`. Gemma 4 31B DFlash accepts better on reasoning/code; ITL beats EAGLE-3 and a standalone FP8 verifier; FP8+DFlash is shorter still.

```
vllm serve -tp 2 RedHatAI/gemma-4-31B-it-speculator.dflash
```

Hidden extraction no longer hooks vLLM internals; it uses the HTTP path in [extract-hidden-states](../architecture/extract-hidden-states.md): online during training, or offline cache. Same data format; mix them (partial offline, fill online). Read with [parallel drafting](parallel-drafting.md).
