---
source: https://vllm.ai/blog/2025-12-13-speculators-v030
lang: en
fetched: 2026-09-01
---

# Speculators v0.3.0: train EAGLE-3 drafts

Chinese: `../../zh/vllm/blog/performance/speculators-v030.md`  
Later DFlash / online training: [v0.5.0](speculators-v050.md). Hidden export: [extract-hidden-states](../architecture/extract-hidden-states.md).

Each verifier wants its own draft. v0.3 wires offline data → train → `vllm serve`. Data: three hidden layers, token ids, loss mask on assistant spans, verifier distribution. A custom worker intercepts prefill hidden; `.pt` files land async; `token_freq.pt` builds a small draft vocab (t2d/d2t). Training uses Eagle3 train-time-testing + FlexAttention sparse masks, concatenating sequences on the seq dim instead of padding.

Artifacts ship `speculators_config` in `config.json`: verifier path, algorithm, default N. Short:

```
vllm serve RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
```

Long form swaps a quantized verifier and `num_speculative_tokens`. Train/serve then: Llama 3.x, Qwen3 dense/MoE, GPT-OSS; Llama 4 vision serve-only. They quote ~**1.5–3×** latency when the draft matches — not a promise.
