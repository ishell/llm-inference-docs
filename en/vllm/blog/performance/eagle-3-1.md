---
source: https://vllm.ai/blog/2026-05-26-eagle-3-1
lang: en
fetched: 2026-09-01
---

# EAGLE 3.1: holding attention drift

Chinese: [zh/vllm/blog/performance/eagle-3-1.md](../../../../zh/vllm/blog/performance/eagle-3-1.md)  
Nightly / then-upcoming v0.22.0. Kimi K2.6 NVFP4, TP4, GB200, non-disagg, SPEED-Bench coding: ~**2.03×** per-user output TPS at c=1, ~1.71× at c=4, ~1.66× at c=16.

Chat templates, long context, OOD system prompts shrink EAGLE-3 accept length. **Attention drift**: deeper speculation, drafter attention leaves sink tokens and stares at its own tokens. Two causes: fused input dominated by higher-layer hidden; unnormalized residual grows magnitude across steps.

3.1: **FC normalization** on each target hidden before the FC; next step eats **post-norm** hidden — more like recursively calling the drafter than stacking layers on the target. Long-context accept length up to ~**2×** vs EAGLE-3. Still `method: eagle3`; old checkpoints work.

```
--speculative-config '{"model":"lightseekorg/kimi-k2.6-eagle3.1-mla","method":"eagle3","num_speculative_tokens":3}'
```

Training: TorchSpec. Read with [spec-decode](spec-decode.md) and [P-EAGLE](p-eagle.md).

Local figures (copyright remains with the original site; study copies):

![pre norm vs post norm](../../../../assets/vllm/blog/performance/eagle-3-1/01-pre-norm-vs-post-norm.png)

![tpot baseline vs eagle31](../../../../assets/vllm/blog/performance/eagle-3-1/02-tpot_baseline_vs_eagle31.png)
