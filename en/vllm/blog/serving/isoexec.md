---
source: https://vllm.ai/blog/2026-08-21-isoexec
lang: en
fetched: 2026-09-01
---

# IsoExec: one execution contract for trainer and engine

Chinese: [zh/vllm/blog/serving/isoexec.md](../../../../zh/vllm/blog/serving/isoexec.md)  
SkyRL × vLLM × Megatron. 8×H100, Qwen3.5-35B-A3B sync DAPO: mean rollout–train logprob gap on contract-covered regions **below 1e-6**, ~**25%** wall-clock vs the SkyRL baseline then. Repo: https://github.com/zanderjiang/SkyRL-IsoExec

On-policy assumes the same policy twice. Floats are non-associative: kernels, batch shapes, TP/EP/SP layouts change token probabilities. VeXact / Fireworks: mismatch can destabilize GRPO and clip nearly half the tokens.

Two pieces:

1. **Execution contract**: each (region, case) pins implementation, accum dtype, split-K leaves. `semantic` / `numerical_policy` SHA-256 must match; deployment buffers need not.
2. **Unified model**: batch-invariant GEMM/attention/norm + deterministic MoE combine. `pik` fixes a binary tree along K; NCCL moves partials. EP combines in routing order, not rank order.

Linear attention is worse: chunkwise train/prefill vs recurrent decode — mean abs ~ , max 0.25. Recurrent-everywhere makes prefill 4×+. **CPR**: recurrent state at chunk boundaries, parallel scan inside; decode resyncs rounding every chunk. Vs native mixed: train ~1.43×, prefill ~1.67×, decode ~1.38×, bitwise.

No clear reward lift in 50 steps — too short to see stability. Next: Blackwell, CP invariance, sparse attn, Block-FP8 MoE. Earlier “two model copies, matched kernels”: [bitwise RL](bitwise-rl.md).

Local figures (copyright remains with the original site; study copies):

![unified execution abstraction](../../../../assets/vllm/blog/serving/isoexec/01-unified_execution_abstraction.png)

![pik figure](../../../../assets/vllm/blog/serving/isoexec/02-pik_figure.png)

![result logprob diff](../../../../assets/vllm/blog/serving/isoexec/03-result_logprob_diff.png)

![result time](../../../../assets/vllm/blog/serving/isoexec/04-result_time.png)

![result reward](../../../../assets/vllm/blog/serving/isoexec/05-result_reward.png)
