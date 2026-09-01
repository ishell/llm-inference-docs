---
source: https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference
lang: en
fetched: 2026-09-01
---

# Bitwise on-policy: vLLM × TorchTitan

Chinese: `../../zh/vllm/blog/serving/bitwise-rl.md`  
Qwen3 1.7B, GSM8K demo. RFC #28326 / #27433.

Trainer and sampler pick different kernels (batch-parallel vs within-instance parallel). Tiny numeric gaps get amplified by RL. `batch_inv_OFF` loses reward over 100 steps; bitwise-on (`kl_div` always 0) trains in fewer steps to a higher reward. They import vLLM’s batch-invariant forwards (fused SiLU MLP, RMSNorm+residual) into TorchTitan and register simple backwards. Sync: trainer and `VLLMRolloutEngine` alternate on one host — illustrative on-policy, not large async RL.

Cost then ~**2.4×** slower; Titan had no matching `torch.compile`, so vLLM stayed eager. Two model copies remain — one edit breaks parity. Later [IsoExec](isoexec.md) attacks that with a contract and a unified model. Read with [Native RL](native-rl.md) and [token IDs](agent-lightning.md).
