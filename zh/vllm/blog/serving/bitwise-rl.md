---
source: https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 逐 bit 对齐：vLLM × TorchTitan 的 on-policy

英文对照：`en/vllm/blog/serving/bitwise-rl.md`  
原文：https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference  
Qwen3 1.7B、GSM8K 演示。RFC #28326 / #27433。

训练和推理选不同 kernel（batch 维并行 vs 单条内部并行），微小数值差会被 RL 放大。`batch_inv_OFF` 一百 step 奖励更差；打开逐 bit（`kl_div` 恒为 0）步数更少、奖励更高。做法：把 vLLM batch-invariant 前向（含融合 SiLU MLP、带 residual 的 RMSNorm）搬进 TorchTitan，补简单 backward。同步：同一台机器上 trainer 与 `VLLMRolloutEngine` 轮流——示意 on-policy，不是大规模 async RL。

当时代价约 **2.4×** 更慢；`torch.compile` 还没在 Titan 侧对齐，vLLM 被迫 eager。两份 model code 仍在，改一处就破等价——后来 [IsoExec](isoexec.md) 用合同 + 统一模型啃这点。和 [Native RL](native-rl.md)、[token IDs](agent-lightning.md) 一起读。

本地图（原文版权仍归原站；学习对照用）：

![rl script demo](../../../../assets/vllm/blog/serving/bitwise-rl/01-rl-script-demo.png)

![reward comparison](../../../../assets/vllm/blog/serving/bitwise-rl/02-reward-comparison.png)
