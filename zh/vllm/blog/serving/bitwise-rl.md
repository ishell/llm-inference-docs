---
source: https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 逐 bit 对齐：vLLM × TorchTitan 的 on-policy

英文对照：[en/vllm/blog/serving/bitwise-rl.md](../../../../en/vllm/blog/serving/bitwise-rl.md)  
原文：https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference  
2025-11-10。署名 **vLLM and TorchTitan Teams**。页上作者：Bram Wasti、Wentao Ye、Teja Rao、Michael Goin、Paul Zhang、Tianyu Liu、Natalia Gimelshein、Woosuk Kwon、Kaichao You、Zhuohan Li。说明：[torchtitan/experiments/deterministic_vllm_rl](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/deterministic_vllm_rl)。RFC：[#28326](https://github.com/vllm-project/vllm/issues/28326)、[#27433](https://github.com/vllm-project/vllm/issues/27433)。后来的合同 + 一份模型：[isoexec](isoexec.md)。Pause / 权重 API：[native-rl](native-rl.md)。

开源的逐 bit on-policy RL：[TorchTitan](https://github.com/pytorch/torchtitan) 训，[vLLM](https://github.com/vllm-project/vllm) 采。底座是 [vLLM batch-invariant inference](https://docs.vllm.ai/en/latest/features/batch_invariance/)。演示：把 **Qwen3 1.7B** 做 RL 微调。

本地图（原文版权仍归原站；学习对照用）：

![rl script demo](../../../../assets/vllm/blog/serving/bitwise-rl/01-rl-script-demo.png)

训练和采样之间那一点数值缝，会被 RL 放大——跑出来不确定、也不稳（[He et al.](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)、[Yao, Liu et al.](https://fengyao.notion.site/off-policy-rl)、[Liu, Li et al.](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)）。他们在这套栈上核对过。

![reward comparison](../../../../assets/vllm/blog/serving/bitwise-rl/02-reward-comparison.png)

采样用的 kernel 和训练不一样（`batch_inv_OFF`）：**100 step 里奖励更差**。打开逐 bit（`batch_inv_ON`，`kl_div` 恒为 **0.0**）：步数更少，总奖励更高。

## Approach

训练框架和推理框架常因负载不同，选完全不同的 kernel。就算困在同一个推理引擎里，选择仍会动：高 batch 的 kernel 在 batch 维上并行；低 batch 的 kernel 更往 **单条内部** 并行，好把 GPU 核喂饱。这些差异已经够走出数值缝，RL 再把它放大。

这边两套框架：TorchTitan 训、vLLM 推。他们把前向里 **每一次 kernel 调用** 都审过，确认跨框架逐 bit。前向 kernel 来自 vLLM 的 [batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)；这些 op 的 [简单 backward](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/deterministic_vllm_rl/batch_invariant_backward.py) 是后来补的。

vLLM 有不少抠过的融合算子——SiLU MLP、带 residual 的 RMSNorm。要保住 bit，就把 **同一份前向** 搬进来（batch-invariant 路径上的 `SiluAndMul`、`rms_norm`）。这些 op 需要自定义 backward，用 TorchTitan 那套原味 PyTorch 注册即可。接线时，原来非 invariant 的 Titan 路径仍能用；开关走 vLLM 暴露的 `vllm_is_batch_invariant`，不多加一项配置。

RL 演示：通用脚本，**GSM8K**，正确性奖励。Trainer 用 TorchTitan 的工具；生成器是自己写的 `VLLMRolloutEngine`，包一层 generate 和权重更新。全部 **同步**，**同一台机器** 上 trainer 和 generator 轮流。这正是严格 on-policy。大规模异步 RL 不是这种写法（那条要看 [native-rl](native-rl.md)）。

## What’s Next

跟踪 RFC：[#28326](https://github.com/vllm-project/vllm/issues/28326)、[#27433](https://github.com/vllm-project/vllm/issues/27433)。页上四条：

**Unified model definition。** 模型代码仍是两份——一份训、一份推。第一次集成够用，长期脆：任何一侧稍改，等价就破。后来 [isoexec](isoexec.md) 用合同啃的，就是这份共享定义。

**Compilation support。** 当时 TorchTitan 模型没用 `torch.compile`，vLLM 只好 **eager**。纸面上拿掉这层约束并不难，但 Titan 侧要先有一份 `torch.compile` 过的模型。vLLM 自己已经能在 `torch.compile` 下保住 batch-invariance；跨框架兼容还得让训练那份对得上。

**RL performance。** 逐 bit 那次比非逐 bit **慢 2.4×**。下一步：把 batch-invariant kernel 调得更好，再加上编译。

**Wider model support。** 走出 Qwen3 1.7B；把审计工具和 backward 推广到更多算子类型，让训练–推理逐 bit 变成可复用的能力，而不是一个模型的演示。

Slack（原文链接）：[\#sig-post-training](https://vllm-dev.slack.com/archives/C07UUL8E61Z)、[\#sig-batch-invariant](https://vllm-dev.slack.com/archives/C09JVU355CG)。

## Background

源稿后半有一段标成 deprecated 的「为什么 bit 重要」，仍按原文件顺序放在 Approach / What’s Next 之后。

预训练那无数 FLOP 里，数值 mismatch 几乎看不见。预训练通常 **batch 固定**，同一套归约 kernel 反复上场，问题被绕过去。

RL 几乎总在跑 **另一套** 归约：它偏推理，被延迟和显存绑住。低 batch 推理 kernel 常常 **不 tiling** 就归约；训练 kernel 为了复用数据和抬高算力利用率，并行得很狠。生成器和训练器于是坐在 **完全不同的 kernel** 上。

训练因此在隐含意义上变成 **off-policy**：生成器吐出的，不必等于训练器拿同一输入会算出的。

浮点是二进制的科学计数：符号位 \(s\)、尾数 \(M\)、指数 \(e\)，每一段都按整数存、也按整数那样舍入。机器学习里最常见的 **bf16**，尾数只有 **7 bit**。**3.0** 能精确表示，**3.6** 不能——新的 bf16 值要圆到最近的可表示数。这条舍入若发生在一串加法的 **不同位置**，即便输入、权重、框架、硬件都一样，图里 **任何一处** 若派发了另一只（仍然正确的）kernel，输出就可以 **不一样**。

这是 on-policy 的数值半边。Token ID 那半边——Agent 字符串别再二次分词——见 [agent-lightning](agent-lightning.md)。
