---
source: https://vllm.ai/blog/2026-06-09-announcing-vime
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vime：slime 训练 + vLLM rollout，CLI 用 `--vllm-` 前缀

英文对照：[en/vllm/blog/serving/vime.md](../../../../en/vllm/blog/serving/vime.md)  
原文：https://vllm.ai/blog/2026-06-09-announcing-vime  
2026-06-09。署名 **vime Contributors and the vLLM Team**。仓库：[vllm-project/vime](https://github.com/vllm-project/vime)。slime 的训练栈 + vLLM 的 rollout，同一套架构。vLLM 自己的 Native RL API 见 [native-rl.md](native-rl.md)。ROCm 续篇：[vime-rocm.md](vime-rocm.md)。GB200 / H200 / A100 数字是他们的合同，不是你的 SLA。

推理侧旋钮用 `--vllm-` 钉在同一条 CLI；默认 rollout 入口 `vime.rollout.vllm_rollout`。Qwen3-30B-A3B、8 卡 colocate、dapo-math-17k、GRPO：GB200 步时约 **147 s**，H200 约 **252 s** → 约 **1.72×**。Qwen3-4B A100：`train_rollout_logprob_abs_diff` 约 **0.011**，基线漂到约 **0.77**。MoE R3：约 **0.019 → 0.013**。GLM-4.5-Air GB200：`raw_reward` 均值约 **0.56**（100 步）；logprob 差 **0.02–0.03**，均值约 **0.028**。训练卡和采样卡若各用各的引擎，权重同步和 logprob 对齐都会裂——vime 的主张是同一条 vLLM 路径。

本地图（原文版权仍归原站；学习对照用）：

![arch v1](../../../../assets/vllm/blog/serving/vime/01-arch_v1.png)

![Qwen3 30B A3B GB200 vs H200 step bar](../../../../assets/vllm/blog/serving/vime/02-Qwen3-30B-A3B_GB200_vs_H200_step_bar.png)

![Qwen3 4B Training raw reward compare](../../../../assets/vllm/blog/serving/vime/03-Qwen3-4B_Training_raw_reward_compare.png)

![Qwen3 30B A3B MoE R3 Comparison](../../../../assets/vllm/blog/serving/vime/04-Qwen3-30B-A3B_MoE_R3_Comparison.png)

![Qwen3 30B A3B GB200 vime baseline compare](../../../../assets/vllm/blog/serving/vime/05-Qwen3-30B-A3B_GB200_vime_baseline_compare.png)

![GLM 4.5 Air GB200 precision](../../../../assets/vllm/blog/serving/vime/06-GLM-4.5-Air_GB200_precision.png)

**Figure（architecture）。** Megatron 训练经解耦 Data Buffer 接到 vLLM rollout。

**Figure（step speed）。** Qwen3-30B-A3B 在 GB200 vs H200 的步速。

**Figure（Qwen3-4B）。** vime 对基线（`raw_reward` / 对齐）。

**Figure（MoE R3）。** routing replay 对训推 mismatch。

**Figure（GB200 MoE）。** colocate 训推对齐对基线。

**Figure（GLM-4.5-Air）。** 奖励往上；logprob 对齐还在。

## 愿景

[slime](https://github.com/THUDM/slime)（在 GLM 一类模型上验过）：开放、轻、效率好——原生不接 vLLM。vLLM：社区里最活跃的推理引擎，多平台，迭代快。vime 要把这两段接成一条，不必在硬件栈、训练稳定、推理性能之间单挑。

## 位置

vLLM 已经坐在好几家后训练框架下面（字母序）：[NeMo RL](https://github.com/NVIDIA-NeMo/RL)、[OpenRLHF](https://github.com/openrlhf/openrlhf)、[verl](https://github.com/verl-project/verl) 以及其他。vime 是 slime 形状的桥，对齐两边的发版钟。社区仍继续养那些别的集成。

## 架构

slime 的三阶段、训推解耦；rollout 后端换成 vLLM：

- **Training（Megatron）** — 更新参数，把权重同步到 rollout。
- **Rollout（vLLM + Router）** — 采样，带奖励 / verifier 信号。
- **Data Buffer** — 两边之间管 prompt 注入和自定义 rollout 逻辑。

## 能力

- **好用。** slime / Megatron 的参数习惯；vLLM 旋钮走 `--vllm-`。默认 rollout：`vime.rollout.vllm_rollout`。
- **训推对齐。** Dense 和 MoE：长跑里 `train_rollout_logprob_abs_diff` 落在可控区间。MoE 的 **R3**（routing replay）再削错位。
- **算法和模型。** GRPO、PPO；Qwen3 Dense/MoE、GLM-4.5——端到端例子，CI 验过的路径。
- **多硬件。** 训练资源、rollout 资源、集群拓扑抽象掉，RL 管线可以跟着 vLLM 的硬件插件走。

## 验证和基准

Qwen3-30B-A3B，8 卡 colocate，dapo-math-17k，GRPO：GB200 步时约 **147 s**，H200 约 **252 s**。同一框架里，GB200 端到端步速约是 H200 的 **1.72×**。

### Qwen3-4B on A100

GRPO，4 训练 + 4 推理 **non-colocate**，gsm8k。vime 的 `train_rollout_logprob_abs_diff` 全程约 **0.011**。基线漂到约 **0.77**。

### Qwen3-30B-A3B MoE + R3

A100，4 训练 + 4 推理，dapo-math-17k，**EP=4**。R3 routing replay：logprob 差约 **0.019 → 0.013**。

### Qwen3-30B-A3B MoE on GB200

8 卡 colocate，dapo-math-17k。vime 和基线的 `raw_reward` 贴着走。两边的 `train_rollout_logprob_abs_diff` 都约 **0.018**；这套里基线没有持续漂。

### GLM-4.5-Air on GB200

GRPO，8 卡 colocate，dapo-math-17k。`raw_reward` **100** 步往上，均值约 **0.56**。`train_rollout_logprob_abs_diff` **0.02–0.03**，均值约 **0.028**。

## 路线

- 更深接 vLLM：Router、PD 分离、FP8、多模型 serving。
- 更多硬件，走 vLLM 插件。
- 训练效率和算法：全异步管线、训推错位校正、Agentic RL（多轮工具、多智能体）、MoE 和 VLM 跟上。

## Quick start

像 slime：配 Megatron 训练资源和 vLLM rollout 资源，准备 checkpoint 和数据，起 `train.py` 或 `train_async.py`。

- **文档：** [Quick Start](https://github.com/vllm-project/vime/tree/main/docs/en/get_started)
- **例子：** `scripts/`、`examples/` —— Qwen3-4B、Qwen3-30B-A3B MoE、GLM-4.5-Air。

## 社区

Apache 2.0。站在 slime、Megatron-LM、vLLM 肩上。

- **代码和文档：** [github.com/vllm-project/vime](https://github.com/vllm-project/vime)
- **贡献：** issue 和 PR；pre-commit 管风格。
- **反馈：** 经验、性能数据、功能建议去 GitHub。

## 致谢

**贡献者：** Ao Shen、kaiyuan、princepride、Dakai An、knlnguyen1802、gcanlin、SamitHuang、Meihan-chen。

感谢 [slime](https://github.com/THUDM/slime)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[vLLM](https://github.com/vllm-project/vllm) 的维护者。组织上的支持：Kaichao You、Roger Wang、Hongsheng Liu、Xiyuan Wang。
