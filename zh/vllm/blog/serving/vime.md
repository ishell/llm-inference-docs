---
source: https://vllm.ai/blog/2026-06-09-announcing-vime
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# vime：slime 训练 + vLLM rollout，CLI 用 `--vllm-` 前缀

英文对照：[en/vllm/blog/serving/vime.md](../../../../en/vllm/blog/serving/vime.md)  
原文：https://vllm.ai/blog/2026-06-09-announcing-vime  
ROCm 见 [vime-rocm](vime-rocm.md)。

slime 管训练；vLLM 管 rollout。参数不要两套词典——`--vllm-` 前缀把推理侧 knobs 钉在同一条 CLI。GB200 vs H200 他们报步时约 **1.72×**。RL 里「训练卡」和「采样卡」若各用各的引擎，权重同步和 logprob 对齐都会裂。vime 的主张是同一条 vLLM 路径。

本地图（原文版权仍归原站；学习对照用）：

![arch v1](../../../../assets/vllm/blog/serving/vime/01-arch_v1.png)

![Qwen3 30B A3B GB200 vs H200 step bar](../../../../assets/vllm/blog/serving/vime/02-Qwen3-30B-A3B_GB200_vs_H200_step_bar.png)

![Qwen3 4B Training raw reward compare](../../../../assets/vllm/blog/serving/vime/03-Qwen3-4B_Training_raw_reward_compare.png)

![Qwen3 30B A3B MoE R3 Comparison](../../../../assets/vllm/blog/serving/vime/04-Qwen3-30B-A3B_MoE_R3_Comparison.png)

![Qwen3 30B A3B GB200 vime baseline compare](../../../../assets/vllm/blog/serving/vime/05-Qwen3-30B-A3B_GB200_vime_baseline_compare.png)

![GLM 4.5 Air GB200 precision](../../../../assets/vllm/blog/serving/vime/06-GLM-4.5-Air_GB200_precision.png)
