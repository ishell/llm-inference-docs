---
source: https://vllm.ai/blog/2026-06-09-announcing-vime
lang: en
fetched: 2026-09-01
---

# vime: slime train + vLLM rollout, `--vllm-` CLI prefix

Chinese: `../../zh/vllm/blog/serving/vime.md`  
ROCm: [vime-rocm](vime-rocm.md).

slime trains; vLLM rollouts. Don’t keep two dictionaries — `--vllm-` pins inference knobs on one CLI. GB200 vs H200 step time ~**1.72×** in their sweep. If train GPUs and sample GPUs run different engines, weight sync and logprob alignment crack. vime’s claim is one vLLM path.

Local figures (copyright remains with the original site; study copies):

![arch v1](../../../../assets/vllm/blog/serving/vime/01-arch_v1.png)

![Qwen3 30B A3B GB200 vs H200 step bar](../../../../assets/vllm/blog/serving/vime/02-Qwen3-30B-A3B_GB200_vs_H200_step_bar.png)

![Qwen3 4B Training raw reward compare](../../../../assets/vllm/blog/serving/vime/03-Qwen3-4B_Training_raw_reward_compare.png)

![Qwen3 30B A3B MoE R3 Comparison](../../../../assets/vllm/blog/serving/vime/04-Qwen3-30B-A3B_MoE_R3_Comparison.png)

![Qwen3 30B A3B GB200 vime baseline compare](../../../../assets/vllm/blog/serving/vime/05-Qwen3-30B-A3B_GB200_vime_baseline_compare.png)

![GLM 4.5 Air GB200 precision](../../../../assets/vllm/blog/serving/vime/06-GLM-4.5-Air_GB200_precision.png)
