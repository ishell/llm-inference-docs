---
source: https://vllm.ai/blog/2026-05-14-verl-omni
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# verl × Omni：扩散 RL 的 rollout 不另起炉灶

英文对照：[en/vllm/blog/serving/verl-omni.md](../../../../en/vllm/blog/serving/verl-omni.md)  
原文：https://vllm.ai/blog/2026-05-14-verl-omni  
H800。v0.2 见 [verl-omni-v020](verl-omni-v020.md)。

训练还在 verl；扩散 rollout 走 vLLM-Omni。FlowGRPO 做 OCR 类奖励。同一套 serve 路径既给推理又给 RL 采样，避免「训练引擎一份权重、推理引擎另一份调度」。数字看原图：吞吐、步时、奖励曲线。接 [Omni](vllm-omni.md)。

本地图（原文版权仍归原站；学习对照用）：

![verl omni arch](../../../../assets/vllm/blog/serving/verl-omni/01-verl-omni-arch.png)

![flowgrpo algo](../../../../assets/vllm/blog/serving/verl-omni/02-flowgrpo-algo.png)

![hidden trail step 0](../../../../assets/vllm/blog/serving/verl-omni/03-hidden-trail-step-0.png)

![hidden trail step 120](../../../../assets/vllm/blog/serving/verl-omni/04-hidden-trail-step-120.png)

![make a wish step 0](../../../../assets/vllm/blog/serving/verl-omni/05-make-a-wish-step-0.png)

![make a wish step 120](../../../../assets/vllm/blog/serving/verl-omni/06-make-a-wish-step-120.png)

![validation reward](../../../../assets/vllm/blog/serving/verl-omni/07-validation-reward.png)

![rollout reward](../../../../assets/vllm/blog/serving/verl-omni/08-rollout-reward.png)

![zero std ratio](../../../../assets/vllm/blog/serving/verl-omni/09-zero-std-ratio.png)

![pg clipfrac](../../../../assets/vllm/blog/serving/verl-omni/10-pg-clipfrac.png)
