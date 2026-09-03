---
source: https://vllm.ai/blog/2026-08-20-verl-omni-v0-2-0
lang: en
fetched: 2026-09-01
---

# verl-Omni v0.2: request-level batching, gen 226s → 108s

Chinese: [zh/vllm/blog/serving/verl-omni-v020.md](../../../../zh/vllm/blog/serving/verl-omni-v020.md)  
Prior: [verl-omni](verl-omni.md).

v0.1 was “it runs”; v0.2 batches generation at request grain. Wall-clock gen **226s → 108s**. MMK12 val **0.833**. Rollout is usually the RL step’s fat part — wait-for-the-whole-batch → request-in/request-out is why the clock drops. Figures still on the original page.

Local figures (copyright remains with the original site; study copies):

![verl omni v0 2 0 blog overview](../../../../assets/vllm/blog/serving/verl-omni-v020/01-verl_omni_v0_2_0_blog_overview.png)

![qwen image gpu utilization](../../../../assets/vllm/blog/serving/verl-omni-v020/02-qwen-image-gpu-utilization.svg)

![qwen image timing gen](../../../../assets/vllm/blog/serving/verl-omni-v020/03-qwen-image-timing-gen.svg)

![qwen image timing step](../../../../assets/vllm/blog/serving/verl-omni-v020/04-qwen-image-timing-step.svg)

![omni ppo adapter flow](../../../../assets/vllm/blog/serving/verl-omni-v020/05-omni-ppo-adapter-flow.svg)

![mmk12 training rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/06-mmk12_training_rewards.svg)

![mmk12 val rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/07-mmk12_val_rewards.svg)
