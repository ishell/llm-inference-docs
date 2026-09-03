---
source: https://vllm.ai/blog/2026-05-14-verl-omni
lang: en
fetched: 2026-09-01
---

# verl × Omni: diffusion RL rollout without a second engine

Chinese: [zh/vllm/blog/serving/verl-omni.md](../../../../zh/vllm/blog/serving/verl-omni.md)  
H800. v0.2: [verl-omni-v020](verl-omni-v020.md).

Training stays in verl; diffusion rollout uses vLLM-Omni. FlowGRPO for OCR-style rewards. One serve path for inference and RL sampling — not a second scheduler for training weights. Throughput, step time, reward curves on the original page. Read with [Omni](vllm-omni.md).

Local figures (copyright remains with the original site; study copies):

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
