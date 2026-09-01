---
source: https://vllm.ai/blog/2026-05-14-verl-omni
lang: en
fetched: 2026-09-01
---

# verl × Omni: diffusion RL rollout without a second engine

Chinese: `../../zh/vllm/blog/serving/verl-omni.md`  
H800. v0.2: [verl-omni-v020](verl-omni-v020.md).

Training stays in verl; diffusion rollout uses vLLM-Omni. FlowGRPO for OCR-style rewards. One serve path for inference and RL sampling — not a second scheduler for training weights. Throughput, step time, reward curves on the original page. Read with [Omni](vllm-omni.md).
