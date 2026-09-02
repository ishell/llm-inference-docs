---
source: https://vllm.ai/blog/2025-12-16-vllm-sr-amd
lang: en
fetched: 2026-09-01
---

# AMD × Semantic Router: control plane on GPU

Chinese: `../../zh/vllm/blog/serving/semantic-router-amd.md`  
Live demo: [MoM on AMD](semantic-router-mom-amd.md).

Three pillars: signal routing (including Multi-LoRA), cross-instance semantic cache / Response store, guardrails (PII / jailbreak / hallucination). Two paths: vLLM on ROCm for router SLMs + many LLMs; ONNX Runtime at the front door. They frame routing as governance — gates on actions, untrusted inputs, long-term state. Longer: train an encoder router on AMD, public betas, GPU CI. Partnership essay; few kernel numbers. Slack `#semantic-router`.

Local figures (copyright remains with the original site; study copies):

![amd 0](../../../../assets/vllm/blog/serving/semantic-router-amd/01-amd-0.png)

![amd 1](../../../../assets/vllm/blog/serving/semantic-router-amd/02-amd-1.png)

![amd 2](../../../../assets/vllm/blog/serving/semantic-router-amd/03-amd-2.png)

![amd 3](../../../../assets/vllm/blog/serving/semantic-router-amd/04-amd-3.png)
