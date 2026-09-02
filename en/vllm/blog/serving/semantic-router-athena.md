---
source: https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release
lang: en
fetched: 2026-09-01
---

# Athena v0.2: new base, system brain

Chinese: `../../zh/vllm/blog/serving/semantic-router-athena.md`  
Vision-path bug: [vision](semantic-router-vision.md).

Iris made routing a signal chain. Athena rebuilds the stack: `mmbert-embed-32k-2d-matryoshka` (32k, 1800+ languages claimed, 307M; STS 80.5; 768→256 ~99% kept; 22L→6L early-exit ~3.3×) plus `mom-multilingual-class` (intent / jailbreak / PII / fact-check… merged and LoRA). `multi-modal-embed-small`: text/image/audio in one 384d, ~120M. ONNX + Flash Attention. ClawOS: route/memory/safety over OpenClaw workers. Memory, RAG, long context, ROCm, dashboard. Release-page numbers, not your latency budget.

Local figures (copyright remains with the original site; study copies):

![athena 0](../../../../assets/vllm/blog/serving/semantic-router-athena/01-athena-0.png)

![athena 1](../../../../assets/vllm/blog/serving/semantic-router-athena/02-athena-1.png)

![athena 1b](../../../../assets/vllm/blog/serving/semantic-router-athena/03-athena-1b.png)

![athena 2](../../../../assets/vllm/blog/serving/semantic-router-athena/04-athena-2.png)

![athena 3](../../../../assets/vllm/blog/serving/semantic-router-athena/05-athena-3.png)

![athena 7](../../../../assets/vllm/blog/serving/semantic-router-athena/06-athena-7.png)

![athena 4](../../../../assets/vllm/blog/serving/semantic-router-athena/07-athena-4.png)

![athena 5](../../../../assets/vllm/blog/serving/semantic-router-athena/08-athena-5.png)

![athena 5b](../../../../assets/vllm/blog/serving/semantic-router-athena/09-athena-5b.png)

![athena 6](../../../../assets/vllm/blog/serving/semantic-router-athena/10-athena-6.png)

![athena 8](../../../../assets/vllm/blog/serving/semantic-router-athena/11-athena-8.png)

![athena 9](../../../../assets/vllm/blog/serving/semantic-router-athena/12-athena-9.png)

![athena 10](../../../../assets/vllm/blog/serving/semantic-router-athena/13-athena-10.png)

![athena 11](../../../../assets/vllm/blog/serving/semantic-router-athena/14-athena-11.png)
