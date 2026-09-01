---
source: https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release
lang: en
fetched: 2026-09-01
---

# Athena v0.2: new base, system brain

Chinese: `../../zh/vllm/blog/serving/semantic-router-athena.md`  
Vision-path bug: [vision](semantic-router-vision.md).

Iris made routing a signal chain. Athena rebuilds the stack: `mmbert-embed-32k-2d-matryoshka` (32k, 1800+ languages claimed, 307M; STS 80.5; 768→256 ~99% kept; 22L→6L early-exit ~3.3×) plus `mom-multilingual-class` (intent / jailbreak / PII / fact-check… merged and LoRA). `multi-modal-embed-small`: text/image/audio in one 384d, ~120M. ONNX + Flash Attention. ClawOS: route/memory/safety over OpenClaw workers. Memory, RAG, long context, ROCm, dashboard. Release-page numbers, not your latency budget.
