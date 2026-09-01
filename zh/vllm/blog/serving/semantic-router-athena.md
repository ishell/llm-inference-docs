---
source: https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Athena v0.2：换底座，当系统脑

英文对照：`en/vllm/blog/serving/semantic-router-athena.md`  
原文：https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release  
图在原网页。视觉路径的坑见 [vision](semantic-router-vision.md)。

Iris 把路由拆成信号链。Athena 换模型栈：`mmbert-embed-32k-2d-matryoshka`（32k、号称 1800+ 语言、307M；STS 80.5；768→256 约留 99%；22L→6L 早退约 3.3×）+ `mom-multilingual-class`（intent / jailbreak / PII / fact-check… merged 与 LoRA）。`multi-modal-embed-small`：文/图/音频同一 384d，约 120M。ONNX + Flash Attention。ClawOS：用路由/记忆/安全编排多只 OpenClaw。记忆、RAG、长上下文、ROCm、dashboard。数字是发布页，不是你的延迟预算。
