---
source: https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Athena v0.2：换底座，当系统脑

英文对照：`en/vllm/blog/serving/semantic-router-athena.md`  
原文：https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release  
视觉路径的坑见 [vision](semantic-router-vision.md)。

Iris 把路由拆成信号链。Athena 换模型栈：`mmbert-embed-32k-2d-matryoshka`（32k、号称 1800+ 语言、307M；STS 80.5；768→256 约留 99%；22L→6L 早退约 3.3×）+ `mom-multilingual-class`（intent / jailbreak / PII / fact-check… merged 与 LoRA）。`multi-modal-embed-small`：文/图/音频同一 384d，约 120M。ONNX + Flash Attention。ClawOS：用路由/记忆/安全编排多只 OpenClaw。记忆、RAG、长上下文、ROCm、dashboard。数字是发布页，不是你的延迟预算。

本地图（原文版权仍归原站；学习对照用）：

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
