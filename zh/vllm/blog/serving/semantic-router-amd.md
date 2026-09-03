---
source: https://vllm.ai/blog/2025-12-16-vllm-sr-amd
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# AMD × Semantic Router：GPU 上的控制面

英文对照：[en/vllm/blog/serving/semantic-router-amd.md](../../../../en/vllm/blog/serving/semantic-router-amd.md)  
原文：https://vllm.ai/blog/2025-12-16-vllm-sr-amd  
现场 demo 见 [MoM on AMD](semantic-router-mom-amd.md)。

三根柱：信号路由（含 Multi-LoRA）、跨实例语义缓存 / Response 存储、护栏（PII / jailbreak / 幻觉）。两条部署：vLLM on ROCm 跑路由 SLM + 多 LLM；前门超高频用 ONNX Runtime。他们把路由说成治理层——动作/输入/长期状态三道闸。长期：在 AMD 上训 encoder 路由模型、发布随附公测、GPU CI。合作愿景文，kernel 数字少。Slack `#semantic-router`。

本地图（原文版权仍归原站；学习对照用）：

![amd 0](../../../../assets/vllm/blog/serving/semantic-router-amd/01-amd-0.png)

![amd 1](../../../../assets/vllm/blog/serving/semantic-router-amd/02-amd-1.png)

![amd 2](../../../../assets/vllm/blog/serving/semantic-router-amd/03-amd-2.png)

![amd 3](../../../../assets/vllm/blog/serving/semantic-router-amd/04-amd-3.png)
