---
source: https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Mixture-of-Models：从选模型到造系统

英文对照：`en/vllm/blog/serving/semantic-router-mom.md`  
原文：https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom  
发布页自称 5k stars / 150+ 贡献者 / HF 累计 30 万下载——当时的社区数字。图在原网页。

MoM ≠ MoE：MoE 在一次前向里按 token 选专家；MoM 在请求级编排不同架构、不同卡上的模型。时间线：14 类快/慢 → Iris 信号链 → Athena 控制面 → Themis 可运营合同 → Fusion / Micro-Agent 选 **协作配方**。下一步：同一份 versioned contract 训练、评测、导出、部署、用一个 API 调用。协作算法见 [micro-agent](semantic-router-micro-agent.md)。
