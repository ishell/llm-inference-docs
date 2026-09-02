---
source: https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Mixture-of-Models：从选模型到造系统

英文对照：`en/vllm/blog/serving/semantic-router-mom.md`  
原文：https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom  
发布页自称 5k stars / 150+ 贡献者 / HF 累计 30 万下载——当时的社区数字。

MoM ≠ MoE：MoE 在一次前向里按 token 选专家；MoM 在请求级编排不同架构、不同卡上的模型。时间线：14 类快/慢 → Iris 信号链 → Athena 控制面 → Themis 可运营合同 → Fusion / Micro-Agent 选 **协作配方**。下一步：同一份 versioned contract 训练、评测、导出、部署、用一个 API 调用。协作算法见 [micro-agent](semantic-router-micro-agent.md)。

本地图（原文版权仍归原站；学习对照用）：

![hero](../../../../assets/vllm/blog/serving/semantic-router-mom/01-hero.png)

![evolution](../../../../assets/vllm/blog/serving/semantic-router-mom/02-evolution.png)

![research arc](../../../../assets/vllm/blog/serving/semantic-router-mom/03-research-arc.png)

![fragmentation before mom](../../../../assets/vllm/blog/serving/semantic-router-mom/04-fragmentation-before-mom.png)

![fragmentation after mom](../../../../assets/vllm/blog/serving/semantic-router-mom/05-fragmentation-after-mom.png)

![execution topologies](../../../../assets/vllm/blog/serving/semantic-router-mom/06-execution-topologies.png)

![preference models](../../../../assets/vllm/blog/serving/semantic-router-mom/07-preference-models.png)

![four planes](../../../../assets/vllm/blog/serving/semantic-router-mom/08-four-planes.png)

![artifact resolution lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/09-artifact-resolution-lifecycle.png)

![matched compute evaluation](../../../../assets/vllm/blog/serving/semantic-router-mom/10-matched-compute-evaluation.png)

![mom lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/11-mom-lifecycle.png)

![portable realizations](../../../../assets/vllm/blog/serving/semantic-router-mom/12-portable-realizations.png)

![next stage roadmap](../../../../assets/vllm/blog/serving/semantic-router-mom/13-next-stage-roadmap.png)

![community](../../../../assets/vllm/blog/serving/semantic-router-mom/14-community.png)
