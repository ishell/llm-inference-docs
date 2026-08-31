---
source: https://vllm.ai/blog/2025-02-21-aibrix-release
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# AIBrix：ByteDance 交出来的 vLLM 控制面

英文对照：`en/vllm/blog/serving/aibrix.md`  
原文：https://vllm.ai/blog/2025-02-21-aibrix-release  
2025-02-21。仓库：https://github.com/vllm-project/aibrix。2024 年初做起，已在字节多个业务上跑。单实例 vLLM 好开；规模上的路由、伸缩、容错是另一件事。他们的说法：系统与推理引擎 **共设计**，在 Kubernetes 上用云原生的方式搭推理。

首发能力（清单）：高密度 LoRA；LLM 网关与路由；按应用的 autoscaler；统一 AI runtime sidecar（指标、拉模型）；分布式推理；分布式 KV；异构卡混部（SLO 约束下省钱）；GPU 故障探测。

往后想做的：分布式 KV 覆盖 P/D 聚合、请求迁移、跨实例复用；把 QoS / 优先级 / 公平性做到请求级多租户；roofline profiling 去守 SLO。白皮书和文档在仓库里。Slack：`#aibrix`。

## FAQ（比功能清单更有用）

**和 production-stack 什么关系？** AIBrix 是字节开源、对着大规模和云原生；当时已生产 6+ 个月。production-stack 由 UChicago LMCache 团队管，从零搭积木、欢迎实验。production-stack 的长处是内置 KV 向优化（传输、blending、路由），尤其长上下文、prefill 重。近期他们计划借用 AIBrix 组件。

**是社区项目吗？** 放进 vllm-project 组织就是为了打开协作。

**和 KServe / KubeAI？** AIBrix 更贴 vLLM：快加载、autoscaling、LoRA，可以按「只有这一台引擎」来做选择。
