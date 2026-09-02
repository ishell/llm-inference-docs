---
source: https://vllm.ai/blog/2025-02-21-aibrix-release
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# AIBrix：ByteDance 交出来的 vLLM 控制面

英文对照：`en/vllm/blog/serving/aibrix.md`  
原文：https://vllm.ai/blog/2025-02-21-aibrix-release  
2025-02-21。仓库：https://github.com/vllm-project/aibrix。2024 年初做起，已在字节多个业务上跑。白皮书和部署文档在仓库里。Slack：`#aibrix`。

单实例 vLLM 好开——那是 quickstart 的礼貌。规模上的路由、伸缩、容错是另一件事。AIBrix 的说法：系统与推理引擎 **共设计**，在 Kubernetes 上用云原生的方式搭推理，而不是把引擎当成一只普通 Deployment 塞进去。Clayton Coleman（GKE inference）和 Robert Nishihara（Ray）的站台原句在网页里，本地不抄。


本地图（原文版权仍归原站；学习对照用）：

![aibrix diagram](../../../../assets/vllm/blog/serving/aibrix/01-aibrix-diagram.png)

## 首发能力

不是路线图，是当时交出来的积木：

- **高密度 LoRA**：许多轻量适配器挤在同一套权重上。
- **LLM 网关与路由**：流量在多模型、多副本之间怎么走。
- **按应用的 autoscaler**：按这份负载的脾气伸缩，而不是只看 CPU。
- **统一 AI runtime sidecar**：指标口径、拉模型、管模型。
- **分布式推理**：跨节点铺开。
- **分布式 KV**：跨引擎复用，容量按集群算。
- **异构卡混部**：SLO 约束下把便宜的卡也用上。
- **GPU 故障探测**：硬件先于请求去死。

往后想做的三件事，读起来像 2025 年之后整条 serving 线的预告：分布式 KV 覆盖 P/D 聚合、请求迁移、跨实例复用；把 QoS / 优先级 / 公平性做到**请求级**多租户；用 roofline profiling 去守 SLO。Router、Mooncake、Elastic EP 后来各自兑现了其中几块。

## FAQ（比功能清单更有用）

**和 production-stack 什么关系？** AIBrix 是字节开源，对着大规模和云原生；当时已生产 6+ 个月。production-stack 由 UChicago LMCache 团队管，从零搭积木、欢迎实验。production-stack 的长处是内置 KV 向优化（传输、blending、路由），尤其长上下文、prefill 重。近期他们计划借用 AIBrix 组件。

**是社区项目吗？** 放进 vllm-project 组织就是为了打开协作。

**和 KServe / KubeAI？** AIBrix 更贴 vLLM：快加载、autoscaling、LoRA，可以按「只有这一台引擎」来做选择。通用 Serving 框架要伺候很多运行时，做不到这么贴。

读这一篇是为了记住：2025 年 2 月，社区同时有两套「引擎上面的盘子」。选哪一套是运维问题；学习上要把它们和后面的 Router（认得 KV 和 P/D 的那只网关）分开——控制面可以换，**记忆亲和**这件事不会消失。
