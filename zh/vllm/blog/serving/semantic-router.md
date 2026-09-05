---
source: https://vllm.ai/blog/2025-09-11-semantic-router
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Semantic Router：按意图决定走不走推理

英文对照：[en/vllm/blog/serving/semantic-router.md](../../../../en/vllm/blog/serving/semantic-router.md)  
原文：https://vllm.ai/blog/2025-09-11-semantic-router  
2025-09-11。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。这是立项文。v0.1 架构翻新：[Iris](semantic-router-iris.md)。这是 **控制面**：按意图决定走哪只模型、开不开 CoT。不是引擎里那只 Rust P/D 负载均衡——那是 [Router](router.md)。名字都叫 router，职责不同。试验数字当演示，不是你集群的 SLA。

本地图（原文版权仍归原站；学习对照用）：

![request](../../../../assets/vllm/blog/serving/semantic-router/01-request.png)

## 行业：推理不是越多越好

原文把过去一年写成：混合推理和自动路由，把辩论从「参数堆多高」拧到每 token 的效率、时延、算力该用在哪。

GPT-5 被拿来当例子：出彩的不是参数，是路由策略和带配额的推理：

- 轻查询 → 轻路径。「天为什么是蓝的」不必点燃昂贵推理。
- 复杂 / 高价值查询 → 开推理的模型。法律分析、财务规划这类多步任务，送到 Chain-of-Thought。

原则：按任务分配算力。每个推理 token 都该换回价值，而不是被消耗掉就算。

同类念头也在别处出现：

- Anthropic Claude 3.7 / 4：区分「快想」和「慢想」。
- Google Gemini 2.5：显式 *thinking budget*，企业可以给推理深度设顶。
- 阿里 Qwen3：指令驱动，在推理 / 非推理模式之间切。
- DeepSeek v3.1：对话和推理收在一只双模模型里。

趋势：未来的推理系统靠**选择性**和**判断**，不只靠模型尺寸。

## 研究：vLLM Semantic Router

vLLM 能把 GPU 喂饱，却没有「这句要不要推理」这一层语义决策。开发者面对二选一：

- 推理全开 → 准确上去，成本也上去。
- 推理全关 → 成本下来，复杂题掉点。

Semantic Router 用语义分类填这个缺口：该准的走准的路径，不必推理的走快路径。

![architecture](../../../../assets/vllm/blog/serving/semantic-router/02-architecture.png)

### 架构：四根柱

1. **Semantic Classification。** ModernBERT——当时是路由进程里一只轻量、独立的分类器——决定走哪条路。
2. **Smart Routing。** 简单查询 → 快路径；复杂查询 → Chain-of-Thought。
3. **High-Performance Engine。** Rust + Hugging Face Candle，高并发、零拷贝推理。
4. **Cloud-Native。** Kubernetes 和 Envoy，经 `ext_proc` 插件开箱能接。

试验里这套设计交出：

- 准确大约 **+10%**
- 时延大约 **−50%**
- token 大约 **−50%**

商科、经济类场景，准确提升可以超过 **20%**。当演示。

## 执行上的两道约束：预算和工具

- **Reasoning budget。** 推理不设顶，冷启动时延和资源会涨。没有动态闸门：简单查询可能把 token 花光，关键查询反而推不深。SLO 要盯 TTFT、p95；推理中途也可能要改。
- **Tool calling。** 工具目录膨胀、工具输出变长，准确会掉。路由侧要先滤工具，目录保持紧。

Classifier 当时跑在路由进程里，**还不是** vLLM 上的 embedding 服务。后面那一节把这扇门留着。

## 项目背景

开源里凑起来的：

- 2025 年初由 [Dr. Chen Huamin](https://www.linkedin.com/in/huaminchen)（Red Hat）提出
- [Xunzhuo Liu](https://www.linkedin.com/in/bitliu)（Tencent）继续往前推
- [Dr. Wang Chen](https://www.linkedin.com/in/chenw615)（IBM Research）和 Dr. Chen Huamin 要在 [KubeCon North America 2025](https://kccncna2025.sched.com/event/27FaI/intelligent-llm-routing-a-new-paradigm-for-multi-model-ai-orchestration-in-kubernetes-chen-wang-ibm-research-huamin-chen-red-hat?iframe=no&w=100%&sidebar=yes&bg=no) 讲

目标：给开源 LLM 做推理加速——语义感知路由、高效的模型切换、企业好部署（Kubernetes & Envoy）。

仓库：[GitHub](https://github.com/vllm-project/semantic-router)。当时的焦点：[Work Group](https://vllm-semantic-router.com/community/work-groups) 和计划中的 [v0.1 Roadmap](https://vllm-semantic-router.com/roadmap/v0.1)。落地形态见 [Iris](semantic-router-iris.md)。

## 集成与下一步：Embeddings 和可插拔

ModernBERT 当时内嵌在路由里做分类，**尚未**由 vLLM serving。后续打算让分类器——以及别的 embedding 模型——可插拔：接到 vLLM 托管的模型，或外部 embedding 服务。语义 cache 和推理定制都指望这层。

## 路线图：v0.1 milestone

[v0.1 milestone](https://github.com/vllm-project/semantic-router/milestone/1) 当时列的能力：

- **Core：** 基于 ExtProc 的模块化；跨 backend 的 semantic cache；多因子路由逻辑
- **Benchmarking：** CLI、性能测试套件、reasoning-mode 评估
- **Networking：** 更深地接 Envoy、GIE、llm-d 网关
- **Observability & UX：** Admin 面板、路由策略可视化、开发者 quickstart、policy cookbook

## 趋势：Just-in-Time Inference

场子从「能不能跑推理」长到「怎样更聪明地跑」。

- GPT-5 用商业价值引导推理深度。
- vLLM Semantic Router 把同类能力交给开源。

往前看：推理策略当场改、不必人手拨开关的系统，会在效率、时延、可持续性上领先。原文把这叫做 just-in-time inference。

## 一句话

- GPT-5：企业路由，把推理做聪明
- vLLM Semantic Router：技术优先的路由，给开源 LLM
- 边缘的未来：感知上下文、少用算力、还要无缝
