---
source: https://vllm.ai/blog/2025-12-16-vllm-sr-amd
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# AMD × Semantic Router：GPU 上的控制面

英文对照：[en/vllm/blog/serving/semantic-router-amd.md](../../../../en/vllm/blog/serving/semantic-router-amd.md)  
原文：https://vllm.ai/blog/2025-12-16-vllm-sr-amd  
2025-12-16。署名 **The AMD and vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。后来的 MI300X 现场池：[mom-amd](semantic-router-mom-amd.md)。LoRA 核：[modular](semantic-router-modular.md)。再往后：[athena](semantic-router-athena.md)、[themis](semantic-router-themis.md)、[mom](semantic-router-mom.md)。不要和引擎里的 [Router](router.md) 混。合作愿景文，**kernel 数字少**。公测 / GPU CI / playground 是这篇上的路线图，不是已交付的 SLA。

同目录还有：[session](semantic-router-session.md)、[vision](semantic-router-vision.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)。

AMD 是 vLLM 的长期伙伴：先把推理引擎搬上 GPU 和 ROCm，现在做下一层——**Mixture-of-Models（MoM）的智能路由和治理**。Serving 离开单模世界，页上的问题不再只是「模型有多大」，而是许多模型怎么 **聪明、安全地** 编排。VSR 被写成 **智能控制面**：语义路由、安全策略、系统放大时仍能被信任。

本地图（原文版权仍归原站；学习对照用）：

![amd 0](../../../../assets/vllm/blog/serving/semantic-router-amd/01-amd-0.png)

**Figure 1.** 三根柱：信号路由、跨实例智力、护栏。

三根战略柱：

1. **Signal-based routing**：keyword、domain classification、semantic similarity、fact-checking——给 Multi-LoRA 和多模部署
2. **Cross-instance intelligence**：vLLM 实例之间共享状态——集中 Response 存储和 semantic cache
3. **Guardrails & governance**：PII、jailbreak、幻觉、alignment 执法

## 从单模到 Mixture-of-Models

页上的典型企业栈：

- **Router SLMs**：分类、路由、执法
- **多只 LLM** 和领域模型（代码、金融、医疗、法律）
- **Tools、RAG**、向量检索、业务系统

没有路由层，这就是一张不透明的网。合作目标：路由当 **一等、GPU 加速的基础设施**，不是服务之间粘的脚本。

## VSR 核心能力

### 1. Multi-LoRA 的信号路由

- **Keyword-based**：快、确定的模式匹配
- **Domain classification**：训练好的分类器做 intent-aware 的 adapter 选择
- **Embedding-based semantic similarity**：按语义理解路由
- **Fact-checking / verification routing**：高风险查询送到专门的校验管线

### 2. 跨实例智力

- **Response API**：集中存响应，给有状态多轮
- **Semantic Cache**：跨实例向量相似，减 token

### 3. 企业级护栏

从单轮到多轮：

- **PII detection**
- **Jailbreak prevention**
- **Hallucination detection**
- **Super Alignment**：原文用语，系统朝 AGI 级能力放大时仍对齐——这篇里的治理口号，不是测过的评测

## AMD GPU 上的两条部署

他们写的近目标：一份能在 AMD GPU 上 **跑得有效率** 的生产级 VSR。两条互补路径。

![amd 1](../../../../assets/vllm/blog/serving/semantic-router-amd/02-amd-1.png)

**Figure 2.** 路径 1：vLLM + router SLM + 多 LLM。路径 2：前门用 ONNX Runtime。

### 路径 1：AMD GPU 上的 vLLM 推理

vLLM 引擎在 AMD GPU 上跑：

**Router SLMs**：任务/意图分类、风险打分和安全闸、工具和工作流选择。

**LLM 和专家**：通用助手和领域任务（金融、法律、代码、医疗）。

VSR 坐在上面当决策织物：语义相似、业务元数据、延迟约束、合规 → 跨模型和端点的 **动态路由**。主张：AMD GPU 的吞吐和显存够在同一集群里跑 **router SLM + 多 LLM**，高 QPS、延迟稳——不只是 demo。这篇没有 kernel 表；现场数字在 [mom-amd](semantic-router-mom-amd.md)。

### 路径 2：轻量 ONNX 路由

不是每一跳都要完整生成栈。超高频、吃延迟的 **前门**：

- 把 router SLM 导出成 **ONNX**
- 经 ONNX Runtime 跑在 AMD GPU 上
- 复杂生成转给 vLLM 或其他 backend LLM

为这些场景写的：

- 漏斗前端的分类和分流
- 大规模策略评测和离线实验
- 想 **把 GPU 统一到 AMD、模型供应商仍可换** 的企业

## 从模型调度员到智力裁判

早期 VSR 目标很实务：**智能选模**——任务类型、成本、性能。vLLM 引擎 = 大模型跑得稳。Semantic Router = 调度器。

![amd 2](../../../../assets/vllm/blog/serving/semantic-router-amd/03-amd-2.png)

**Figure 3.** 有了引擎和调度器仍不够：还有动作、不信任的输入、长期状态。

页上的转向：系统朝 AGI 级能力走时，只谈引擎效率、不谈刹车和交通法，不完整。**真正的难处是模型更强时仍能控制。** 和 AMD 一起，他们把 Semantic Router 改口成 **治理**：从流量导演变成 **Intelligence Control Plane**。一层按责任定义的 **constitution**，不是功能清单。

### 三道必须守住的控制生命线

![amd 3](../../../../assets/vllm/blog/serving/semantic-router-amd/04-amd-3.png)

**Figure 4.** 闸在 world output（动作）、world input（不信任数据）、长期状态上。

**1. World output（动作）**

危险的能力是 **执行**。Tool call、写库、调 API、改配置，必须先过一道 **外部检查点**。主张：AMD GPU 能把这些检查点 **嵌进生产规模**——风险、策略、日志——还不至于变成瓶颈。没有贴延迟数字。

**2. World input（输入）**

外部输入默认不信任：网页、检索、上传、插件返回——prompt injection、数据投毒、提权。VSR 当模型前的 **边境检查**：分类器、清洗、校验当第一道，不当事后补丁。

**3. Long-term state（记忆 / 状态）**

最难修的失败：**错答案写进记忆、系统状态或自动工作流**。他们点名的一等事项：谁能写、能写什么、怎么撤销、怎么隔离污染。持续校验和回滚，是这篇上 GPU 托底的愿望。

他们抛出的终极问题：怎么把 alignment 从 **训练时的愿望** 收成 **运行时的制度**。

## 长期愿景（这篇上的路线图）

### 在 AMD GPU 上训下一代 encoder 路由模型

更远：一只 **encoder-only** 路由模型，跑在 AMD GPU 上，做语义路由、RAG、安全分类。他们写 ModernBERT 一类 encoder 在上下文长度、多语覆盖、长上下文 attention 对齐上仍有限。目标：一只能插进 VSR 的 **开放 encoder**，再加硬件多样的训练。这里没有已发布的 checkpoint。

### AMD 基础设施上的社区公测

每次 VSR 大版本配一套 **公测环境**，AMD 赞助、社区免费：验证路由 / cache / 安全，上手 GPU，发版前收反馈。路线图条目。

### AMD GPU 驱动的 CI/CD 和端到端试验台

长远：AMD GPU 托住 VSR **怎么构建、校验、发货**。GPU 托底的 CI/CD 和 E2E：

- Router SLM、LLM、领域模型、检索、工具一起跑在 AMD GPU 集群
- 多域、多风险数据集当流量回放
- 每次改动自动评测：路由/策略回归、策略 A/B、延迟/成本/规模压测、幻觉和合规套件

页上的目标句：

> Every VSR release comes with a reproducible, GPU-driven evaluation report, not just a changelog.

AMD GPU 当 **路由基础设施自己的校验引擎**，不只是 serving 箱子。

### AMD 托底的 Mixture-of-Models playground

AMD GPU 上的在线 MoM playground（后来他们交出一只活的；见 [mom-amd](semantic-router-mom-amd.md)）：试路由策略和拓扑；看调了哪只模型、何时检索、何时检查或回退；比质量 / 延迟 / 成本。对供应商：一套在路由和治理约束下的 **中立** 试验场。

## 为什么这次合作（他们的野心）

超出「这只模型能不能在这块 GPU 上跑」：

- AMD 上 **GPU 加速路由的参考架构**：vLLM 推理路径、ONNX 轻量路由路径、多模协调和安全
- 路由当 **被信任的基础设施**：GPU CI/CD 和 E2E 评测、幻觉感知和风险感知策略、在线学习和自适应
- 一只 **长期存在、AMD GPU 托底的 MoM playground**，想法、模型、策略公开演化

硬件 = 执行层。VSR = 控制面。Alignment「不靠希望，靠 **架构**」。把它当这篇的论题，不当测过的结果。

## 致谢

- **AMD**：Andy Luo、Haichen Zhang，以及 AMD AIG Teams
- **vLLM SR**：Xunzhuo Liu、Huamin Chen、Chen Wang、Yue Zhu，以及 vLLM Semantic Router OSS team

联系人：Haichen Zhang（`haichzha@amd.com`）、Xunzhuo Liu（`xunzhuo@vllm-semantic-router.ai`）。

资源：[AMD ROCm](https://www.amd.com/en/products/software/rocm.html)、[GitHub](https://github.com/vllm-project/semantic-router)、[文档](https://vllm-semantic-router.com)。Slack：vLLM Slack 的 `#semantic-router`（[频道](https://vllm-dev.slack.com/archives/C09CTGF8KCN)）。
