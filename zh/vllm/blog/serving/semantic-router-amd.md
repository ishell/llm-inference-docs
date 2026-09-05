---
source: https://vllm.ai/blog/2025-12-16-vllm-sr-amd
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# AMD × Semantic Router：GPU 上的控制面

英文对照：[en/vllm/blog/serving/semantic-router-amd.md](../../../../en/vllm/blog/serving/semantic-router-amd.md)  
原文：https://vllm.ai/blog/2025-12-16-vllm-sr-amd  
2025-12-16。署名 **The AMD and vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[signal-decision](semantic-router-signal.md)。分类核 LoRA：[modular](semantic-router-modular.md)。随后的现场 MoM：[mom-amd](semantic-router-mom-amd.md)。v0.1：[iris](semantic-router-iris.md)。后来 ROCm 变成一等 serve 路径：[athena](semantic-router-athena.md)。不要和引擎里的 [Router](router.md) 混。合作愿景文，kernel 数字少。Slack `#semantic-router`。

同目录还有：[halugate](halugate.md)、[themis](semantic-router-themis.md)、[session](semantic-router-session.md)、[mom](semantic-router-mom.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)、[vision](semantic-router-vision.md)。

AMD 是 vLLM 的长期伙伴（引擎走 ROCm）。这篇是下一层：Mixture-of-Models 的 **智能路由和治理**。栈从一只模型走到许多只，问题不再只是模型有多大，而是许多模型怎么聪明、安全地编排。VSR 被写成 **智能控制面**。

本地图（原文版权仍归原站；学习对照用）：

![amd 0](../../../../assets/vllm/blog/serving/semantic-router-amd/01-amd-0.png)

**Figure 1.** 三根柱：信号路由、跨实例智力、护栏。

三根战略柱：

1. **基于信号的路由**——keyword、域分类、语义相似、事实核验，给 Multi-LoRA 和多模型部署
2. **跨实例智力**——共享状态：集中的响应存储和 semantic cache
3. **护栏与治理**——PII、jailbreak、幻觉、对齐执行

## 从单模型到 Mixture-of-Models

典型企业栈：**router SLM** 负责分类 / 路由 / 执法；**多只 LLM** 和领域模型（代码、金融、医疗、法律）；**工具、RAG**、向量检索、业务系统。没有路由层，这就变成一张不透明的网。合作想把路由做成 **一等的、GPU 加速的基础设施组件**，不是服务之间粘的脚本。

## VSR 核心能力（页上列的）

### 1. Multi-LoRA 上的信号路由

- Keyword：快、确定性的模式
- 域分类：意图感知的 adapter 选择
- Embedding 相似：更细的语义路由
- 事实核验路由：高风险查询进专门管线（后来专篇：[halugate](halugate.md)）

### 2. 跨实例智力

- **Response API：** 集中存储，给有状态多轮
- **Semantic cache：** 跨实例向量相似，减 token

### 3. 企业级护栏

PII；jailbreak；关键域的幻觉检测；页上还印了 **Super Alignment**（系统往 AGI 级能力涨时仍对齐——他们的说法，不是测出来的承诺）。

## AMD GPU 上两条部署路

近期目标：能在 AMD GPU 上有效跑的生产级 VSR。两条互补。

![amd 1](../../../../assets/vllm/blog/serving/semantic-router-amd/02-amd-1.png)

**Figure 2.** 路径 1：ROCm 上的 vLLM 跑 router SLM + 多 LLM。路径 2：前门 ONNX Runtime。

### 路径 1：AMD GPU 上的 vLLM 推理

Router SLM：任务/意图分类、风险打分和安全闸、工具和工作流选择。LLM 和专家：通用协助加领域活。VSR 坐在上面当决策布——语义相似、业务 metadata、延迟约束、合规——**动态路由** 跨模型和端点。AMD GPU 被写成能在同一集群里跑 **router SLM + 多 LLM**，高 QPS、延迟稳，不只一次性 demo。具体 playground：[mom-amd](semantic-router-mom-amd.md)。

### 路径 2：轻量 ONNX 路由

不是每一跳都要完整推理栈。前门、超高频、延迟敏感的阶段：把 router SLM 导出成 **ONNX**，经 ONNX Runtime 在 AMD GPU 上跑，生成活再转给 vLLM 或其他 backend。瞄准：漏斗前端分类和分拣；大规模政策评测和离线实验；想 **在 AMD GPU 上标准化、又让模型提供方保持灵活** 的企业。Athena 后来把 ONNX + CK Flash Attention 落在这条路上。

## 下一阶段：从模型导演到智力裁判

早期 VSR：智能 **选模**（任务、成本、性能）。vLLM 引擎 = 地基（大模型稳着跑）。Semantic Router = 调度（派到对的能力）。页上说，系统往 AGI 级走时，这套说法不够——只谈引擎效率，不谈刹车、交通法、安全。

![amd 2](../../../../assets/vllm/blog/serving/semantic-router-amd/03-amd-2.png)

**Figure 3.** 控制面，不只调度。

和 AMD 一起，他们把演化改写成 **治理**：交通导演 → **Intelligence Control Plane**。不只在 AMD 硬件上拧吞吐和延迟。一层用职责而不是功能定义的 **宪制层**。

### 三道必须守住的控制生命线

![amd 3](../../../../assets/vllm/blog/serving/semantic-router-amd/04-amd-3.png)

**Figure 4.** 世界输出（动作）、世界输入（不可信数据）、长期状态。

1. **世界输出（动作）。** 危险的能力是 **执行**。工具调用、写库、打 API、改配置，发生前必须过一道 **外部检查点**。AMD GPU 被写成能把这些检查 **嵌在生产规模上**——风险、政策、日志——而不变成瓶颈。
2. **世界输入（输入）。** 外部输入默认不可信：网页、检索、上传、插件返回——提示注入、数据投毒、提权。数据进模型之前做 **边境检查**：分类器、清洗、核验当第一道，不当事后补丁。
3. **长期状态（记忆/状态）。** 最难修的失败：错答案被 **写进** 长期记忆、系统状态、或自动化工作流。谁能写、能写什么、怎么撤销、污染怎么隔离。持续核验和回滚当成一等事。

三道守住，Semantic Router 就不再只是选模器。页上印的问：**怎么把对齐从训练时的愿望，变成运行时的制度？**

## 长期愿景（列出来的倡议）

### 在 AMD GPU 上训下一代 encoder router

更远期：一只 **encoder-only** 的 router 模型，给语义路由、RAG、安全分类。ModernBERT 一类被写成强，但上下文、多语言、和长上下文注意力对齐仍有限。目标：一只给 VSR 和现代管线用的 **开放 encoder**，训练和部署硬件更多样。Athena 后来在相近弧上发出 `mmbert-embed-32k-2d-matryoshka`。

### AMD 基础设施上的社区公测

每次 VSR 大版本配一套 AMD 赞助的 **公测**，对社区免费：验证路由 / cache / 安全；在 AMD GPU 上手摸；早反馈。后来真有的 playground：[play.vllm-semantic-router.com](https://play.vllm-semantic-router.com)。

### AMD GPU 驱动的 CI/CD 和端到端试验台

Router SLM、LLM、领域模型、检索、工具一起跑在 AMD GPU 集群上；多域、多风险数据集当流量回放；每次改动过自动评测：路由/政策回归、新旧策略 A/B、延迟/成本/伸缩压测、幻觉和合规专项。

印出来的目标：

> 每次 VSR 发版带着一份可复现、GPU 驱动的评测报告，不只一份 changelog。

GPU 当 **路由基础设施自己的核验引擎**，不只给模型 serving。

### AMD 撑腰的 Mixture-of-Models playground

AMD GPU 上的在线 MoM playground：试路由策略和拓扑；看叫了哪只模型、何时检索、何时加检查或回退；比 **质量、延迟、成本**。给厂商和工具作者：一套 **中立** 的、在真实路由和治理约束下的 AMD GPU 试验环境。

## 这场合作要紧在哪（他们的共同野心）

- AMD 平台上智能、GPU 加速路由的 **参考架构**：vLLM 推理路径、ONNX 轻量 router 路径、多模型协调和安全。
- 路由当 **可信基础设施**：GPU CI/CD 和端到端评测、认得幻觉和风险的政策、在线学习和自适应。
- 一套 **活得久的 AMD GPU MoM playground**，想法、模型、政策能公开试。

硬件 = 执行（以及后来的核验）层。VSR = 控制面。路线图条目（幻觉检测、在线学习、多模型编排）为这套使命服务。他们对齐「靠架构」，按他们自己的写法。

## 致谢 / 加入

AMD：Andy Luo、Haichen Zhang、AMD AIG 团队。vLLM SR：Xunzhuo Liu、Huamin Chen、Chen Wang、Yue Zhu 和 OSS 团队。

页上印的联系：Haichen Zhang（`haichzha@amd.com`）、Xunzhuo Liu（`xunzhuo@vllm-semantic-router.ai`）。

- [AMD ROCm Software](https://www.amd.com/en/products/software/rocm.html)
- GitHub：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- 文档：[vllm-semantic-router.com](https://vllm-semantic-router.com)
- Slack：[vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN) 的 `#semantic-router`
