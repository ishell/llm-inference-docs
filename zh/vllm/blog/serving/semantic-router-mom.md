---
source: https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Mixture-of-Models：从选模型到造系统

英文对照：[en/vllm/blog/serving/semantic-router-mom.md](../../../../en/vllm/blog/serving/semantic-router-mom.md)  
原文：https://vllm.ai/blog/2026-07-21-vllm-sr-new-chapter-mom  
2026-07-21。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[semantic-router-signal](semantic-router-signal.md)。v0.1：[iris](semantic-router-iris.md)。v0.2：[athena](semantic-router-athena.md)。v0.3：[themis](semantic-router-themis.md)。SAAR：[session](semantic-router-session.md)。协作配方：[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)。AMD 现场池：[mom-amd](semantic-router-mom-amd.md)。不要和引擎里的 [Router](router.md) 混。发布页社区数字（5k stars / 150+ 贡献者 / HF 累计 30 万下载）和 1,734 commit 快照是**他们的**。

同目录还有：[modular](semantic-router-modular.md)、[amd](semantic-router-amd.md)、[vision](semantic-router-vision.md)。

多数 AI 应用钉在一只模型端点上。模型、设备、部署约束散开；没有一只 checkpoint 能伺候所有请求和环境。系统题是：专门模型怎么被协调、评测、从同一个界面伺候出去。他们把这套做法叫 **Mixture-of-Models**。

公开立项不到一年，Semantic Router 自称 **5,000** stars、**150+** 贡献者、Hugging Face 累计下载 **超过 30 万**。穿过 **Iris、Athena、Themis**，边界从选一只模型，走到治理多模型推理，再走到跨 session 保住状态。这些发版是他们说从第 0 天就想要的 MoM 架构地基。

下一步：从模型之间做路由，走到用它们造靠得住的模型 **系统**。同一份 versioned contract 下面，独立模型、策略、偏好、执行路径收成能训练、评测、导出、导入、部署、用一个界面调用的东西。目标：Semantic Router 当 Mixture-of-Models 的训练、评测、推理引擎。

**MoM ≠ MoE。** MoE 在一次前向里按 token 选专家。MoM 在请求级编排不同架构——甚至不同箱子上的模型。一只 MoE checkpoint 本身可以当 MoM 的一个零件。

本地图（原文版权仍归原站；学习对照用）：

![hero](../../../../assets/vllm/blog/serving/semantic-router-mom/01-hero.png)

**Figure 1.** Mixture-of-Models 把异构模型篮子收成一次模型体验。

## vLLM-SR 怎么走到这儿

[立项文](semantic-router.md) 问：简单请求和难请求为什么要同一份推理预算？轻分类器用固定域标签在快路径和推理路径之间选。

生产流量很快把极限露出来。单靠 domain 装不下隐私、安全、上下文、语言、modality、工具、偏好、延迟、授权。静态标签也解释不了：端点便宜但过载、能力强但在远处、或 agent session 半路上换进去不安全。

他们把分类器层重建成模块化模型支持、共享 LoRA、Rust/Candle 推理、Go 集成，再用 Signal–Decision 换掉固定分类：证据和政策、执行分开。这成了后面三版的脊柱。细节见 [semantic-router-signal](semantic-router-signal.md)，LoRA 核见 [modular](semantic-router-modular.md)。

| Milestone | When | What changed |
| --- | --- | --- |
| Incubation | Apr 2025 | 早期语义路由原型；MoM 当长期系统目标 |
| Initial release | Sep 2025 | 按意图在快路径和推理路径之间选 |
| v0.1 Iris | Jan 2026 | 信号、决策、按路由作用域的插件，换掉固定分类 |
| v0.2 Athena | Mar 2026 | 选模、记忆、RAG、长上下文、多模态——推理控制系统 |
| v0.3 Themis | Jun 2026 | 有状态路由、projection、replay、协议、session 连续、一份生产配置合同 |
| Fusion and Micro-Agent | Jun 2026 | 路由开始选协作形态，不只选单只模型 |

![evolution](../../../../assets/vllm/blog/serving/semantic-router-mom/02-evolution.png)

**Figure 2.** 控制单位在变：模型、决策、系统、session，再到完整模型生命周期。

[Iris](semantic-router-iris.md) 让路由可组合。Domain、keyword、embedding、factuality、feedback、preference 信号喂给显式决策；安全、PII、cache、幻觉检测、工具选择变成按路由作用域的行为。Iris 也交出 MoM 模型家族，把 vLLM-SR 写成 “System Level Intelligence for Mixture-of-Models”。

[Athena](semantic-router-athena.md) 加上一等选模、记忆和 RAG、多语言多模态栈、ROCm 加速、可运营 dashboard。围着多模型推理的控制系统，不只是 vLLM 前面一只分类器。

[Themis](semantic-router-themis.md) 把这收成可运营合同：

> 信号变成 projection，projection 喂决策，决策选算法，算法选模型。

Themis 加上 [session-aware agentic routing](semantic-router-session.md)、可 replay 的迹、更硬的协议、operator console，以及 AMD ROCm、NVIDIA CUDA、Intel OpenVINO、CPU 上的 runtime 路径。一条路由变得可解释：证据、策略、算法、物理模型。

### 从 Signal–Decision 到 Workload–Router–Pool

runtime 背后有两篇论文。

[白皮书 *Signal Driven Decision Routing for Mixture-of-Modality Models*](https://vllm-sr.ai/white-paper/) 把神经证据和符号策略分开写死。快启发式和学来的分类器，把 prompt、上下文、身份、安全、modality 收成结构化信号向量；Boolean 引擎再把它们组成可审计策略。typed 神经–符号 DSL 先 parse、校验，再编译成可部署配置。发文时：十三种信号、十三种选模算法，按决策的插件覆盖 cache、RAG、记忆、安全、provider 处理、响应校验。

[愿景文 *The Workload–Router–Pool Architecture for LLM Inference Optimization*](https://vllm-sr.ai/vision-paper/) 把框拉大。三个变量必须一起设计：

- **Workload：** chat 或 agent、单轮或多轮、暖或冷、prefill 重或 decode 重
- **Router：** 静态语义策略、在线反馈或 bandit、基于 RL 的选择、质量感知 cascade
- **Pool：** 同构或异构加速器、prefill/decode 拓扑、模型放置、KV-cache 管理

不能各自优化。工作负载形状改哪套路由策略有用；路由策略改需要的 pool 规模和拓扑；pool 状态改哪条路由高效。安全和隐私横切三面；成本、质量、延迟、能量划出前沿。论文把研究映射进 3 × 3 WRP 矩阵，列出二十一个开口。

![research arc](../../../../assets/vllm/blog/serving/semantic-router-mom/03-research-arc.png)

**Figure 3.** 白皮书：可编程路由引擎。愿景文：工作负载和物理 pool 共设计。

runtime 当时已经越过单模选择。[Fusion](semantic-router-fusion.md)、ReMoM、Confidence、Ratings、有界 Workflows，让一条请求能调用受控的模型协作。[Micro-Agent](semantic-router-micro-agent.md) 写过：客户端调一个模型名，serving 层选配方、扇出到 worker、校验或合成，交回一条普通响应。

| First chapter | New chapter |
| --- | --- |
| 给请求找路 | 造一套模型系统 |
| 选模型或能力路径 | 训练、评测、执行整部 MoM |
| 配 runtime 策略 | 打一份可移植、带版本的模型产物 |
| 优化一次路由决策 | 在质量、成本、延迟、安全、能量上优化系统智力 |
| 用一个 API 把 backend 藏起来 | 让整套多模型系统表现得像一只模型 |

路由仍是机制：MoM 怎么分配工作、套策略、协调零件。**模型系统才是产品。**

## 为什么模型边界必须挪

今天的栈沿四条轴碎开：

- **模型。** 闭源前沿、开源通用、领域专家、紧凑本地、校验器、多模态。没有谁能同时赢质量、成本、延迟、信任、隐私、领域契合。
- **算力。** GPU、CPU、专用加速器、边缘、云、私有集群，在内存、kernel、可用性、价格、能耗上不同。选模型和放哪里正在变成同一件事。
- **位置。** 云、数据中心、边缘。隐私或驻留可能把更强的远程模型挡掉；本地负载仍可能要按需的云专家。
- **偏好。** 没有宇宙级「最好」。产品和用户在精度、延迟、价格、隐私、安全、文风、多模态上做不同买卖。这些选择该直接塑形执行。

今天每只应用自己把碎片粘回去。

![fragmentation before mom](../../../../assets/vllm/blog/serving/semantic-router-mom/04-fragmentation-before-mom.png)

**Figure 4.** MoM 之前，碎掉的智力变成应用侧的路由胶水。

Mixture-of-Models 把这责任收到一条模型边界后面。在那条边界上，**智能分配** 是模型的一部分：哪些模型有资格、执行能跑在哪、要不要协作、硬约束怎么满足。

能量让分配和效率分不开。硬件和推理引擎改善 **供给** 侧（每瓦每美元多少 token）。分配层管 **需求**：哪些工作配得上这些 token，哪只模型或哪种协作能在质量、延迟、能量预算里交出来。

应用选一个带版本的模型身份，收回一条可归因的响应。物理实现仍可以跨开源和闭源、云和边缘、不同代加速器。碎片还在，但变成系统内部，不再漏进每只应用。

![fragmentation after mom](../../../../assets/vllm/blog/serving/semantic-router-mom/05-fragmentation-after-mom.png)

**Figure 5.** 有了 MoM，同样的碎片变成一只模型的内部实现。

## 他们说的 Mixture-of-Models 是什么

**Mixture-of-Models** 是一份带版本的复合模型：引擎按偏好、在资源边界里，穿过独立模型和 operator 实现每一条请求。用一个模型界面交给用户，交回一条可归因的结果。

多上游 gateway 可以转发流量，不必拥有系统质量。MoM 拥有目标、评测合同、可复现的组成、以及执行它的 runtime。

| | Conventional model | Mixture-of-Models |
| --- | --- | --- |
| 智力单位 | 一只 checkpoint | 一套被治理的模型系统 |
| 专门化 | 主要写进权重 | 跨独立专家组合 |
| 执行 | 一条生成路径 | 选择、cascade、校验、fusion、或 workflow |
| 优化目标 | 单模质量和效率 | 质量、成本、延迟、安全、隐私、能量上的系统前沿 |
| 部署边界 | 一个 runtime | 云、数据中心、边缘 |
| 用户合同 | 一个模型身份 | 一个模型身份 |

![execution topologies](../../../../assets/vllm/blog/serving/semantic-router-mom/06-execution-topologies.png)

**Figure 6.** 选择只是一种 MoM 拓扑。Cascade、并行 fusion、有界 workflow 共用同一条模型边界。

可移植的 MoM 要的不只是权重和配置：零件清单、能力元数据、路由和协作配方、策略、偏好、评测套件、runtime 约束、出处、版本史。

开源 checkpoint 可以跟着产物走；闭源模型仍是带鉴权的外部引用，带着显式能力和策略合同。导出 MoM 不会让专有 checkpoint 变得可移植。它让 **模型系统** 可复现。

### 把偏好收成模型

偏好变成具体，是在被发布成模型身份的时候。一个 MoM 家族可以给几个工作点：

| Model identity | Contract |
| --- | --- |
| `vllm-sr/mom-v1-flash` | 尽量压预期延迟 |
| `vllm-sr/mom-v1-light` | 在质量地板之上尽量压成本 |
| `vllm-sr/mom-v1-ultra` | 在声明预算里尽量抬质量 |
| `vllm-sr/mom-v1-halu` | 要 grounding 检查，失败则 fail-closed |
| `vllm-sr/mom-v1-secu` | 执行前强制 jailbreak 和 PII 策略 |

每个名字是一份带版本的模型合同，不是 router preset。应用选它要的行为；vLLM-SR 选出并协调能交出来的模型，同时保住硬的隐私、驻留、授权、安全约束。

对应用仍是普通模型调用：

```json
{
  "model": "vllm-sr/mom-v1-ultra",
  "messages": [
    {"role": "user", "content": "Review this design and identify its weakest assumption."}
  ]
}
```

这个身份可能选一只模型、沿 cascade 升级、比较并行答案、要求 grounding、或跑有界 workflow——外部界面、版本、响应合同不变。

![preference models](../../../../assets/vllm/blog/serving/semantic-router-mom/07-preference-models.png)

**Figure 7.** 偏好发布成有界、带版本的模型合同——不是藏在应用侧的路由预设。

四层平面把所有权分开：

| Plane | 管什么 | vLLM-SR 里已有的地基 | 下一步 |
| --- | --- | --- | --- |
| **Artifact** | 零件、能力、目标、策略、评测合同、出处 | 规范 config、模型引用、DSL、带版本策略 | 可移植的 MoM 导入/导出规格 |
| **Learning** | router 自管模型、偏好、结果、配方改进 | 训练栈、Router Learning、replay、outcome API | 联合训练和系统级发版闸 |
| **Execution** | 信号、projection、决策、选择器、looper、插件 | Signal–Decision runtime、Fusion、ReMoM、Workflows、安全和记忆 | 一只认生命周期的 MoM 引擎 |
| **Physical** | provider、模型池、加速器、局部性、cache 和能量状态 | vLLM backend、云 provider、ROCm、CUDA、OpenVINO、CPU | 云、数据中心、边缘、本地设备上的可移植放置 |

![four planes](../../../../assets/vllm/blog/serving/semantic-router-mom/08-four-planes.png)

**Figure 8.** 完整 MoM 跨四层：产物、学习、执行、物理实现。

部署必须把逻辑需求映射到环境里有的模型和机器。四个对象：

1. **bundle** 钉死界面、图、策略、行为变体、边界、不可变语义资产。
2. **binding** 把逻辑零件映射到有资格的部署，不改模型的决策语义。
3. **resolution lock** 冻结组成修订、runtime、镜像、加速器、provider 观察。
4. **run record** 把每一次决策、调用、约束检查、成本、结果，归因到造出它的 bundle、binding、lock。

![artifact resolution lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/09-artifact-resolution-lifecycle.png)

**Figure 9.** 一个稳定的模型身份，从可移植合同到可归因的一次 run。

同一只 `mom-v1-ultra` 可以 bind 到 ROCm、CUDA、私有 CPU 或 NPU 节点、或混合部署，而不承诺不透明 provider 吐出一模一样的输出。它保住控制语义、把替换摊开，让 serving 和评测看见同一套已解析系统。

## vLLM-SR 当 MoM 引擎

训练、评测、推理必须共享一份合同；否则研究、基准、生产会漂成三套系统。

### 训练的是分配，不只是权重

MoM 训练覆盖 router 自管的 embedding、信号编码器、偏好和安全模型、选择器。也学分配和协作：哪条路径贴合工作负载和预算、cascade 该何时停、面板该怎么判或合成、agent session 该何时换模。组成可能独立或闭源，所以进步不必对它们全部反传；策略、阈值、池、prompt、合同、拓扑可以从迹和结果里优化。

目标是质量、延迟、成本、安全、隐私、可靠、局部性、能量上的一条前沿。Replay 和结果把生产经验喂回离线训练，热路径不许悄悄改写策略。

### 把 MoM 当一只模型来评

评测必须端到端给这个模型身份打分；backend 基准是输入，不是结果。带版本的 scorecard 该量：路由 regret、协作增益、恢复、session 连续、尾延迟、成本、安全、隐私、能量。该压：provider 失败、设备丢失、模型分歧、工作负载漂移、偏好变化。每个声明的工作点也要自己的测试：`flash` 看延迟–质量前沿，`light` 看质量地板，`ultra` 看预算。

科学测试比「多打几次分更高」更严。在 **匹配的主动算力** 下，条件系统能不能比最好的固定模型更好地用上互补的长处和失败模式？没有这个对照，MoM 可以把蛮力扩规模藏在一张聪明图后面。评测必须把调用次数、token、成本、延迟、能量和质量一起报——组合帮不上忙的时候也要公开。

![matched compute evaluation](../../../../assets/vllm/blog/serving/semantic-router-mom/10-matched-compute-evaluation.png)

**Figure 10.** 组合增益只在匹配主动算力下有意义；质量要和调用、token、成本、延迟、能量一起报。

### 推理时执行智力

推理时引擎先判：一只模型够不够。它可以选本地专家、保住暖 session、沿置信 cascade 升级、要求检索或校验、跑 Fusion 面板、或执行有界 workflow。runtime 拥有预算、拓扑、回退、迹、响应合同；应用仍做一次普通模型调用。

![mom lifecycle](../../../../assets/vllm/blog/serving/semantic-router-mom/11-mom-lifecycle.png)

**Figure 11.** 闭环生命周期：训练分配策略、评整套系统、执行、把结果收成下一版通过校验的版本。

## 一只能挪的模型

目标：完整 MoM 能被 **建造、导出、导入、定版本、评测、部署、当作统一模型调用**。逻辑规格编译成不可变 bundle，bind 到环境，解析出具体部署，serving 和评测仍用同一个身份。

产物该能跑在开发机、私有集群、云舰队、边缘上，同时物理实现在变。专家可以解析成可接受的本地 checkpoint 或托管端点；加速器 runtime 可以替换。隐私让远程专家不可用时，引擎走声明过的回退或 abstention。binding 不能悄悄改图、放松护栏、把面板改成 cascade——那些改动要新模型版本。

「在任何硬件上跑」是架构要求，不是声称每个零件今天都可移植。项目已经有 ROCm、CUDA、OpenVINO、CPU 路径。下一步：硬件能力和放置写进 MoM 合同，让引擎把系统映射到手头有的东西上。

> **One model identity. Many models. Any hardware.**

![portable realizations](../../../../assets/vllm/blog/serving/semantic-router-mom/12-portable-realizations.png)

**Figure 12.** 一个逻辑模型身份，可以在开发机、数据中心、云、边缘硬件上实现。

如果应用必须知道每只子模型归哪家 provider、跑在哪块设备、走哪张回退图，抽象就漏了。

## 现在改什么

四块连着的工作：

1. **定义可移植的 MoM 规格。** 把零件、目标、策略、偏好、评测、约束、执行语义打成一份带版本的产物。
2. **合上训练–评测–推理环。** 从评测和 replay 改进模型和配方，再经可审查、可回滚的发版送出去。
3. **造异构 runtime。** 用硬件、局部性、能量、数据边界当输入，把一只 MoM 映射过云、数据中心、边缘。
4. **让模型界面无聊。** 让 MoM 像单模一样好导入、部署、调用。

![next stage roadmap](../../../../assets/vllm/blog/serving/semantic-router-mom/13-next-stage-roadmap.png)

**Figure 13.** 四条工作流：可移植规格、闭环、异构 runtime、一个模型 API。

这是研究纲领：独立模型该怎么专门化、竞争、校验、协作；怎么量这套系统；一份模型合同怎么在设备和环境之间活下来。他们写的使命：

> **Advancing the science of intelligence across models, devices, and environments.**

他们会研究组合何时造出单只 checkpoint 没有的能力，把放置和能量当成智力的一部分，把同一份模型合同从边缘带到云、从研究带到生产。

## 一起造

造 Mixture-of-Models 要的不只是路由。工作跨模型训练、评测、serving 系统、硬件、生产运营。

Iris、Athena、Themis 变好，是因为贡献者带来真实负载、加 backend、训练模型、发基准、找到失败、争更好的界面。MoM 要同样宽的工作：学来的分配、偏好优化、模型合作、能量感知推理、可移植产物、公开评测、异构 runtime。

如果你做这些：造一个工作点、加一个 runtime、测一套协作配方、或公开组合失败的案例。假设在公开处被压过，MoM 会更硬。

### 致谢

他们感谢 [Xunzhuo Liu](https://www.linkedin.com/in/bitliu)、[Huamin Chen](https://www.linkedin.com/in/huaminchen)、[Bowei He](https://www.linkedin.com/in/bowei-he-8a9450199/)、[Yankai Chen](https://www.linkedin.com/in/yankai-chen-923001154/)、[Fuyuan Lyu](https://www.linkedin.com/in/fuyuan-lyu-560756167/)、[Steve Liu](https://ca.linkedin.com/in/xueliu) 塑形技术和研究方向；[Andy Luo](https://www.linkedin.com/in/andyluo77/) 和 [Haichen Zhang](https://www.linkedin.com/in/haichen-zhang-9010b6382/) 做 ROCm 启用、router 模型训练、公开 MoM 实验。

还点名：[FAUST](https://github.com/FAUST-BENCHOU)、[David Shrader](https://www.linkedin.com/in/shraderdm/)、[Yang Wu](https://github.com/drivebyer)、[Ramakrishnan Sathyavageeswaran](https://github.com/ramkrishs)、[Kuntai Wu](https://github.com/WUKUNTAI-0211)、[Aayush Saini](https://github.com/AayushSaini101)、[siloteemu](https://github.com/siloteemu)、[Chen Wang](https://www.linkedin.com/in/chenw615/)、[Yue Zhu](https://www.linkedin.com/in/yue-zhu-b26526a3/)、[Senan Zedan](https://www.linkedin.com/in/senan-zedan-2041855b/)、[Yossi Ovadia](https://www.linkedin.com/in/yossi-ovadia-336b314/)、[Samzong Lu](https://www.linkedin.com/in/samzong)、[Liav Weiss](https://www.linkedin.com/in/liav-weiss-2a0428208)、[Asaad Balum](https://www.linkedin.com/in/asaad-balum-0928771a9/)、[Yehudit](https://www.linkedin.com/in/yehuditkerido/)、[Noa Limoy](https://www.linkedin.com/in/noalimoy/)、[Marina Koushnir](https://github.com/mkoushni)、[Jared Wen](https://github.com/JaredforReal)、[Abdallah Samara](https://www.linkedin.com/in/abdallah-samara)、[Hen Schwartz](https://www.linkedin.com/in/henschwartz)、[Srinivas A](https://www.linkedin.com/in/sriniabhiram)、[Yang Zhu](https://github.com/carlory)、[Jintao Zhang](https://www.linkedin.com/in/jintao-zhang-402645193/)、[yuluo-yx](https://github.com/yuluo-yx)、[cryo](https://github.com/cryo-zd)、[Bishen Yu](https://github.com/OneZero-Y)、[Zhijie Wang](https://github.com/aeft)、[Hao Wu](https://github.com/haowu1234)、[Qiping Pan](https://www.linkedin.com/in/qiping-pan-8662ab215/)。

这个里程碑：**1,734** 个 commit，**150+** 贡献者。点名 MBZUAI、McGill、Mila、Rice，以及更宽的 vLLM、AMD、Intel、Meta、Red Hat、Microsoft、Google、IBM、NVIDIA、Hugging Face、NASA、Nutanix、DaoCloud 和开源社区。

![community](../../../../assets/vllm/blog/serving/semantic-router-mom/14-community.png)

**Figure 14.** 造 MoM 引擎是开放系统题，要模型和基础设施整圈社区。

- GitHub：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- 文档：[vllm-sr.ai](https://vllm-sr.ai)
- 模型：[Hugging Face MoM family](https://huggingface.co/LLM-Semantic-Router)
- Slack：[vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN) 的 `#semantic-router`

收束：Semantic Router 起手帮基础设施给每条请求选对模型。现在把地基从单模往外推——往能在设备和环境之间协调、评测、运营多只模型的系统走。
