---
source: https://vllm.ai/blog/2025-11-19-signal-decision
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 信号–决策：14 类不够用之后

英文对照：[en/vllm/blog/serving/semantic-router-signal.md](../../../../en/vllm/blog/serving/semantic-router-signal.md)  
原文：https://vllm.ai/blog/2025-11-19-signal-decision  
2025-11-19。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router.md](semantic-router.md)。进 v0.1：[semantic-router-iris.md](semantic-router-iris.md)。分类 LoRA：[semantic-router-modular.md](semantic-router-modular.md)。后来：[athena](semantic-router-athena.md)、[session](semantic-router-session.md)、[themis](semantic-router-themis.md)、[mom](semantic-router-mom.md)。不要和引擎里的 [router.md](router.md) 混。这篇的信号目录是 **三种**（keyword / embedding / domain）；Iris 后来列六种。

同系列：[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[vision](semantic-router-vision.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)。

更早的 Semantic Router 把查询打进 **14** 个 MMLU 域，再送到对应模型。基础场景够用。生产上的企业流量把上限露出来了。

页上的例子：「帮我紧急审一段认证代码的安全漏洞」。分类器叫它 computer science，送到通用 coding 模型。丢掉的是：

- **urgency**：要立刻处理
- **security**：要专门能力和 jailbreak 防护
- **code review** 意图：推理有用
- **authentication** 复杂度：要仔细分析

约束：基于分类的路由只抓住意图的 **一维**——域——其余信号全丢。

这篇引入 **Signal-Decision Architecture**：从 14 个固定类，长到不限数量的路由决策。多维抽信号，AND/OR 加优先级，内置插件编排。

本地图（原文版权仍归原站；学习对照用）：

![signal 0](../../../../assets/vllm/blog/serving/semantic-router-signal/01-signal-0.png)

## 问题：分类路由为什么撑不住规模

以前的管线：

```text
User Prompt → MMLU Domain Classification → Model Selection
```

### 单维分析

只看域 / 科目。抓不住：

- **Urgency：** “urgent”、“immediate”、“critical”
- **Security sensitivity：** “vulnerability”、“exploit”、“breach”
- **Intent types：** code review、architecture design、troubleshooting
- **Complexity：** 简单 FAQ vs 复杂推理
- **Compliance：** PII 处理、监管约束

**页上的真实冲击：** 医疗查询「紧急患者数据泄露」可能到医疗模型，却没有 PII 保护和安全过滤——可能违 HIPAA。

### 固定类约束

锁在 14 个预定义 MMLU 类（math、physics、computer science、business、…）。做不到：

- 给具体业务域造自定义类
- 在一个域里写细粒度路由规则
- 走出学术科目分类

**真实冲击：** 企业有 **50+** 专用用例（法律合同、金融合规、医疗诊断、代码安全审计），塞不进 14 个类。

### 逻辑拧不动

不能组合条件，也不能实现复杂策略：

- 没有 AND/OR：「只有 urgent **AND** security-related 才上专家模型」
- 多条件命中没有优先级
- 不能按信号组合条件开插件

**真实冲击：** 做不了分层策略，比如「高优先级安全问题走推理 + jailbreak 防护，一般问题走 cached 回答」。

![signal](../../../../assets/vllm/blog/serving/semantic-router-signal/02-signal.png)

## 引入 Signal-Decision Architecture

把抽信号和路由决策拆开。灵活的决策引擎，加上内置插件编排。

### 架构总览

![signal 1](../../../../assets/vllm/blog/serving/semantic-router-signal/03-signal-1.png)

点名的三项：

1. **Multi-Signal Extraction** — 一次抽几维意图
2. **Decision Engine** — AND/OR 加按优先级选
3. **Plugin Chain** — 内置的 cache、安全、优化

### 完整请求流

![signal 2](../../../../assets/vllm/blog/serving/semantic-router-signal/04-signal-2.png)

## 核心概念

### Signals：多维 prompt 分析

这篇抽 **三种** 互补信号。各自技法不同。

![signal 3](../../../../assets/vllm/blog/serving/semantic-router-signal/05-signal-3.png)

#### Keyword signals：可解释的模式匹配

基于 regex 的词或短语。**人能看懂**——哪几个关键词打中了，看得见。

技术路：

- 编译过的 regex，匹配便宜
- AND/OR 布尔算子
- 区分大小写 / 不区分
- **不做模型推理**（零 ML 开销）

好处：合规审计和生产调试都透明。

点名的用例：urgency 标记（“urgent”、“immediate”、“asap”、“critical”）；security 关键词（“vulnerability”、“exploit”、“breach”、“CVE”）；合规词（“HIPAA”、“GDPR”、“PII”、“confidential”）；意图模式（“code review”、“architecture design”、“troubleshooting”）。

#### Embedding signals：可伸缩的语义理解

神经 embedding，查询和候选短语之间的语义相似。意图不必精确撞上关键词。

技术路：

- 候选短语的 embedding 离线预计算
- 运行时用轻量模型（例如 sentence-transformers）embed 查询
- 余弦相似，阈值可配
- 聚合：**max**（任一命中）、**mean**（平均相似）、**any**（按阈值）

好处：缩到几千条候选短语。加一条模式不必重训——加短语、算 embedding。

用例：意图改写（“I need help” → “technical support request”）；“How do I fix this bug?” ≈ “debugging assistance”；多语 embedding 做跨语言路由；拼写、缩写、口语。

#### Domain signals：数据集驱动的分类

MMLU 训过的分类器，打学术 / 专业域。自定义扩展走 **LoRA**。细节：[semantic-router-modular.md](semantic-router-modular.md)。

技术路：

- 在 MMLU 上微调分类（**14** 个基类）
- 自定义域用 LoRA adapter 扩
- 多标签分类
- 打 confidence

好处：企业可以加 **私有** 域类，不必整模重训。页上的例子：

- Healthcare：`medical_imaging`、`clinical_trials`、`pharmaceutical_research`
- Finance：`risk_modeling`、`algorithmic_trading`、`regulatory_compliance`
- Legal：`contract_law`、`intellectual_property`、`litigation_support`

![signal 4](../../../../assets/vllm/blog/serving/semantic-router-signal/06-signal-4.png)

用例：域专家模型（math → math-expert）；域合适的策略（medical → PII 保护）；专用知识库（legal → legal retrieval）；域专用插件（code → syntax validation）。

### Signal comparison

| Signal type | Technique | Interpretability | Scalability | Extensibility |
| --- | --- | --- | --- | --- |
| Keyword | Regex matching | High（规则透明） | Medium（手工模式） | 手工加 |
| Embedding | Neural embeddings | Low（黑盒相似） | High（几千条短语） | 动态加短语 |
| Domain | MMLU + LoRA | Medium（域标签） | Medium（14+ 类） | 自定义域用 LoRA adapter |

### 为什么要三种信号？

互补，不是重复：

- **Keyword** — 已知模式，快、可解释
- **Embedding** — 语义变体、大短语集
- **Domain** — 学术数据集、域专长

三种一起上，才是点。

### Decisions：灵活的路由逻辑

每条决策有：

**Signal combination。** AND（高精度）/ OR（高召回）。

**Priority。** 整数冲突解决。高的赢。分层策略靠它。

**Model reference。** 哪只模型，可选 LoRA adapter。推理模式和 effort 级别。

**Plugin chain。** 有序列表：semantic caching、jailbreak 检测、PII 保护、system prompt 注入、header mutation。

#### Decision evaluation flow

![signal 5](../../../../assets/vllm/blog/serving/semantic-router-signal/07-signal-5.png)

多条命中 → 最高优先级。都不中 → default 模型。

### Plugins：内置智力

这篇表里有 **五只** 内置插件，按决策配：

| Plugin | Purpose | Key features |
| --- | --- | --- |
| **semantic-cache** | 缓存相似查询 | 可配相似阈值，省钱 |
| **jailbreak** | 检测 prompt injection | 按阈值检测，可拦请求 |
| **pii** | 保护敏感信息 | Redact / hash / mask，GDPR / HIPAA |
| **system_prompt** | 注入自定义指令 | Replace 或 insert，角色定制 |
| **header_mutation** | 改 HTTP header | Add / update / delete，metadata 往下传 |

插件按配置顺序跑。每只可以改请求、拦住执行、或给下游写 metadata。

Iris 后来加了 `hallucination` 等；**这一页的清单就是上面五只**。

#### Plugin chain execution flow

![signal 6](../../../../assets/vllm/blog/serving/semantic-router-signal/08-signal-6.png)

## 从 14 到不限数量

**Traditional（有限）：**

```text
14 MMLU Categories → 14 Routing Rules → 14 Model Selections
```

不能造自定义类、不能组合条件、不能每条规则不同策略、不能走出域分类。

**Signal-Decision（不限）：**

```text
3 Signal Types × N Conditions × AND/OR Logic → Unlimited Decisions
```

不限自定义规则、灵活组合、每条决策自己的插件链、企业复杂度。

### 伸缩例子：企业 IT 支持

Traditional：14 条域路由（`computer_science` → code-model，`engineering` → engineering-model，再加 12 条写死的）。

Signal-Decision：几百条专用路由，点名的例子：

- Urgent + Security + Computer Science → security-expert + reasoning + jailbreak
- Code Review + High Complexity → architecture-model + reasoning
- FAQ + General → cached-model + semantic-cache
- Medical + PII Detected → medical-expert + PII-protection + disclaimer
- Legal + Confidential → law-expert + PII-hash + audit-headers

每条决策可以有自己的选模型、推理配置、插件链。

## Kubernetes-native 设计

两只 CRD：**IntelligentPool** 和 **IntelligentRoute**。

### 完整例子：企业 IT 支持

#### IntelligentPool：定义模型池

![signal code 0](../../../../assets/vllm/blog/serving/semantic-router-signal/09-signal-code-0.png)

**Caption（YAML 在图里，不把 HTML 再倒一遍）：** 池子里基座模型 `qwen3`，**4** 只专用 LoRA adapter，非专用查询的 fallback `qwen3`，每只模型自己的 reasoning-family 配置。

#### IntelligentRoute：定义路由逻辑

![signal code 1](../../../../assets/vllm/blog/serving/semantic-router-signal/10-signal-code-1.png)

**Caption：** 截图里的 route spec。周围正文列了：

**Multi-signal extraction**

- **3** 条 keyword 信号：urgency、security、code-review
- **2** 条 embedding 信号：technical-support、architecture-design
- **1** 条 domain 信号：computer-science

**Layered decision logic**

- Priority **100**：Urgent + Security + CS → security-expert + high reasoning + jailbreak + PII protection
- Priority **80**：Code Review + CS → code-reviewer + medium reasoning + cache + custom prompt
- Priority **60**：Architecture Design + CS → architecture-expert + high reasoning + cache
- Priority **40**：General Support → 基座模型 + aggressive cache

**Plugin orchestration**

- 安全关键查询：jailbreak + PII
- Code review：semantic cache + 自定义 system prompt
- Architecture 查询：更长 cache TTL（**2h vs 1h**）
- 一般查询：aggressive caching（阈值 **0.90**，TTL **4h**）

**Fallback**

- 都不中 → `defaultModel`（`general-assistant`）
- 多条命中 → 最高优先级

### Dynamic configuration flow

![signal 7](../../../../assets/vllm/blog/serving/semantic-router-signal/11-signal-7.png)

宣称的 Kubernetes-native 性质：零停机改配置、GitOps、多集群、namespace 隔离和 RBAC。

## 真实场景

### 企业 IT 支持

挑战：urgency、技术域、security sensitivity。

办法：优先级分层——100 Urgent+Security+CS → security-expert + reasoning + jailbreak；80 Technical Support+Debugging → code-expert + semantic-cache；60 General → general-model + aggressive-cache。

宣称结果：模型对口、cache 省钱、敏感问题上有安全。

### Healthcare platform

挑战：HIPAA——PII 保护和医疗免责声明。

办法：Health Domain → medical-expert + PII-redaction + disclaimer-prompt + audit-headers。

宣称结果：自动 PII、免责声明一致、有审计轨迹。

### Financial services

挑战：分层安全、PII、jailbreak、成本。

办法：Economics Domain → finance-expert + jailbreak + PII-hash + disclaimer + cache + compliance-headers。

宣称结果：企业级安全、监管合规、成本效率。

### Educational platform

挑战：科目 + 学习意图。

办法：Math + Learning Intent → math-expert + reasoning + patient-tutor-prompt + cache；Science + Tutorial → science-expert + engaging-educator-prompt。

宣称结果：个性化教学、复杂题上开推理、成本优化。

### Code assistant

挑战：不同复杂度要不同模型。

办法：Architecture Design → reasoning-model + high-effort + complexity-header；Code Review → code-expert + medium-reasoning + cache；Simple Questions → code-expert + cache-only。

宣称结果：选型合适、推理划算、简单查询快。

## Future roadmap

给后来工作垫底。两桶：

### Routing core performance

- **Radix tree for keyword matching** — 换掉 regex；目标是 **10,000+** 条 keyword 规则仍性能稳定。
- **HNSW for embedding search** — 近似近邻；规模目标点名「数百万候选短语」。
- **Parallel LoRA for decode-only models** — Decode 里多只 LoRA adapter，一只基座伺候多个域；少多租户切模型的税。

### Feature enhancements

- **Visual configuration console** — web UI，实时校验和测试，不必改 YAML。
- **Custom plugin framework** — SDK、社区市场。
- **Advanced analytics** — 实时决策 / 信号 / 成本监控，ML 驱动的建议。
- **Model evaluation via multi-turn dialogue** — 跟候选模型并行对话，LLM-as-a-Judge 看 coherence、relevance、safety、域专长。按真实表现动态路由，而不是静态规则。
- **Intent-aware internal/external model selection** — 敏感 / 专有 → 内部模型；一般查询可以用外部 API（OpenAI、Anthropic、…）。成本、延迟、合规按查询特征配平。

![signal 8](../../../../assets/vllm/blog/serving/semantic-router-signal/12-signal-8.png)

## Conclusion

从固定分类拧到灵活的信号决策：

- **Unlimited scalability** — 14 类 → 不限自定义规则
- **Multi-dimensional intelligence** — keyword、embedding、domain 一起
- **Flexible logic** — AND/OR 和优先级
- **Built-in security** — jailbreak、PII、合规插件
- **Cloud-native** — Kubernetes CRDs、动态配置、零停机更新

框给企业 AI 网关、多租户 SaaS、行业助手。

## Getting started

页尾是邀请，不是 CLI：去试 Signal-Decision 路由，进社区，给反馈。具体安装路径落在 [semantic-router-iris.md](semantic-router-iris.md)。
