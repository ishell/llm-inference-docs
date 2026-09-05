---
source: https://vllm.ai/blog/2025-12-14-halugate
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# HaluGate：工具已经说对了，模型还在编

英文对照：[en/vllm/blog/serving/halugate.md](../../../../en/vllm/blog/serving/halugate.md)  
原文：https://vllm.ai/blog/2025-12-14-halugate  
2025-12-14。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。文档：[vllm-semantic-router.com](https://vllm-semantic-router.com)。挂在 [Iris](semantic-router-iris.md) 的插件链上；`fact_check` 是 [signal-decision](semantic-router-signal.md) 里的一种信号。立项：[semantic-router](semantic-router.md)。分类核：[modular LoRA](semantic-router-modular.md)。不要和引擎里的 structured decode 混，也不要和 P/D [Router](router.md) 混。Slack `#semantic-router`。时延和 F1 是他们测的。

工具回了艾菲尔 1887–1889 / 330 m，模型仍说 1950 / 500 m——**外在幻觉**。HaluGate **不用 LLM-as-judge**：工具消息当 context，用户句当 question，助手句当要核的 claim。三只模型走原生 Rust / Candle：Sentinel（CPU 约 **12 ms**，验证准确率 **96.4%**）→ Detector + NLI Explainer。检测真跑起来，整段 **76–162 ms**。单靠 token 级幻觉类 F1 只有 **59%**；五类统一头只有 **21.7%** F1——所以留下两模型 ensemble。不核内在幻觉；没上下文就打 unverified header，不装没看见。

本地图（原文版权仍归原站；学习对照用）：

![halugate 0](../../../../assets/vllm/blog/serving/halugate/01-halugate-0.png)

## 问题：幻觉挡住投产

原文把幻觉写成生产部署 LLM 的最大拦路石。各行同一出戏：**法律**（捏造判例）、**医疗**（药物相互作用写错）、**金融**（编造财务数字）、**客服**（不存在的政策）。听起来像权威，一查就碎。

难的不是一眼假的胡话，是**嵌在大体正确的回答里的细伪造**——要领域知识或外部核验才抓得住。对企业来说，这种不确定是负债，不是资产。

## 场景：工具对了，模型仍错

原文的函数调用例子：

> **User**: "When was the Eiffel Tower built?"
>
> **Tool Call**: `get_landmark_info("Eiffel Tower")`
>
> **Tool Response**: `{"name": "Eiffel Tower", "built": "1887-1889", "height": "330 meters", "location": "Paris, France"}`
>
> **LLM Response**: "The Eiffel Tower was **built in 1950** and stands at **500 meters** tall in Paris, France."

工具数据是对的。模型那两处「事实」是直接顶着上下文的 **extrinsic hallucination**。

这招阴的地方：

- **用户信**：他们看见工具被调过了。
- **传统过滤漏**：没有有毒、没有有害内容。
- **再叫一只 LLM 当法官**：贵。

原文问：能不能自动、实时、毫秒级抓住？

## 洞察：函数调用已经是 ground truth

现代 function-calling API 自己就带着 grounding。事实问题会去调工具——查库、打 API、取文档。这些结果，原文当成和 RAG 检索文档语义等价。

**图。** 接地已经在 API 流里：工具输出是 context，用户是 question，助手是 claim。

不必另搭检索。不必 GPT-4 当法官。从现成流量里抽出三件：

| Component | 来源 | 用途 |
| --- | --- | --- |
| **Context** | Tool message 内容 | 核验的 ground truth |
| **Question** | User message | 意图 |
| **Answer** | Assistant response | 要核的 claim |

问题变成：**回答对上下文忠实吗？**

## 为什么不 LLM-as-judge

再叫一只模型来核，生产上根本不对劲：

| 做法 | Latency | Cost | Explainability |
| --- | --- | --- | --- |
| GPT-4 as judge | 2–5 seconds | $0.01–0.03/request | 低（黑盒） |
| Local LLM judge | 500 ms–2 s | GPU compute | 低 |
| **HaluGate** | **76–162 ms** | **CPU only** | **高（token-level + NLI）** |

LLM 法官还有：**position bias**、**verbosity bias**、**self-preference**、**inconsistency**（同一输入、不同判决）。他们要更快、更便宜、更能讲清楚。

## HaluGate：有条件的两段管线

效率对精度。不是每条查询都付 token 级检测的税。

![halugate 1](../../../../assets/vllm/blog/serving/halugate/02-halugate-1.png)

**图。** 两段：先 Sentinel，只有事实寻求才进 Detector + NLI。

### Stage 1：HaluGate Sentinel（提示分类）

不是每句都要核幻觉：

| Prompt | 要不要 fact-check | 理由 |
| --- | --- | --- |
| "When was Einstein born?" | 要 | 可核验事实 |
| "Write a poem about autumn" | 不要 | 创作 |
| "Debug this Python code" | 不要 | 技术协助 |
| "What's your opinion on AI?" | 不要 | 意见 |
| "Is the Earth round?" | 要 | 事实主张 |

对写诗、审代码跑 token 级检测，浪费，还容易误报（「你的诗里有未支撑的主张」）。

**预分类为什么要紧：** token 级检测随上下文长度线性涨。**4K** token 的 RAG 上下文约 **125 ms**；**16K** 约 **365 ms**。生产里约 **35%** 查询非事实，预分类宣称 **72.2%** 的效率增益——创作、代码、意见整段跳过贵检测。

[HaluGate Sentinel](https://huggingface.co/llm-semantic-router/halugate-sentinel) 是 ModernBERT 分类器，只答一句：*这句提示值不值得做事实核验？*

![halugate 2](../../../../assets/vllm/blog/serving/halugate/03-halugate-2.png)

**图。** Sentinel 二分类：FACT_CHECK_NEEDED 还是不必。

训练配比：

**要核（正类）：**

- **问答：** SQuAD、TriviaQA、Natural Questions、HotpotQA
- **真实性：** TruthfulQA（常见误解）
- **幻觉基准：** HaluEval、FactCHD
- **求知对话：** FaithDial、CoQA
- **RAG：** neural-bridge/rag-dataset-12000

**不核（负类）：**

- **创作：** WritingPrompts、故事生成
- **代码：** CodeSearchNet docstring、编程任务
- **意见 / 指令：** Dolly 非事实、Alpaca 创作

二分类：**96.4%** 验证准确率，原生 Rust / Candle 约 **12 ms**。

### Stage 2：token 级检测 + NLI 解释

事实寻求的提示，再跑两只模型。

#### Token 级幻觉检测

不是整句一个「幻觉 / 不是」。**Token-level** 标出*哪些* token 没被上下文撑住。

![halugate 3](../../../../assets/vllm/blog/serving/halugate/04-halugate-3.png)

**图。** 只给 answer token 打标：0 = supported，1 = hallucinated。

架构：

```text
Input: [CLS] context [SEP] question [SEP] answer [SEP]
                                          ↓
                              ModernBERT Encoder
                                          ↓
                    Token Classification Head (Binary per token)
                                          ↓
              Label: 0 = Supported, 1 = Hallucinated (for answer tokens only)
```

设计选择：

- **只分类 answer：** 不分类 context、不分类 question
- **Span merging：** 连续幻觉 token 并成 span
- **Confidence thresholding：** 可配；默认 **0.8**，在 precision / recall 之间拧

#### NLI 解释层

知道「有问题」不够，还要知道「为什么」。NLI 拿每个检出 span 对上下文分类：

![halugate 4](../../../../assets/vllm/blog/serving/halugate/05-halugate-4.png)

**图。** 每个 span 的 NLI：CONTRADICTION / NEUTRAL / ENTAILMENT。

| NLI label | 含义 | Severity | 动作 |
| --- | --- | --- | --- |
| **CONTRADICTION** | 和上下文冲突 | 4（高） | 标成错误 |
| **NEUTRAL** | 上下文撑不住 | 2（中） | 标成不可核 |
| **ENTAILMENT** | 上下文支持该主张 | 0 | 滤掉误报 |

**为什么要 ensemble：** 单 token 级在幻觉类上只有 **59% F1**——幻觉漏掉近一半，标出来的大约三分之一是误报。他们试过统一五类头（SUPPORTED / CONTRADICTION / FABRICATION 等），只有 **21.7% F1**——token 分类分不出*为什么*错。两段：LettuceDetect 那路负责召回，NLI 负责精度和可解释。

## 接到 Signal-Decision

HaluGate 是 [signal-decision](semantic-router-signal.md) 上的新信号类型，也是一只插件。随 [Iris](semantic-router-iris.md) 发出。

### `fact_check` 作为一种信号

和 keyword、embedding、domain 并列，`fact_check` 是一等信号。

![halugate 5](../../../../assets/vllm/blog/serving/halugate/06-halugate-5.png)

**图。** `fact_check` 条件化决策；`hallucination` 插件挂在那条决策上。

> **原文注：** 即便前沿模型，发版之间幻觉也会漂。[GPT-5.2 system card](https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf) 被引来说明相对前代有可测的幻觉差——再强的模型也要持续核验。

```yaml
decisions:
  - name: "factual-query-with-verification"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
        - type: "domain"
          name: "general"
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true
          hallucination_action: "header"
```

### 请求–响应之间的状态

分类发生在 **request time**，检测发生在 **response time**。状态得跨过这条界。

![halugate 6](../../../../assets/vllm/blog/serving/halugate/07-halugate-6.png)

**图。** `RequestContext` 先装分类和工具上下文，再装检测结果。

```yaml
RequestContext:
  # Classification results (set at request time)
  FactCheckNeeded: true
  FactCheckConfidence: 0.87

  # Tool context (extracted at request time)
  HasToolsForFactCheck: true
  ToolResultsContext: "Built 1887-1889, 330 meters..."
  UserContent: "When was the Eiffel Tower built?"

  # Detection results (set at response time)
  HallucinationDetected: true
  HallucinationSpans: ["1950", "500 meters"]
  HallucinationConfidence: 0.92
```

这段 YAML 里的数字是页上的艾菲尔例题，不是基准均值。

### `hallucination` 插件

按决策配置：

```yaml
plugins:
  - type: "hallucination"
    configuration:
      enabled: true
      use_nli: true  # Enable NLI explanations

      # Action when hallucination detected
      hallucination_action: "header"  # "header" | "body" | "block" | "none"

      # Action when fact-check needed but no tool context
      unverified_factual_action: "header"

      # Include detailed info in response
      include_hallucination_details: true
```

| Action | 行为 |
| --- | --- |
| `header` | 加警告 header，响应照过 |
| `body` | 把警告写进响应 body |
| `block` | 返回错误，不转发 LLM 输出 |
| `none` | 只记日志，用户看不见 |

## 响应 header

检测结果走 HTTP header，下游自己定政策：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-fact-check-needed: true
x-vsr-hallucination-detected: true
x-vsr-hallucination-spans: 1950; 500 meters
x-vsr-nli-contradictions: 2
x-vsr-max-severity: 4
```

没工具、事实却要核：

```http
HTTP/1.1 200 OK
x-vsr-fact-check-needed: true
x-vsr-unverified-factual-response: true
x-vsr-verification-context-missing: true
```

header 能做：**UI 免责**、**人工复核队列**、**审计日志**、对高严重度 CONTRADICTION **有条件拦截**。

## 三条路径

![halugate 7](../../../../assets/vllm/blog/serving/halugate/08-halugate-7.png)

**图。** Path 1 跳过，Path 2 只打 unverified header，Path 3 全检测。

| Path | 条件 | 加上的 Latency | 动作 |
| --- | --- | --- | --- |
| **Path 1** | 非事实提示 | ~12 ms（只跑分类器） | 放过 |
| **Path 2** | 事实 + 无工具 | ~12 ms | 加警告 header |
| **Path 3** | 事实 + 有工具 | 76–162 ms | 全检测 + header |

## 模型架构

三只模型：

![halugate 8](../../../../assets/vllm/blog/serving/halugate/09-halugate-8.png)

**图。** Sentinel / Detector / Explainer，都是 ModernBERT-base 一家。

### HaluGate Sentinel：提示二分类

**架构：** ModernBERT-base + LoRA adapter + 二分类头

**训练：**

- **Base：** `answerdotai/ModernBERT-base`
- **Fine-tuning：** LoRA（rank=16，alpha=32，dropout=0.1）
- **数据：** 14 个数据集里抽 **50,000** 条
- **Loss：** 带 class weight 的 CrossEntropy（处理不平衡）
- **优化：** AdamW，lr=2e-5，3 epochs

**推理：**

- **Input：** 原始提示文本
- **Output：** (class_id, confidence)
- **Latency：** CPU 上约 12 ms

LoRA：只更新 **2.2%** 参数（**149M** 里的 **3.4M**）。

### HaluGate Detector：token 级二分类

**架构：** ModernBERT-base + token classification head

**输入格式：**

```text
[CLS] The Eiffel Tower was built in 1887-1889 and is 330 meters tall.
[SEP] When was the Eiffel Tower built?
[SEP] The Eiffel Tower was built in 1950 and is 500 meters tall. [SEP]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Answer tokens (classification targets)
```

**输出：** 每个 answer token 一个二值（0=Supported，1=Hallucinated）

**后处理：**

1. 只留 answer 段的预测
2. confidence 阈值（默认 0.8）
3. 连续幻觉 token 并成 span
4. 带回 confidence 的 span

### HaluGate Explainer：三路 NLI

**架构：** 在 NLI 上微调过的 ModernBERT-base

**输入格式：**

```text
[CLS] The Eiffel Tower was built in 1887-1889. [SEP] built in 1950 [SEP]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^
                    Premise (context)                Hypothesis (span)
```

**输出：**

- **ENTAILMENT** (0)：上下文支持该主张
- **NEUTRAL** (1)：从上下文判不了
- **CONTRADICTION** (2)：上下文和主张冲突

**严重度映射：**

| NLI label | Severity | 读法 |
| --- | ---: | --- |
| ENTAILMENT | 0 | 多半误报——滤掉 |
| NEUTRAL | 2 | 主张不可核 |
| CONTRADICTION | 4 | 直接事实错误 |

## 为什么要原生 Rust / Candle

三只模型都走 **Candle**（Hugging Face 的 Rust ML 框架），再用 CGO 接到 Go：

![halugate 9](../../../../assets/vllm/blog/serving/halugate/10-halugate-9.png)

**图。** 进程内 Candle；没有 Python sidecar。

| 方面 | Python (PyTorch) | Native (Candle) |
| --- | --- | --- |
| **Cold start** | 5–10 s | <500 ms |
| **Memory** | 每模型 2–4 GB | 每模型 500 MB–1 GB |
| **Latency** | +50–100 ms 开销 | 近零开销 |
| **Deployment** | 要 Python 运行时 | 单二进制 |
| **Scaling** | GIL 争用 | 真并行 |

不必另开 Python 服务、sidecar、模型服务器——全在进程里。同一扇 Candle 门见 [modular LoRA](semantic-router-modular.md)。

### Latency 拆开

他们测的生产管线各件：

| Component | P50 | P99 | 备注 |
| --- | ---: | ---: | --- |
| Fact-check classifier | 12 ms | 28 ms | ModernBERT 推理 |
| Tool context extraction | 1 ms | 3 ms | JSON 解析 |
| Hallucination detector | 45 ms | 89 ms | Token 分类 |
| NLI explainer | 18 ms | 42 ms | 每个 span 一次分类 |
| **Total overhead** | **76 ms** | **162 ms** | 检测真跑的时候 |

他们说 **76–162 ms** 比起典型 LLM 生成（**5–30 seconds**）可以忽略，所以能同步拦在请求路径上。以原文测量为准，不是你的 SLA。

## 配置参考

```yaml
# Model configuration
hallucination_mitigation:
  # Stage 1: Prompt classification
  fact_check_model:
    model_id: "models/halugate-sentinel"
    threshold: 0.6  # Confidence threshold for FACT_CHECK_NEEDED
    use_cpu: true

  # Stage 2a: Token-level detection
  hallucination_model:
    model_id: "models/halugate-detector"
    threshold: 0.8  # Token confidence threshold
    use_cpu: true

  # Stage 2b: NLI explanation
  nli_model:
    model_id: "models/halugate-explainer"
    threshold: 0.9  # NLI confidence threshold
    use_cpu: true

# Signal rules for fact-check classification
fact_check_rules:
  - name: needs_fact_check
    description: "Query contains factual claims that should be verified"
  - name: no_fact_check_needed
    description: "Query is creative, code-related, or opinion-based"

# Decision with hallucination plugin
decisions:
  - name: "verified-factual"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "fact_check"
          name: "needs_fact_check"
    plugins:
      - type: "hallucination"
        configuration:
          enabled: true
          use_nli: true
          hallucination_action: "header"
          unverified_factual_action: "header"
          include_hallucination_details: true
```

页上的阈值：Sentinel **0.6**，Detector **0.8**，NLI **0.9**。

## 离线也能当评测架

同一套管线可以离线给模型打幻觉分。把基准数据集送进检测，而不是拦线上请求。

![halugate 10](../../../../assets/vllm/blog/serving/halugate/11-halugate-10.png)

**图。** HaluGate 当 QA / RAG 数据集上的幻觉打分器。

流程：

1. **Load dataset：** TriviaQA、Natural Questions、HotpotQA，或企业自己的 context–question
2. **Generate：** 待测模型，带着给定上下文
3. **Detect：** (context, query, response) 过 HaluGate Detector
4. **Classify severity：** HaluGate Explainer 给每个标出的 span
5. **Aggregate：** 幻觉率、矛盾比、按类拆开

页上**没有**发布一份用 HaluGate 打出来的、点名 LLM 的排行榜。

## 限制和范围

盯的是 **extrinsic hallucination**——工具 / RAG 上下文才是地。已知做不到：

### 抓不住什么

| 限制 | 例子 | 原因 |
| --- | --- | --- |
| **Intrinsic hallucinations** | 没任何 tool call，模型说 "Einstein was born in 1900" | 没有可对的上下文 |
| **No-context scenarios** | 用户问事实，没定义工具 | 缺 ground truth |

### 透明降级

判定要核事实、却没有工具上下文：标成 "unverified factual"，不装放过：

```http
x-vsr-fact-check-needed: true
x-vsr-unverified-factual-response: true
x-vsr-verification-context-missing: true
```

## 致谢

- **Token 级架构：** [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)（KRLabs）——ModernBERT 幻觉检测
- **NLI：** [tasksource/ModernBERT-base-nli](https://huggingface.co/tasksource/ModernBERT-base-nli)
- **训练数据：** TruthfulQA、HaluEval、FaithDial、RAGTruth，以及其他公开基准

## 收束

页上的宣称：

- **有条件核验：** 非事实跳过，事实才核
- **Token 级精度：** 知道哪些主张没撑住
- **可解释：** NLI 说*为什么*
- **Zero-latency integration：** 原生 Rust，没有 Python sidecar（他们的口号；测到的开销是上面那张表）
- **可执行的透明：** header 给下游政策用

下一次 LLM 调了工具、拿到准数据、答案仍编——他们希望 HaluGate 比用户先抓住。

**资源：** [signal-decision 原文](https://blog.vllm.ai/2025/11/19/signal-decision.html)（笔记：[semantic-router-signal](semantic-router-signal.md)）、[GitHub](https://github.com/vllm-project/semantic-router)、[文档](https://vllm-semantic-router.com)。Slack `#semantic-router`。
