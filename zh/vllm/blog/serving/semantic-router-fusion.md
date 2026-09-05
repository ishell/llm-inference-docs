---
source: https://vllm.ai/blog/2026-06-16-vllm-sr-fusion-api
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Fusion：面板、法官、合成，但是一条路由决策

英文对照：[en/vllm/blog/serving/semantic-router-fusion.md](../../../../en/vllm/blog/serving/semantic-router-fusion.md)  
原文：https://vllm.ai/blog/2026-06-16-vllm-sr-fusion-api  
2026-06-16。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router.md](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。MoM：[mom](semantic-router-mom.md)。Looper：[micro-agent](semantic-router-micro-agent.md)。AMD 路由随笔：[semantic-router-amd.md](semantic-router-amd.md)。OpenRouter DRACO 分数是**他们的表**，不是 vLLM-SR 评测。页上说这次发的是 serving primitive；更大的公开质量评测是后续工作。

Fusion 是策略，不是全局 slug。先抽信号；只有 Fusion 决策才为面板付钱。[OpenRouter 的 Fusion 发布](https://openrouter.ai/blog/announcements/fusion-beats-frontier/) 被当成市场信号：模型面板是活着的 serving 形态——这篇不是在克隆一只托管端点。

本地图（原文版权仍归原站；学习对照用）：

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-fusion/01-hero-v2.png)

**Figure 1。** Fusion API 把模型多样性收成 vLLM-SR 的路由原语：panel、judge、synthesis、trace。

## vLLM-SR 的论点

多年默认的 serving 问题是：*这条请求该交给哪一只模型？* 仍然有用；已经不够。页上的生产策略还要：

- 简单请求送到快、便宜的模型
- 难请求升到更强的专家
- 切模型会伤上下文时，保住 session 连续
- 模型执行前先套隐私、安全、租户策略
- 分歧有价值时，扇出到几只模型
- 记下决策路径，运维能调试、能改进

核心看法：模型质量不只是 checkpoint 的属性。也是围着这只 checkpoint 的 serving 系统的属性。

[AMD GPU 上的 Mixture-of-Models](semantic-router-mom-amd.md) 引入了这套以路由为中心的看法：抽信号、选模型、协调异构 backend、把路由露出来。ReMoM 把它扩成多轮协作（见 [micro-agent](semantic-router-micro-agent.md)）。Fusion 补上更直接的 panel–judge–synthesis：当多次独立前向值得那截延迟时。

## Fusion 加了什么

Fusion 不是整部 MoM 故事。它是路由器工具箱里的一种算法。在 vLLM-SR 里它是 **路由策略**，不是钉死的全局端点：

1. **Signals** 描述请求：域、复杂度、上下文、安全、反馈，或其他证据。
2. **Decisions** 选普通路由还是 Fusion 路由。
3. **Fusion-only 入口** 用 `model: "vllm-sr/fusion"` 把匹配收窄到能 Fusion 的决策——不会悄悄掉回单模路由。
4. **Panel models** 各自独立出候选答案。
5. **一只 judge 模型** 抽出共识、矛盾、部分覆盖、独特洞察、盲区。
6. **一次 synthesis 调用** 交回给用户看的一答。
7. **Trace** 记下哪些模型参与、发生了什么。

托管的 model slug 把这些大多藏起来。vLLM-SR 把 panel、judge、policy、trace 摊开，让运维选 Fusion **该出现在哪**，而不是每条请求都付钱。

## 为什么 OpenRouter 的结果是有用的信号

OpenRouter 的发布被当成同一套系统想法的公开证明点。在 [DRACO](https://ar5iv.labs.arxiv.org/html/2602.11685)（难的开放式深度研究任务）上，OpenRouter 报过 fused 面板压过单模。

**这些是 OpenRouter 的数字，不是 vLLM-SR 基准。** 外部证据：模型组合配得上成为一等 serving 原语：

| Configuration reported by OpenRouter | Score |
| --- | ---: |
| Fusion: Fable 5 + GPT-5.5, synthesized by Opus 4.8 | 69.0% |
| Fusion: Opus 4.8 + GPT-5.5 + Gemini 3.1 Pro, synthesized by Opus 4.8 | 68.3% |
| Fusion: Opus 4.8 + Opus 4.8, synthesized by Opus 4.8 | 65.5% |
| Solo Claude Fable 5 | 65.3% |
| Fusion: Gemini 3 Flash + Kimi K2.6 + DeepSeek V4 Pro, synthesized by Opus 4.8 | 64.7% |
| Solo DeepSeek V4 Pro | 60.3% |
| Solo Kimi K2.6 | 53.7% |
| Solo Gemini 3 Flash | 43.1% |

对 vLLM-SR 有意思的是 **budget panel** 那一行（64.7%）：独立多样性把单只更便宜模型缺的质量找回来。这是路由器该管的买卖。

## Fusion 在 vLLM-SR 里怎么跑

原则：Fusion 是路由算法，不是全局模型开关。

全局 runtime 配置只登记哪些 model slug 会触发直接 Fusion 执行。Panel、judge、错误策略、模板、runtime 旋钮，都住在 **命中的路由决策** 上——按工作负载。研究路由可能要三家多样 provider。Code-review 路由可能要两只本地专家加一只更强的 synthesis 模型。隐私敏感路由可能把整块面板留在自托管 vLLM backend 上。

![fusion entry modes](../../../../assets/vllm/blog/serving/semantic-router-fusion/02-fusion-entry-modes.png)

**Figure 2。** Fusion 由信号驱动。Auto 路由可以选任意决策；直接 Fusion 路由只在 Fusion 决策里选；请求插件覆盖执行，不覆盖全局策略。

三条入口进同一只算法：

| Entry path | How vLLM-SR handles it |
| --- | --- |
| `model: "vllm-sr/auto"` | 完整信号和决策策略。只有选中的决策用 `algorithm.type: fusion` 才跑 Fusion；否则走命中的非 Fusion 路由。遗留别名 `auto` 和 `MoM` 仍支持。 |
| `model: "vllm-sr/fusion"` | 同样抽信号，但决策匹配只限能 Fusion 的决策。都不中就给明确的 no-match 错，除非请求自带 panel override。 |
| `plugins: [{ "id": "fusion", ... }]` | 单次覆盖 judge、panel、以及选中的 runtime 旋钮。没有 Fusion 决策命中、但给了 `analysis_models`，vLLM-SR 会建一次请求范围的 `fusion_direct` 执行。 |

Fusion looper 一旦跑起来，执行是明示的：

1. **Resolve policy。** 合并决策级 Fusion 配置、决策的 model refs、请求级插件覆盖。
2. **Protect the router。** 登记过的 Fusion slug **不能**当 judge 或 panel 模型——Fusion 请求不能递归调 Fusion。
3. **Run the panel。** Analysis 模型并发执行，`max_concurrent` 封顶。
4. **Handle failures by policy。** `on_error: skip` 允许残局面板；`on_error: fail` 让 provider 失败立刻可见。
5. **Analyze disagreement。** Judge 对共识、矛盾、部分覆盖、独特洞察、盲区做结构化分析。
6. **Synthesize or call a tool。** 最终 judge/synthesis 调用交回一条 assistant 回答；客户端给了 tools 时，也可以是 OpenAI 兼容的 `tool_calls`。
7. **Return trace and accounting。** Fusion trace、中间 panel 输出、失败模型记录、panel / judge / synthesis 汇总的 token 用量。

调用方拿到 OpenAI 兼容响应；运维拿到哪条决策打中、哪些模型参与、跑了几轮、什么失败了、总 token 用量。

这次发版：策略可控的面板、明示的阶段合同、provider 互操作、可追溯执行。质量评测（Fusion vs 单模 vs 前沿面板）点名为后续工作。

## Fusion 是决策，不是默认

独立视角有用时，Fusion 才有用。它贵：panel 调用 + judge + synthesis，通常更多延迟。生产问题不只是「能不能 fuse」，而是「**什么时候 Fusion 值得？**」

`model: "vllm-sr/auto"` 让路由器决定这条请求要不要 Fusion。简单 prompt 留在快的单模路由。难研究、含糊分析、高风险合成、或分歧有价值的任务，可以命中 Fusion 决策。同一层信号–决策可以在付延迟之前就编码域、租户、隐私、成本、session、安全策略。

`model: "vllm-sr/fusion"` 是 Fusion-only 路由。仍然抽信号、做决策；匹配收窄，不会悄悄 fallback。请求级 Fusion 插件覆盖一次调用的面板。

![fusion decision not default](../../../../assets/vllm/blog/serving/semantic-router-fusion/03-fusion-decision-not-default.png)

**Figure 3。** Fusion 是决策，不是默认。策略决定多出来的延迟值不值。

控制面对一只托管 Fusion slug：

| Production question | vLLM-SR control |
| --- | --- |
| 这条请求该不该 Fusion？ | `vllm-sr/auto` 加信号和决策 |
| 该用哪套 Fusion 策略？ | 能 Fusion 的决策，带优先级和规则 |
| 哪些模型该参与？ | 按决策配 judge 和 panel |
| 延迟和失败怎么处理？ | `max_concurrent`、`on_error`、可选 token 策略 |
| 模型能跑在哪？ | 本地 vLLM backend、私有端点、公开 provider |
| 运维怎么调试这条路由？ | 决策 metadata、Fusion trace、失败、汇总用量 |

## 决策之后：可追溯的 Fusion

一小段多模型工作流，阶段边界明示。Panel → 独立候选。Judge → 结构化分析。最后一阶段 → 一条 assistant 回答，或客户端给了 tools 时的 tool call。

面板模型失败：`on_error: skip` 带着残局证据继续，并记下失败模型；`on_error: fail` 立刻停。结构化 judge 输出解析不了：保住原始分析，标上 parse failure，不藏。最终响应可以带 Fusion trace、中间 panel 输出、失败模型记录、总 token 用量。

![fusion stage contracts](../../../../assets/vllm/blog/serving/semantic-router-fusion/04-fusion-stage-contracts.png)

**Figure 4。** 明示的阶段合同：panel 输出、judge 分析、synthesis、trace 记账都可检查。

Fusion 因此成为可编程 Mixture-of-Models 控制面的一种实现——不只是一个功能。

## 用 vLLM-SR 试

### 让路由器决定

`vllm-sr/auto` 在所有已配置决策里选：

```json
{
  "model": "vllm-sr/auto",
  "messages": [
    {
      "role": "user",
      "content": "What are the strongest arguments for and against carbon taxes?"
    }
  ]
}
```

命中的决策带 `algorithm.type: fusion` → Fusion。否则走普通的选中模型路径。

### 显式请求 Fusion

`vllm-sr/fusion`：仍然抽信号；只有能 Fusion 的决策有资格：

```json
{
  "model": "vllm-sr/fusion",
  "messages": [
    {
      "role": "user",
      "content": "What are the strongest arguments for and against carbon taxes?"
    }
  ]
}
```

### 单次覆盖面板

请求范围；不会把 judge 或 panel 默认搬进全局配置：

```json
{
  "model": "vllm-sr/fusion",
  "messages": [{ "role": "user", "content": "..." }],
  "plugins": [{
    "id": "fusion",
    "model": "google/gemini-3-flash-preview",
    "analysis_models": [
      "google/gemini-3-flash-preview",
      "moonshotai/kimi-k2.6",
      "deepseek/deepseek-v4-pro"
    ]
  }]
}
```

### Agent 循环里用 Fusion

保住 OpenAI 兼容的 tool 循环。Fusion 把 tool-call 权限 **只给最终 judge**。Panel 模型和结构化 judge-analysis 调用跑 **text-only**：它们看见对话历史，包括先前的 tool 结果，但 **不**收到 `tools` 或 `tool_choice`。

```json
{
  "model": "vllm-sr/fusion",
  "messages": [
    {
      "role": "user",
      "content": "Find the latest benchmark result and explain whether it changes our launch plan."
    }
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "web_search",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string" }
        },
        "required": ["query"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

Panel 出独立文本分析；judge 比较；只有最终 judge 能直接回答，或交回标准 OpenAI 兼容的 `tool_calls`。非 streaming：普通 Chat Completions JSON。Streaming：tool-call SSE chunk，`finish_reason: "tool_calls"`。客户端追加的 tool 结果在下一轮 Fusion 里保留——多轮 agent 循环继续能跑。

### 配入口和决策

全局配置只登记 API 入口别名：

```yaml
global:
  router:
    auto_model_names:
      - vllm-sr/auto
      - auto
      - MoM
```

Fusion slug 挂在 looper 集成下：

```yaml
global:
  integrations:
    looper:
      fusion:
        model_names:
          - vllm-sr/fusion
```

按决策的配置拥有路由语义、judge、panel、runtime 旋钮：

```yaml
routing:
  decisions:
    - name: deep-research-fusion
      description: Use model diversity for research prompts with high synthesis risk.
      rules:
        operator: AND
        conditions:
          - type: domain
            name: research
          - type: complexity
            name: needs_reasoning:hard
      algorithm:
        type: fusion
        fusion:
          model: google/gemini-3-flash-preview
          analysis_models:
            - google/gemini-3-flash-preview
            - moonshotai/kimi-k2.6
            - deepseek/deepseek-v4-pro
          max_concurrent: 3
          on_error: skip
```

拆开是故意的。`global` 是跟路由无关的 runtime 状态。Judge、panel、可选 token 预算、并发、路由语义，都属于决策。

可选的 OpenRouter 风格别名，给已有客户端：

```yaml
global:
  integrations:
    looper:
      fusion:
        model_names:
          - vllm-sr/fusion
          - openrouter/fusion
```

**默认，vLLM-SR 只登记 `vllm-sr/fusion`。**

## 接下来

OpenRouter 的 DRACO 结果是信号：模型面板值得认真评。点名的下一步：

- 跑更大的公开评测，超出 smoke 覆盖
- 对照 Fusion、ReMoM、AutoMix、Router-R1、单模基线
- 研究 budget 面板对前沿模型面板
- 把 trace 级诊断露出来：分歧、缺覆盖、judge 行为
- 让路由策略决定多出来的延迟何时正当

方向：最好的答案不会永远来自最大的模型。越来越多来自最好的 **model system**，而 vLLM-SR 该是那套系统可编程的地方。
