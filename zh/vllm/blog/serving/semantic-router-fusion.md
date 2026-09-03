---
source: https://vllm.ai/blog/2026-06-16-vllm-sr-fusion-api
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Fusion：面板、法官、合成，但是一条路由决策

英文对照：[en/vllm/blog/serving/semantic-router-fusion.md](../../../../en/vllm/blog/serving/semantic-router-fusion.md)  
原文：https://vllm.ai/blog/2026-06-16-vllm-sr-fusion-api  
2026-06-16。OpenRouter DRACO 分数是**他们的表**，不是 vLLM-SR 评测。MoM 总论见 [new chapter](semantic-router-mom.md)；Looper 全家见 [Micro-Agent](semantic-router-micro-agent.md)。

Fusion 不是全局开关。信号先描述请求，决策再决定走单模还是 Fusion。入口三条：

- `model: "vllm-sr/auto"`（别名 `auto` / `MoM`）：全决策；只有 `algorithm.type: fusion` 才进面板。
- `model: "vllm-sr/fusion"`：只匹配 Fusion 决策；没有匹配就报错（除非请求自带 panel override）。
- `plugins: [{ "id": "fusion", ... }]`：单次覆盖 judge / panel。

面板并发、`max_concurrent` 封顶。`on_error: skip` 允许残局；`fail` 立刻失败。Judge 抽共识、矛盾、盲区；合成一答。注册过的 Fusion slug 不能当 judge/panel，防递归。Agent 循环里 **只有最终 judge 能 `tool_calls`**；面板只看文本。

OpenRouter 演示行（DRACO）：Fable 5 + GPT-5.5、Opus 4.8 合成 **69.0%**；三面板 **68.3%**；同模两份 Opus **65.5%**；单 Fable 5 **65.3%**；预算三件套 **64.7%**；单 DeepSeek V4 Pro **60.3%** / Kimi K2.6 **53.7%** / Gemini 3 Flash **43.1%**。他们自己说还要做公开对照——当外部信号，不当承诺。

本地图（原文版权仍归原站；学习对照用）：

![hero v2](../../../../assets/vllm/blog/serving/semantic-router-fusion/01-hero-v2.png)

![fusion entry modes](../../../../assets/vllm/blog/serving/semantic-router-fusion/02-fusion-entry-modes.png)

![fusion decision not default](../../../../assets/vllm/blog/serving/semantic-router-fusion/03-fusion-decision-not-default.png)

![fusion stage contracts](../../../../assets/vllm/blog/serving/semantic-router-fusion/04-fusion-stage-contracts.png)
