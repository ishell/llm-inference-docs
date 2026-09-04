---
source: https://vllm.ai/blog/2025-01-14-struct-decode-intro
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Structured decoding：给会说话的模型一套不会说错格式的栅栏

英文对照：[en/vllm/blog/performance/struct-decode.md](../../../../en/vllm/blog/performance/struct-decode.md)  
原文：https://vllm.ai/blog/2025-01-14-struct-decode-intro  
2025-01-14。文中 V1「即将发布」是**当时时态**——那些路线图后来大部分进了 V1。JSON mode、function calling、agent 工具参数，底下往往就是这件事。

原文 TL;DR：

- Structured decoding 管的是输出**格式**，采样仍是采样
- 当时 vLLM 同时接 [Outlines](https://github.com/dottxt-ai/outlines) 和 [XGrammar](https://github.com/mlc-ai/xgrammar)
- XGrammar：负载下 TPOT 最多大约 **5×**
- 当时计划的 V1：性能，以及 **scheduler 级** mask 广播，好让混合 batch 里的普通人不受阻

作者还请人用哲学的方式读：这是对「模型吐出来的东西」怎么负责的一次改口，也是搭 agent 系统的一块砖。落地 [PR #10785](https://github.com/vllm-project/vllm/pull/10785)。当时的路线图 [#8779](https://github.com/vllm-project/vllm/issues/8779)。

本地图（原文版权仍归原站；学习对照用）：

![shogoth gpt](../../../../assets/vllm/blog/performance/struct-decode/01-shogoth-gpt.png)

![mermaid intro](../../../../assets/vllm/blog/performance/struct-decode/02-mermaid-intro.svg)

![constrained json fsm](../../../../assets/vllm/blog/performance/struct-decode/03-constrained-json-fsm.webp)

![vllm new xgrammar](../../../../assets/vllm/blog/performance/struct-decode/04-vllm-new-xgrammar.png)

![vllm xgrammar decode time per output token](../../../../assets/vllm/blog/performance/struct-decode/05-vllm-xgrammar-decode-time-per-output-token.png)

## 一点历史（原文的坐标系）

Turing 1950 之后两条路：

1. **GOFAI** —— 规则、专家系统（Haugeland；MYCIN 那种 rule trace）。语义表征撑不到通用任务，遇上「AI 冬天」。
2. **NFAI** —— 联结主义 / 统计（Rosenblatt → PDP 的 hidden layer）。如今做文本生成的 decoder-only transformer，坐在这边。

GOFAI 是确定的：意图写进程序。NFAI 是黑盒：内部表征由数据长出来。

Figure 1（Shogoth as GPTs）：RLHF / 任何 post-training，都是在往大复合系统里**注射规则**——一次 GOFAI 动作。

## 为什么需要 structured decoding

LLM 擅长「这段话接下来最可能是什么」。它不欠你一份合法 JSON。提示写得干净有帮助，不能当保证。Few-shot（「给我这样的 JSON……」）仍是采样，非法 JSON 仍然被允许。为 JSON 单独微调，训练和评测都贵。

**Structured / constrained / guided decoding** 是同一件事：按 schema **改下一 token 的概率**（通常是 logit mask），采样仍是采样，格式却过关。OpenAI 的 JSON mode 是产品化的版本。做过 agent、function calling、编程助手，多半已经在用。

作者的比喻：guided decoding 对 LLM，就像校验对 API。Dottxt 还写过：有时它甚至能[改善](https://blog.dottxt.co/coalescence.html)原生 Decode 性能。

## 当时 vLLM 怎么做

给引擎一份 schema，它把非法 token 滤掉。[Outlines](https://github.com/dottxt-ai/outlines) 用 **FSM** 跟踪 schema 状态、加 logit bias（Willard & Louf, 2023）。Figure 2 是顶层流程；Figure 3 是 constrained JSON 的 FSM（转引自 [LMSys compressed FSM, 2024](https://lmsys.org/blog/2024-02-05-compressed-fsm/)）。

在 vLLM 里：把 JSON schema 塞进 **sampling params**（Python SDK 或 HTTP）。

### 当时 Outlines 路径的疼（V0）

1. **Decode 慢。** FSM 按 token 走，一步一态，一次一个 token。
2. **组 batch 的瓶颈。** 实现是采样热路径上的 **logit processor**（当时的 [`outlines_logits_processors.py`](https://github.com/vllm-project/vllm/blob/80c751e7f68ade3d4c6391a0f3fce9ce970ddad0/vllm/model_executor/guided_decoding/outlines_logits_processors.py)）。每条请求编 FSM、同步算 mask，**同一 batch 里所有人**都得等 → TTFT 被拖高，吞吐下降。编 FSM 本身就贵，是 TTFT 的大头。
3. **CFG 模式。** JSON 还算快；CFG 慢很多，偶尔还能[把引擎弄崩](https://github.com/vllm-project/vllm/issues/10081)。
4. **没有 jump-forward。** [Jump-forward](https://lmsys.org/blog/2024-02-05-compressed-fsm/) 要一次填好已经确定的 k 个 token。Logit processor 只看得见**下一个**。

## XGrammar

[XGrammar](https://github.com/mlc-ai/xgrammar) 用 **PDA** 做 batch constrained decoding——可以想成**一堆 FSM**，每坨是一份 CFG。递归让它一次跳多步状态。语法编译的额外优化见 [MLC](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)。编译从 Python 挪到 C / `pthread`（针对疼点 1），也为后来的 jump-forward 铺路（疼点 4）。

Figure 4–5（Michael Goin / Red Hat）：XGrammar 对 Outlines；每个输出 token 的 Decode 时间。负载下 TPOT 最多大约 **5×**。

V0 仍把它当 [logit processor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/xgrammar_decoding.py)，只是 tokenizer 数据有 cache。成绩令人鼓舞，他们仍觉得还能挖。

**当时 XGrammar 还缺的**（相对 Outlines 的功能对等）——保持「文中当时」的标签：

- 还不会非 **GBNF** 语法（[vLLM PR](https://github.com/vllm-project/vllm/pull/10870)）
- 还不会 **regex**
- 还不会带 **regex / 数值范围** 的复杂 JSON（[vLLM #10899](https://github.com/vllm-project/vllm/pull/10899)，上游 [xgrammar #106](https://github.com/mlc-ai/xgrammar/pull/106)）

vLLM 当时默认走**基本的 XGrammar**；知道它伺候不了这条请求，就**回落到 Outlines**。

仓库里还有 **lm-format-enforcer**。他们测过：某些**长上下文**测例约束会漏，性能也不如 Outlines 稳。

## 当时给 V1 的tentative 计划

1. Guided decoding 升到 **scheduler**：调度器认得谁在用 structured decoding，就不该挡住同一 batch 里的普通人（疼点 2）。离开热路径。Jump-forward 也更自然（疼点 4）。
2. Bitmask **在一个进程里算**，再 **broadcast** 给 GPU worker，而不是每个 worker 算一遍。每条 sample 的 mask 广播带宽，当时说要仔细量。
3. 给**投机解码**和 **tool-use** 同一套底座——XGrammar 计划接 tool-use，好离开 Python [tool parser](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)。投机解码的 tree scoring 可以和 jump-forward 共用 API（取决于 scheduler 级 guided decoding）。

当时 Slack：`#feat-structured-output`。Anatomy 里 Structured Output Manager 就是这间房间。致谢名单见英文对照。

脚注里值得留下的：structured / constrained / guided 只是同一机制的三个名字；HuggingFace 的 logits-processor zoo 是更一般的阀门；MYCIN、IBM alignment、BoW、Attention 与 scaling law、LSTM 对 decoder-only，是史，不是 serving API。

功能页：[speculative-decoding.md](../../features/speculative-decoding.md) 管的是「猜字」；这篇管的是「猜的字还得长对形状」。
