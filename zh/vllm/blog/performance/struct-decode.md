---
source: https://vllm.ai/blog/2025-01-14-struct-decode-intro
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Structured decoding：给会说话的模型一套不会说错格式的栅栏

英文对照：`en/vllm/blog/performance/struct-decode.md`  
原文：https://vllm.ai/blog/2025-01-14-struct-decode-intro  
2025-01-14。V1 当时还「即将发布」——那些路线图条目后来大部分进了 V1。JSON mode、function calling、agent 工具参数，底下往往就是这件事。

LLM 擅长「这段话接下来最可能是什么」。它不擅长「必须是合法 JSON」。few-shot 可以劝，不能保证；为 JSON 单独微调又太贵。Structured / constrained / guided decoding 是同一件事：按 schema 改下一 token 的概率（通常是 logit mask），让采样仍是采样，格式却过关。作者的比喻：它对 LLM，就像校验对 API。

Outlines 用 FSM 跟踪 schema 状态、滤掉非法 token。vLLM 里把 JSON schema 塞进 sampling params（Python 或 HTTP）即可。


本地图（原文版权仍归原站；学习对照用）：

![shogoth gpt](../../../../assets/vllm/blog/performance/struct-decode/01-shogoth-gpt.png)

![mermaid intro](../../../../assets/vllm/blog/performance/struct-decode/02-mermaid-intro.svg)

![constrained json fsm](../../../../assets/vllm/blog/performance/struct-decode/03-constrained-json-fsm.webp)

![vllm new xgrammar](../../../../assets/vllm/blog/performance/struct-decode/04-vllm-new-xgrammar.png)

![vllm xgrammar decode time per output token](../../../../assets/vllm/blog/performance/struct-decode/05-vllm-xgrammar-decode-time-per-output-token.png)

## 当时 Outlines 路径的疼

- FSM 按 token 走，一步一态，decode 慢。
- 靠 logit processor，在采样热路径上。组 batch 时每条请求都要编 FSM、算 mask，别人也得等——TTFT 被拖高。
- CFG 模式更慢，偶尔还能把引擎弄崩。
- jump-forward（一次填好 k 个已经确定的 token）和 logit processor「只看下一个」合不来。

## XGrammar

改走 PDA（可以想成一堆 FSM，每坨是一份 CFG），能一次跳多步状态。语法编译从 Python 挪到 C/`pthread`。负载下 TPOT 最多大约 **5×** 改善（Michael Goin / Red Hat 的图）。V0 仍把它当 logit processor，只是 tokenizer 数据有 cache。XGrammar 当时还缺：非 GBNF 语法、regex、带 regex / 数值范围的复杂 JSON。不够用就回落到 Outlines。仓库里还有 lm-format-enforcer，长上下文上他们测过会漏约束，也不如 Outlines 稳。

V1 计划（对 2026 年的读者已经多半是现在时）：guided decoding 升到 **scheduler**，不挡同一 batch 里的普通人；bitmask 在一个进程算、广播给 GPU worker；给投机解码的 tree scoring 和 tool-use 同一套 API。Anatomy 里 Structured Output Manager 就是这间房间。

功能页摘译：[speculative-decoding.md](../../features/speculative-decoding.md) 管的是「猜字」；这篇管的是「猜的字还得长对形状」。
