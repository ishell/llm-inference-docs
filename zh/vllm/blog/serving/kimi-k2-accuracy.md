---
source: https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Kimi K2 tool-calling：坏的是 chat template 握手，不是 MoE kernel

英文对照：[en/vllm/blog/serving/kimi-k2-accuracy.md](../../../../en/vllm/blog/serving/kimi-k2-accuracy.md)  
原文：https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy  
2025-10-28。署名 **Linian Wang (Peking University)**。v0.11.0，K2-Vendor-Verifier。模板需晚于 Kimi-K2-0905 `94a4053eb8863059dd8afc00937f054e1365abbd` / Kimi-K2 `0102674b179db4ca5a28cd9a4fb446f87f0c1454`。后来的 K3 上线：[kimi-k3.md](kimi-k3.md)。token id 调试亲戚：[agent-lightning.md](agent-lightning.md)。结构化输出：[../performance/struct-decode.md](../performance/struct-decode.md)。GitHub raw 当时 404，按缓存摘录 + 页面重建。**不是 MoE kernel 的事**——坏在 chat template 握手。

坏基线用的是 `moonshotai/Kimi-K2-Instruct-0905`，Hub commit `09d5f937b41ae72c90d7155c9a901e2b5831dfaf`。官方 Moonshot API schema error **0**；vLLM 初始成功 tool call 218/1200+（<20%）。三刀之后 Hub 模板：解析 218→~1000，F1 **83.57%**，schema 仍 **76%**。Moonshot 有 Enforcer 做约束解码，当时 vLLM 没有。调试要落到 `/v1/completions` 和 token id，别只盯 `/v1/chat/completions`。

**页上 TL;DR：** 要跟 vLLM 对齐，用上述两个 commit 之后更新过 chat template 的 Kimi K2（每个模型各自提交）。

本地图（原文版权仍归原站；学习对照用）：

![k2 vendor verifier](../../../../assets/vllm/blog/serving/kimi-k2-accuracy/01-k2-vendor-verifier.jpeg)

**Figure.** K2-Vendor-Verifier 打 Moonshot 官方 API（金标准：上千次 tool call，schema error 为零）。

## Introduction

Agent 工作流靠 tool-calling 真正能解析。Moonshot 的 Kimi K2 就卖这个。作者拿官方 **K2-Vendor-Verifier** 先打 Moonshot 原生 API，再打 vLLM，想对齐官方那根杆。

Moonshot API 上的板（schema validation errors **0**）：

| Model Name | Provider | `finish_reason: stop` | `finish_reason: tool_calls` | `finish_reason: others` | Schema Validation Errors | Successful Tool Calls |
|---|---|---:|---:|---:|---:|---:|
| Moonshot AI | MoonshotAI | 2679 | 1286 | 35 | **0** | 1286 |
| Moonshot AI Turbo | MoonshotAI | 2659 | 1301 | 40 | **0** | 1301 |

开箱 vLLM 差很远。

初始：vLLM **v0.11.0**，HF `moonshotai/Kimi-K2-Instruct-0905` @ `09d5f937…`：

| Model Name | `stop` | `tool_calls` | `others` | Schema Validation Errors | Successful Tool Calls |
|---|---:|---:|---:|---:|---:|
| Kimi-K2-Instruct-0905（当时 Hub） | 3705 | 248 | 44 | 30 | **218** |

1200+ 次潜在 tool call 里只解析出 **218**——**<20%**。不是 kernel 算错。三刀都在 chat template / parser 兼容。

## 调试：三处核心问题

### Problem 1：缺了 `add_generation_prompt`

本该出 tool call 的请求，`finish_reason: stop`。比工具更宽：模型根本没开 assistant 这一轮——当续写散文。

**隔离。** 绕开 `/v1/chat/completions`。在 vLLM **外面** 调 `tokenizer.apply_chat_template`，字符串送 `/v1/completions`。大部分失败就好了。问题在 vLLM 怎么套模板，不在采样。

**根因。** Kimi 的 `apply_chat_template` 用 `**kwargs` 收模型私有参数。其中一个是 `add_generation_prompt=True`：给 assistant 回合加前缀。

正确后缀：

```text
...<|im_assistant|>assistant<|im_middle|>
```

不传这个 flag，prompt 停在 user 消息后。模型看不到「开始当助手」的 token，既不 tool call，也不出结构化回复。

vLLM 出于安全（[PR #25794](https://github.com/vllm-project/vllm/pull/25794)）只转发函数签名里 **显式声明** 的参数。`add_generation_prompt` 藏在 `**kwargs` 里，被丢掉。格式静默坏掉。

**修。** Kimi 在 Hub 上改 `tokenizer_config.json`：把 `add_generation_prompt` 写成正经参数，vLLM 才能传。作者另提 [PR #27622](https://github.com/vllm-project/vllm/pull/27622)：tokenizer 只从 `**kwargs` 收标准 chat-template 参数时，白名单放行。

### Problem 2：空的 `content` 把 prompt 带歪

第一刀之后，还有一类更细的格式错。

**隔离。** 历史 tool call 的 `content` 是空串 `''`。vLLM 为了内部表示统一，把它抬成 list-of-dicts：

```text
content: ''  →  content: [{'type': 'text', 'text': ''}]
```

**根因。** Kimi 的 Jinja 模板按 **string** 渲。给到 list，就把字面量塞进 prompt。

错误片段：

```text
...<|im_end|><|im_assistant|>assistant<|im_middle|>[{'type': 'text', 'text': ''}]<|tool_calls_section_begin|>...
```

正确片段：

```text
...<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|>...
```

**修。** 模板先看 `content` 类型：string 直接渲；iterable 再遍历。Kimi 推到 Hub。

### Problem 3：tool-call ID parser 太死

语法上看着像样的 tool call，有时仍然解析失败。毒常常在 **历史**，不在当前轮。

**隔离。** 看裸的 `/v1/completions` 文本。例如：

```text
...<|tool_calls_section_begin|><|tool_call_begin|>search:2<|tool_call_argument_begin|>...
```

官方 Kimi ID 形状是 `functions.func_name:idx`。这里模型吐了 `search:2`。

**根因（Kimi 团队）。** K2 要求历史消息里的 tool-call ID 也是 `functions.…`。别的系统留下 `search:0`，会教它造一个「像但不对」的 ID。Moonshot 官方 API 在模型看见之前，会把历史 ID **改名** 成 `functions.func_name:idx`。直接打 vLLM 没有这道护栏。

vLLM parser 按官方形状写，大约：

```python
function_id.split('.')[1].split(':')[0]
```

碰到 `search:2`，`split('.')[1]` 抛 `IndexError`，整次本来还能用的 call 被丢掉。

**他们建议的修法：** 厂商/用户在发送前把历史 ID 规范成 `functions.func_name:idx`——跟 Moonshot 同一道预处理。前两刀修好之后，模型乱造 ID 也少了。parser 鲁棒性：[PR #27565](https://github.com/vllm-project/vllm/pull/27565)。

## 终局，以及新发现

Hub tokenizer 更新后再跑 K2-Vendor-Verifier：

| Metric | Value | Description |
|---|---|---|
| Tool-Call F1 Score | **83.57%** | precision / recall 的调和平均（该不该在这一刻调用）。 |
| Precision | 81.96% | TP / (TP + FP)。 |
| Recall | 85.24% | TP / (TP + FN)。 |
| Schema Accuracy | **76.00%** | 句法正确且通过校验。 |
| Successful Tool Calls | 1007 | 解析并校验通过。 |
| Total Tool Calls Triggered | 1325 | 模型尝试次数。 |
| Schema Validation Errors | 318 | 触发了但解析/校验失败。 |
| Overall Success Rate | **99.925%** | 3997 / 4000 请求跑完。 |

正文另写成功解析 **218 → 971（4.4×）**，随后 **316** 条 `schema_validation_error_count`。表和正文对不上（1007 vs 971；318 vs 316）。两边都留，不擅自取舍。

新失败模式：vLLM 上的模型会调用 **本轮请求没声明** 的工具（例如历史里的 `img_gen`）。已知 hallucination。Moonshot API 有 **Enforcer**：约束解码，只能吐当前提供的工具名。**当时 vLLM 没有。** Kimi 说在和 vLLM 一起把 Enforcer 接进去。

## Key takeaways

- **握手就是 chat template。** 每一支都要对着 serving 栈的假设验一遍。
- **揭开抽象。** `/v1/chat/completions` 把 prompt 藏起来。落到 `/v1/completions`，字符串自己拼。
- **Token ID 才是真相。** OpenAI 兼容的 `return_token_ids`——见 [agent-lightning.md](agent-lightning.md)。这三刀作者没用上；仍是最后一件工具。
- **丢掉 `**kwargs` 是安全设计**，不是随机 bug（[PR #25794](https://github.com/vllm-project/vllm/pull/25794)）。
- **Enforcer 级的约束工具名** 当时仍是专有缺口。

## Conclusion

跟 Kimi 联调：Hub 模板 + tokenizer 修复、parser PR，解析成功 **>4×**。不是新引擎。剩下的 schema 缝是没有 Enforcer 时的模型幻觉，不是 MoE 算术。

## Acknowledgements

Kimi 工程师：根因和 Hub 更新。**Kaichao You**、**Chauncey Jiang**（vLLM）：入门和 tool-call 内部。
