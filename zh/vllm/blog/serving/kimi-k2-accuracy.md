---
source: https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Kimi K2 tool-calling：坏的是 chat template 握手，不是 MoE kernel

英文对照：[en/vllm/blog/serving/kimi-k2-accuracy.md](../../../../en/vllm/blog/serving/kimi-k2-accuracy.md)  
原文：https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy  
2025-10-28。署名 **Linian Wang (Peking University)**。vLLM **v0.11.0**。基准：[MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)。Hub 模板需晚于 [Kimi-K2-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) commit `94a4053eb8863059dd8afc00937f054e1365abbd`，或 [Kimi-K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) commit `0102674b179db4ca5a28cd9a4fb446f87f0c1454`。后来 K3 的 chat template 是 Python 渲染器，不是 Jinja：[kimi-k3.md](kimi-k3.md)。Token ID 调试亲戚：[agent-lightning.md](agent-lightning.md)。握手坏在 **chat template**，不在 MoE kernel。

**原文 TL;DR：** 要和 vLLM 处得好，用上述 commit 之后更新过 chat template 的 Kimi K2。更新按模型分别提交。

本地图（原文版权仍归原站；学习对照用）：

![k2 vendor verifier](../../../../assets/vllm/blog/serving/kimi-k2-accuracy/01-k2-vendor-verifier.jpeg)

## 引言

Agent 工作流要靠得住的 tool-calling。Moonshot 的 Kimi K2 以此出名。作者拿官方 K2-Vendor-Verifier 打 vLLM，想对齐 Moonshot 原生 API：几千次 tool call，schema 校验错误 **0**。

**K2-Vendor-Verifier，Moonshot AI 官方 API**

| 模型 | Provider | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Moonshot AI | MoonshotAI | 2679 | 1286 | 35 | **0** | **1286** |
| Moonshot AI Turbo | MoonshotAI | 2659 | 1301 | 40 | **0** | **1301** |

同一套 bench，vLLM 开箱却是坏的。

**vLLM 上的初测**

- vLLM：`v0.11.0`
- HF：`moonshotai/Kimi-K2-Instruct-0905`，commit `09d5f937b41ae72c90d7155c9a901e2b5831dfaf`

| 模型 | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2-Instruct-0905（当时 Hub） | 3705 | 248 | 44 | 30 | **218** |

一千多次潜在 tool call，只解析出 **218**——不到 **20%**。不是小 bug：模型和 serving 引擎之间的通信断了。Kimi K2 的 `chat_template` 和 vLLM 之间，三处兼容问题。

## 调试：三处核心问题

### 问题 1：失踪的 `add_generation_prompt`

本该触发 tool call 的请求，以 `finish_reason: stop` 结束。更宽的一层：模型根本没在生成结构化的 assistant 回复——它把对话当 **普通文本** 接着写。任何聊天场景都会钝，不只是 tool-calling。

**排查。** 绕开 vLLM 高层的 `/v1/chat/completions`：在外面调 tokenizer 的 `apply_chat_template`，再把字符串送给 `/v1/completions`。这跳过了 vLLM 内部套模板，多数失败消失。bug 在 vLLM **怎么用** 这份 chat template。

**根因。** Kimi 的 `apply_chat_template` 用 `**kwargs` 接模型专用参数。其中 `add_generation_prompt=True` 用来标记 assistant 回合开始。

正确后缀：`...<|im_assistant|>assistant<|im_middle|>`

不传 `add_generation_prompt=True`，prompt 在用户消息后就被截断。模型没有「开始你的回合」的指令。

vLLM 出于安全（[PR #25794](https://github.com/vllm-project/vllm/pull/25794)）检查函数签名，只传 **显式声明** 的参数。`add_generation_prompt` 藏在 `**kwargs` 里，被丢掉。格式化 **静默** 失败。

**修复。** Kimi 在 Hub 上更新 `tokenizer_config.json`：把 `add_generation_prompt` 声明成 chat template 支持的参数，vLLM 才能传进去。作者另提 [PR #27622](https://github.com/vllm-project/vllm/pull/27622)：tokenizer 经 `**kwargs` 接标准 chat-template 参数时，把它们加入白名单。

### 问题 2：空的 `content` 把 prompt 带歪

第一处修好后，更细的格式错误还在。

**排查。** 历史 tool call 里 `content` 是空字符串 `''`。vLLM 为了内部表示统一，把 `content: ''` **抬成** `content: [{'type': 'text', 'text': ''}]`。

**根因。** Kimi 的 Jinja chat template 按 **字符串** 来渲。塞进 list，就把 list 的字面量写进 prompt。

错误片段：

```
...<|im_end|><|im_assistant|>assistant<|im_middle|>[{'type': 'text', 'text': ''}]<|tool_calls_section_begin|>...
```

正确：

```
...<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|>...
```

**修复。** 模板先看 `content` 的类型：字符串直接渲；可迭代（list）按列表处理，不再把字面量倒进去。Kimi 发了 Hub 更新。

### 问题 3：过于严格的 tool-call ID parser

句法正确的 tool call，有时仍然解析失败。祸首常常是 **对话历史**，不是当前这一轮。

**排查。** 看原始 `text_completion`。边角——尤其被畸形历史带偏时——模型吐出的 ID 对不上 Kimi 规范。例如：`search:2`。官方格式：`functions.func_name:idx`（[tool_call_guidance.md](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/main/docs/tool_call_guidance.md)）。

**根因。** Kimi-K2 期望历史消息里的 tool-call ID 都是 `functions.func_name:idx`。另一套系统留下的 `search:0` 会让它模仿着生成「像那么回事」的错误 ID。

Moonshot 官方 API **不会** 撞上这个：调 K2 之前，它会把历史 ID **改名** 成规范格式。直接打 vLLM 时，这道护栏不在。

vLLM 的 parser 脆：等价于 `function_id.split('.')[1].split(':')[0]`。碰到 `search:2`，按 `.` 切开会 `IndexError`，整条本来有效的 tool call 被丢掉。

**修复。** Kimi 的建议：送进模型前，把历史 ID **归一** 成 `functions.func_name:idx`。前两处 prompt 修好之后，不合规 ID 也少了许多——上下文格式对了，模型更愿意吐对的 ID。Parser 鲁棒：[PR #27565](https://github.com/vllm-project/vllm/pull/27565)。

## 最终结果，和一个新发现

Hub 模板更新后再跑 K2-Vendor-Verifier：

| 指标 | 值 | 说明 |
| --- | --- | --- |
| Tool-Call F1 Score | 83.57% | precision / recall 的调和平均——有没有在对的时候触发 |
| Precision | 81.96% | TP / (TP + FP) |
| Recall | 85.24% | TP / (TP + FN) |
| Schema Accuracy | 76.00% | 句法正确且通过校验 |
| Successful Tool Calls | 1007 | 解析并校验通过 |
| Total Tool Calls Triggered | 1325 | 模型尝试次数 |
| Schema Validation Errors | 318 | 触发了却解析或校验失败 |
| Overall Success Rate | 99.925% | 3997/4000 请求跑完 |

正文叙述：成功解析 **218 → 971（4.4×）**；随后 **316** 条 `schema_validation_error_count`。上表是页上的数字块（1007 / 318）。更靠近官方 API，仍不是同一份。

新问题：vLLM 上的模型有时会调 **当前请求没声明** 的工具（例如历史里的 `img_gen`）。已知 hallucination。Moonshot API 有 **Enforcer**：约束解码，模型只能生成 **这一次请求里** 提供的工具对应的 token。当时 vLLM 没有。页上把它写成开源机会；Kimi 当时在和 vLLM 一起把 Enforcer 接进去。

## 要点和做法

- **魔鬼在 chat template。** 它是模型和 serving 框架之间的握手。模板逻辑的每一块，都要拿框架的假设对一遍。
- **把抽象揭开。** `/chat/completions` 方便，也会把根因藏住。落到 `/completions`。手工拼输入，才能把问题隔离。
- **Token ID 才是最终真相。** 最细的问题，要看真正送给模型的那串 token ID。能返回 token ID 的 OpenAI 兼容 API 有用——和 [agent-lightning.md](agent-lightning.md) 同一扇门。
- **先懂框架的设计哲学。** vLLM 对 `**kwargs` 的严格处理是 **安全选择**，不是 bug（[PR #25794](https://github.com/vllm-project/vllm/pull/25794)）。
- **开源生态的挑战。** Enforcer 是打磨过的专有 API 的标志。在 vLLM 里把它做稳、做干净，是社区的活。

## 结语

系统、协作的调试，把 Kimi K2 在 vLLM 上关键的 tool-calling 兼容问题收掉，成功率抬了 **四倍以上**，靠近预期。页上把这段经历写成路线图：落到模板层，方法要稳，别先怀疑 kernel。

## 致谢

Kimi 工程师：根因判断，Hub 上改得快。vLLM 的 Kaichao You、Chauncey Jiang：带进项目，把 tool-call 路径讲清楚。
