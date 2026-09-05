---
source: https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
lang: en
fetched: 2026-09-05
---

# Chasing 100% Accuracy: A Deep Dive into Debugging Kimi K2's Tool-Calling on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k2-accuracy.md](../../../../zh/vllm/blog/serving/kimi-k2-accuracy.md)

2025-10-28. **Linian Wang (Peking University)**. vLLM **v0.11.0**. Benchmark: [MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier). Hub templates after [Kimi-K2-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) commit `94a4053eb8863059dd8afc00937f054e1365abbd` or [Kimi-K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) commit `0102674b179db4ca5a28cd9a4fb446f87f0c1454`. Updates are committed per model. Later K3 chat-template is a Python renderer, not Jinja: [kimi-k3.md](kimi-k3.md). Token-ID debug cousin: [agent-lightning.md](agent-lightning.md). Study note; the handshake is the **chat template**, not the MoE kernel.

**TL;DR from the page:** For best compatibility with vLLM, use Kimi K2 models whose chat templates were updated after those commits.

Local figures (copyright remains with the original site; study copies):

![k2 vendor verifier](../../../../assets/vllm/blog/serving/kimi-k2-accuracy/01-k2-vendor-verifier.jpeg)

## Introduction

Agentic workflows need robust tool-calling. Moonshot’s Kimi K2 is known for it. The author ran the official K2-Vendor-Verifier against vLLM, aiming to match Moonshot’s native API: thousands of tool calls, **zero** schema validation errors — the gold standard.

**K2-Vendor-Verifier on Moonshot AI’s API**

| Model Name | Provider | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Moonshot AI | MoonshotAI | 2679 | 1286 | 35 | **0** | **1286** |
| Moonshot AI Turbo | MoonshotAI | 2659 | 1301 | 40 | **0** | **1301** |

Out of the box on vLLM the same bench was broken.

**Initial Test Results on vLLM**

- vLLM: `v0.11.0`
- HF: `moonshotai/Kimi-K2-Instruct-0905` at commit `09d5f937b41ae72c90d7155c9a901e2b5831dfaf`

| Model Name | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2-Instruct-0905 (then-Hub) | 3705 | 248 | 44 | 30 | **218** |

Out of over 1200 potential tool calls, only **218** parsed — below **20%**. Not a minor bug: communication between model and serving engine was broken. Three compatibility issues between Kimi K2’s `chat_template` and vLLM. After Hub and parser fixes, successful parses jumped over 4×.

## The Debugging Journey: Uncovering Three Core Issues

### Problem 1: The Case of the Missing `add_generation_prompt`

Requests that should have triggered tool calls ended with `finish_reason: stop`. Broader: the model was not generating a structured assistant reply at all — it continued the conversation as **plain text**. That dulls any chat scenario, not just tool-calling.

**Investigation.** Bypass vLLM’s high-level `/v1/chat/completions`: call the tokenizer’s `apply_chat_template` outside, then send the string to `/v1/completions`. That skips vLLM’s internal template application; most failures disappear. The bug is how vLLM **uses** the chat template.

**Root cause.** Kimi’s `apply_chat_template` takes model-specific parameters via `**kwargs`. `add_generation_prompt=True` marks the start of the assistant turn.

Correct suffix: `...<|im_assistant|>assistant<|im_middle|>`

Without `add_generation_prompt=True`, the prompt is truncated after the user message. The model has no “begin your turn” instruction.

vLLM, for security ([PR #25794](https://github.com/vllm-project/vllm/pull/25794)), inspects the function signature and only passes **explicitly declared** arguments. `add_generation_prompt` hid in `**kwargs` and was dropped. Formatting failed **silently**.

**Fix.** Kimi updated `tokenizer_config.json` on the Hub: declare `add_generation_prompt` as a supported chat-template parameter. The author also filed [PR #27622](https://github.com/vllm-project/vllm/pull/27622): whitelist standard chat-template parameters when tokenizers accept them via `**kwargs`.

### Problem 2: How an Empty `content` Derailed the Prompt

After the first fix, subtler formatting errors remained.

**Investigation.** Historical tool calls had `content` as empty string `''`. vLLM promotes `content: ''` to `content: [{'type': 'text', 'text': ''}]` for a uniform internal representation.

**Root cause.** Kimi’s Jinja chat template renders **string** `content`. Handed a list, it wrote the list’s literal into the prompt.

Incorrect:

```
...<|im_end|><|im_assistant|>assistant<|im_middle|>[{'type': 'text', 'text': ''}]<|tool_calls_section_begin|>...
```

Correct:

```
...<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|>...
```

**Fix.** The template checks the type of `content`: render a string directly; process an iterable (list) as a list. Kimi shipped a Hub update.

### Problem 3: A Tool-Call ID Parser That Was Too Strict

Syntactically correct tool calls still sometimes failed to parse. The culprit was often **conversation history**, not the current turn.

**Investigation.** Raw `text_completion` output. In edge cases — especially when misled by malformed history — the model emitted IDs that did not match Kimi’s spec. Example: `search:2`. Official format: `functions.func_name:idx` ([tool_call_guidance.md](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/main/docs/tool_call_guidance.md)).

**Root cause.** Kimi-K2 expects historical tool-call IDs to be `functions.func_name:idx`. A leftover `search:0` from another system can make it generate a “similar” wrong ID.

Moonshot’s official API does **not** hit this: before invoking K2, it **renames** historical IDs to the canonical format. Direct vLLM has no such guardrail.

vLLM’s parser was brittle: equivalent to `function_id.split('.')[1].split(':')[0]`. On `search:2`, the `.` split raised `IndexError` and discarded the whole valid tool call.

**Fix.** Kimi’s recommendation: normalize historical IDs to `functions.func_name:idx` before sending them to the model. Fixing the first two prompt issues also reduced non-compliant IDs. Parser robustness: [PR #27565](https://github.com/vllm-project/vllm/pull/27565).

## Final Results and a New Discovery

After Hub template updates, K2-Vendor-Verifier again:

| Metric | Value | Description |
| --- | --- | --- |
| Tool-Call F1 Score | 83.57% | Harmonic mean of precision and recall — whether the model triggers at the right time |
| Precision | 81.96% | TP / (TP + FP) |
| Recall | 85.24% | TP / (TP + FN) |
| Schema Accuracy | 76.00% | Syntactically correct and passing validation |
| Successful Tool Calls | 1007 | Parsed and validated |
| Total Tool Calls Triggered | 1325 | Model attempts |
| Schema Validation Errors | 318 | Triggered but failed parse or validation |
| Overall Success Rate | 99.925% | 3997/4000 requests completed |

Prose on the page: successful parses **218 → 971 (4.4×)**; then **316** `schema_validation_error_count`. The table above is the page’s number block (1007 / 318). Closer to the official API, still not the same sheet.

New issue: the model on vLLM sometimes called tools **not declared in the current request** (e.g. `img_gen` from chat history). Known hallucination. Moonshot’s API has an **Enforcer**: constrained decoding so the model can only generate tokens for tools **in this request**. vLLM lacked it then. The page frames that as an open-source opportunity; Kimi was working with vLLM to land Enforcer.

## Key Takeaways and Best Practices

1. **The devil is in the chat template.** It is the handshake between model and serving framework. Validate every piece of template logic against the framework’s assumptions.
2. **Peel back the abstraction.** `/chat/completions` is convenient and can hide the root cause. Drop to `/completions`. Manually building the input isolates the problem.
3. **Token IDs are the ultimate ground truth.** The finest bugs need the actual token-ID sequence sent to the model. An OpenAI-compatible API that returns token IDs helps — same door as [agent-lightning.md](agent-lightning.md). The issues above did not require that layer, but it belongs in the toolbox.
4. **Understand the framework’s design philosophy.** vLLM’s strict `**kwargs` handling is a **security choice**, not a bug ([PR #25794](https://github.com/vllm-project/vllm/pull/25794)).
5. **The open-ecosystem challenge.** An Enforcer is a hallmark of polished proprietary APIs. Doing it cleanly in vLLM is community work.

## Conclusion

Systematic, collaborative debugging resolved the critical tool-calling compatibility issues for Kimi K2 on vLLM, lifting the success rate **over 4×** toward expectations. The page writes this as a roadmap: drop to the template layer; keep the method steady; do not start by blaming the kernel.

## Acknowledgements

Kimi engineers: root-cause judgment and fast Hub fixes. vLLM’s Kaichao You and Chauncey Jiang: onboarding and the tool-call path. vLLM’s place in serving is visible only after the screws come off.
