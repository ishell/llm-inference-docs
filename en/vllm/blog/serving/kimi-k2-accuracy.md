---
source: https://vllm.ai/blog/2025-10-28-kimi-k2-accuracy
lang: en
fetched: 2026-09-04
---

# Chasing 100% Accuracy: A Deep Dive into Debugging Kimi K2's Tool-Calling on vLLM

Chinese: [zh/vllm/blog/serving/kimi-k2-accuracy.md](../../../../zh/vllm/blog/serving/kimi-k2-accuracy.md)

2025-10-28. **Linian Wang (Peking University)**. vLLM **v0.11.0**. Benchmark: [MoonshotAI/K2-Vendor-Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier). Hub templates after [Kimi-K2-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905) commit `94a4053eb8863059dd8afc00937f054e1365abbd` or [Kimi-K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct) commit `0102674b179db4ca5a28cd9a4fb446f87f0c1454`. Later K3 chat-template is a Python renderer, not Jinja: [kimi-k3.md](kimi-k3.md). Token-ID debug cousin: [agent-lightning.md](agent-lightning.md). Study note; the handshake is the **chat template**, not the MoE kernel.

**TL;DR from the page:** For best compatibility with vLLM, use Kimi K2 models whose chat templates were updated after those commits. Updates are committed per model.

Local figures (copyright remains with the original site; study copies):

![k2 vendor verifier](../../../../assets/vllm/blog/serving/kimi-k2-accuracy/01-k2-vendor-verifier.jpeg)

## Introduction

Agentic workflows need robust tool-calling. Moonshot’s Kimi K2 is known for it. The author ran the official K2-Vendor-Verifier against vLLM, aiming to match Moonshot’s native API: thousands of tool calls, **zero** schema validation errors.

**K2-Vendor-Verifier on Moonshot AI’s API**

| Model Name | Provider | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Moonshot AI | MoonshotAI | 2679 | 1286 | 35 | **0** | **1286** |
| Moonshot AI Turbo | MoonshotAI | 2659 | 1301 | 40 | **0** | **1301** |

Out of the box on vLLM the same bench was broken.

**Initial test on vLLM**

- vLLM: `v0.11.0`
- HF: `moonshotai/Kimi-K2-Instruct-0905` at commit `09d5f937b41ae72c90d7155c9a901e2b5831dfaf`

| Model Name | finish_reason: stop | finish_reason: tool_calls | finish_reason: others | Schema Validation Errors | Successful Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi-K2-Instruct-0905 (initial HF) | 3705 | 248 | 44 | 30 | **218** |

Over 1200 potential tool calls, only **218** parsed — below **20%**. Not a minor bug: a breakdown between the model and the serving engine. Three compatibility issues between Kimi K2’s `chat_template` and vLLM.

## The Debugging Journey: Uncovering Three Core Issues

### Problem 1: The Case of the Missing `add_generation_prompt`

Requests that should have triggered tool calls ended with `finish_reason: stop`. Broader: the model was not generating a structured assistant reply at all — it continued the conversation as **plain text**. That would hurt any chat scenario, not only tool-calling.

**Investigation.** Bypass vLLM’s high-level `/v1/chat/completions`: call the tokenizer’s `apply_chat_template` **externally**, then send the string to `/v1/completions`. That skipped vLLM’s internal template application and resolved most failures. The bug was how vLLM **used** the chat template.

**Root cause.** Kimi’s `apply_chat_template` takes `**kwargs` for extra model-specific parameters. One of them, `add_generation_prompt=True`, is required to mark the start of the assistant turn.

Correct suffix: `...<|im_assistant|>assistant<|im_middle|>`

Without `add_generation_prompt=True`, the prompt truncated right after the user message. The model had no instruction to begin its turn.

vLLM, for security ([PR #25794](https://github.com/vllm-project/vllm/pull/25794)), inspects the function signature and only passes arguments that are **explicitly defined**. `add_generation_prompt` hid in `**kwargs`, so vLLM dropped it. Formatting failed **silently**.

**Fix.** Kimi updated `tokenizer_config.json` on the Hub: declare `add_generation_prompt` as a supported chat-template parameter so vLLM can pass it. The author also submitted [PR #27622](https://github.com/vllm-project/vllm/pull/27622): whitelist standard chat-template parameters when tokenizers accept them via `**kwargs`.

### Problem 2: How an Empty `content` Derailed the Prompt

After problem 1, a subtler class of formatting errors remained.

**Investigation.** Conversations with historical tool calls where `content` was an empty string `''`. vLLM, aiming at a standardized internal representation, **promotes** `content: ''` to `content: [{'type': 'text', 'text': ''}]`.

**Root cause.** Kimi’s Jinja chat template expected a **string**. Handed a list, it dumped the literal list into the prompt.

Incorrect snippet:

```
...<|im_end|><|im_assistant|>assistant<|im_middle|>[{'type': 'text', 'text': ''}]<|tool_calls_section_begin|>...
```

Correct:

```
...<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|>...
```

**Fix.** Template now checks the type of `content`: string → render directly; iterable (list) → process it, no literal dump. Kimi shipped the Hub update.

### Problem 3: A Tool-Call ID Parser That Was Too Strict

Even a syntactically correct tool call sometimes failed to parse. Often the culprit was **conversation history**, not the current turn.

**Investigation.** Raw `text_completion` output. In edge cases — especially when misled by a malformed history — the model emitted IDs that did not match Kimi’s spec. Example: `search:2`. Official format: `functions.func_name:idx` ([tool_call_guidance.md](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/main/docs/tool_call_guidance.md)).

**Root cause.** Kimi-K2 expects historical tool-call IDs in `functions.func_name:idx`. A history message from another system with `search:0` can confuse it into generating a “similar” wrong ID.

Moonshot’s official API is **not** exposed to this: before invoking K2 it **renames** historical IDs to the spec. That guardrail was missing in a direct vLLM setup.

vLLM’s parser was brittle: equivalent to `function_id.split('.')[1].split(':')[0]`. On `search:2`, the `.` split raises `IndexError` and the whole valid tool call is discarded.

**Fix.** Kimi’s recommendation: **normalize** historical IDs to `functions.func_name:idx` before sending. Fixing the first two prompt issues also cut how often non-compliant IDs appeared — a correctly formatted context makes the model more likely to emit correct IDs. Parser robustness: [PR #27565](https://github.com/vllm-project/vllm/pull/27565).

## Final Results and a New Discovery

After Hub template updates, K2-Vendor-Verifier again:

| Metric | Value | Description |
| --- | --- | --- |
| Tool-Call F1 Score | 83.57% | Harmonic mean of precision and recall — does the model trigger at the right time |
| Precision | 81.96% | TP / (TP + FP) |
| Recall | 85.24% | TP / (TP + FN) |
| Schema Accuracy | 76.00% | Syntactically correct and pass validation |
| Successful Tool Calls | 1007 | Parsed and validated |
| Total Tool Calls Triggered | 1325 | Model attempts |
| Schema Validation Errors | 318 | Triggered calls that failed parsing or validation |
| Overall Success Rate | 99.925% | 3997/4000 requests completed |

Prose on the page: successfully parsed **218 → 971 (4.4×)**; then **316** `schema_validation_error_count`. Table above is the numeric block (1007 / 318). Closer to the official API; not identical.

New issue: the model on vLLM sometimes called tools **not declared in the current request** (e.g. `img_gen` from chat history). Known hallucination. Moonshot’s API has an **Enforcer**: constrained decoding so the model can only emit tokens for tools **in this request**. vLLM did not have that then. The page flags it as an open-source opportunity; Kimi was working with vLLM to integrate the Enforcer.

## Key Takeaways and Best Practices

- **The devil is in the chat template.** It is the handshake between model and serving framework. Validate every piece of template logic against the framework’s assumptions.
- **Peel back the abstraction.** `/chat/completions` is convenient and can hide the root cause. Drop to `/completions`. Manually building the input isolates the problem.
- **Token IDs are the ultimate ground truth.** For the most subtle issues, inspect the final token-ID sequence sent to the model. OpenAI-compatible APIs that return token IDs help — same door as [agent-lightning.md](agent-lightning.md).
- **Understand framework design philosophy.** vLLM’s strict `**kwargs` handling is a **security choice**, not a bug ([PR #25794](https://github.com/vllm-project/vllm/pull/25794)).
- **The open-ecosystem challenge.** An Enforcer is a hallmark of polished proprietary APIs. Doing it robustly in vLLM is community work.

## Conclusion

Collaborative debugging resolved the critical K2 tool-calling compatibility issues on vLLM, boosting success by **over 4×** toward expected performance. The page offers the story as a roadmap for integrating complex models: methodical, template-level, not kernel-level.

## Acknowledgements

Kimi engineers: root-cause expertise, swift Hub fixes. Kaichao You and Chauncey Jiang (vLLM) for onboarding and walking through vLLM’s tool-call path.
