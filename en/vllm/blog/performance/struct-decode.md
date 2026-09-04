---
source: https://vllm.ai/blog/2025-01-14-struct-decode-intro
lang: en
fetched: 2026-09-04
---

# Structured Decoding in vLLM: a gentle introduction

Chinese: [zh/vllm/blog/performance/struct-decode.md](../../../../zh/vllm/blog/performance/struct-decode.md)

2025-01-14. Study note. **“V1 upcoming” in the post is historical** — most of that roadmap later landed. JSON mode, function calling, and agent tool arguments are usually this underneath.

**TL;DR from the page:**

- Structured decoding controls output *format* while sampling stays sampling
- vLLM then supported both [Outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar)
- XGrammar: up to **5×** better TPOT under load
- Then-planned V1: performance + **scheduler-level** mask broadcasting for mixed batches

The authors also ask you to read it philosophically: structured decoding as a shift in how we treat LLM outputs, and as a piece of agentic systems. XGrammar landing: [PR #10785](https://github.com/vllm-project/vllm/pull/10785). Roadmap issue then: [#8779](https://github.com/vllm-project/vllm/issues/8779).

Local figures (copyright remains with the original site; study copies):

![shogoth gpt](../../../../assets/vllm/blog/performance/struct-decode/01-shogoth-gpt.png)

![mermaid intro](../../../../assets/vllm/blog/performance/struct-decode/02-mermaid-intro.svg)

![constrained json fsm](../../../../assets/vllm/blog/performance/struct-decode/03-constrained-json-fsm.webp)

![vllm new xgrammar](../../../../assets/vllm/blog/performance/struct-decode/04-vllm-new-xgrammar.png)

![vllm xgrammar decode time per output token](../../../../assets/vllm/blog/performance/struct-decode/05-vllm-xgrammar-decode-time-per-output-token.png)

## Language models, briefly (the post’s frame)

Turing 1950 → two lineages:

1. **GOFAI** — rule-based expert systems (Haugeland; MYCIN-style traces). Did not scale to general tasks (“AI Winter”).
2. **NFAI** — connectionist / statistical nets (Rosenblatt → PDP hidden layers). Decoder-only transformers for text generation sit here.

GOFAI is deterministic: intentionality is programmed. NFAI is the black box: data-driven internal representations.

Figure 1 (Shogoth as GPTs): RLHF / post-training as injecting **rules** (a GOFAI move) into a large compound system.

## Why structured decoding

LLMs are good at “most probable continuation of this blob.” They are not obliged to emit valid JSON. Coherent prompts help; they do not *guarantee* a spec. Few-shot (“give me JSON like …”) still samples — invalid JSON is allowed. A JSON-finetuned specialist is expensive to train and eval.

**Structured / constrained / guided decoding** are the same mechanism: a schema **biases next-token probabilities** (usually logit masks) so sampling remains stochastic but the format holds. OpenAI JSON mode is the productized version. If you have built agents, function calling, or coding assistants on those APIs, you were likely already using this.

The post’s analogy: guided decoding is to LLMs what **validation** is to APIs. Dottxt also claimed it can sometimes **improve** native decode performance ([coalescence](https://blog.dottxt.co/coalescence.html)).

## How vLLM did it then

Give the engine a schema; it masks illegal tokens. [Outlines](https://github.com/dottxt-ai/outlines) proposed guided generation via an **FSM** over the schema (Willard & Louf, 2023): track state, apply logit bias. Figure 2 is the top-level flow; Figure 3 is the constrained-JSON FSM (via [LMSys compressed FSM, 2024](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).

In vLLM: pass a JSON schema in **sampling params** (Python SDK or HTTP).

### Then-current Outlines pain (V0)

1. **Slow Decode.** FSM is token-level: one state transition per step, one token at a time.
2. **Batch bottleneck.** Implementation was a **logit processor** on the sampling hot path ([then-current outlines_logits_processors.py](https://github.com/vllm-project/vllm/blob/80c751e7f68ade3d4c6391a0f3fce9ce970ddad0/vllm/model_executor/guided_decoding/outlines_logits_processors.py)). Compiling the FSM **per request** and computing the mask **synchronously** blocks **every** request in the batch → high TTFT, lower throughput. FSM compile itself was expensive, a large TTFT contributor.
3. **CFG mode.** JSON was relatively fast; CFG was much slower and could [crash the engine](https://github.com/vllm-project/vllm/issues/10081).
4. **No jump-forward.** [Jump-forward](https://lmsys.org/blog/2024-02-05-compressed-fsm/) wants to prefill *k* already-determined tokens. A logit processor only sees the *next* token.

## XGrammar

[XGrammar](https://github.com/mlc-ai/xgrammar) batches constrained decoding via a **PDA** — think a **collection of FSMs**, each a CFG. Recursion lets it take **multiple state transitions**. Extra grammar-compile optimizations: [MLC write-up](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar). Compilation moved out of Python into C / `pthread` (limitation 1). Groundwork for jump-forward later (limitation 4).

Figures 4–5 (Michael Goin, Red Hat): XGrammar vs Outlines; Decode time per output token. Up to **5×** TPOT under load.

V0 still implemented XGrammar as a [logit processor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/xgrammar_decoding.py) with a **tokenizer-data cache**. Gains were real; they still expected more.

**Then-current XGrammar gaps** (feature parity vs Outlines) — keep labeled as of the post:

- Grammars other than **GBNF** not yet ([vLLM PR](https://github.com/vllm-project/vllm/pull/10870))
- **Regex** not yet
- Complex JSON with **regex patterns or numeric ranges** not yet (bugfix [vLLM #10899](https://github.com/vllm-project/vllm/pull/10899), upstream [xgrammar #106](https://github.com/mlc-ai/xgrammar/pull/106))

vLLM then defaulted to **basic XGrammar**, and **fell back to Outlines** when XGrammar could not serve the request.

Also in-tree: **lm-format-enforcer**. Their tests: some **long-context** cases failed to enforce correct outputs; performance not up to Outlines.

## Tentative V1 plans (then-current)

1. **Scheduler-level** guided decoding — the scheduler knows which requests are structured, so they should **not block** others in the batch (limitation 2). Moves work off the hot path. Also a more natural home for jump-forward (limitation 4).
2. **One process computes the bitmask**, then **broadcast** to GPU workers, instead of repeating per worker. Bandwidth of broadcasting per-sample masks needed careful analysis.
3. Shared baseline for **speculative decoding** and **tool-use** — XGrammar planned tool-use so they could move off Python [tool parsers](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers). Tree scoring in spec decode could share the jump-forward API (depends on scheduler-level guided decoding).

Slack then: `#feat-structured-output`. Anatomy’s Structured Output Manager is that room once V1 existed.

Acknowledgements: vLLM team, XGrammar team, Aaron Pham (BentoML), Michael Goin (Red Hat), Chendi Xue (Intel), Russell Bryant (Red Hat).

Footnote-grade caveats worth keeping: structured/constrained/guided are interchangeable names for “sample under a format”; HuggingFace’s logits-processor zoo is the general mechanism; GOFAI footnotes (MYCIN rule traces, IBM alignment models, BoW-era stats, Attention / scaling laws, LSTM vs decoder-only) are history, not serving APIs.

Read with [speculative-decoding.md](../../features/speculative-decoding.md): that post is “guess the next tokens”; this one is “guessed tokens still have to fit the schema.”
