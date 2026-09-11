---
source: https://vllm.ai/blog/2025-01-14-struct-decode-intro
lang: en
fetched: 2026-09-05
---

# Structured Decoding in vLLM: a gentle introduction

Chinese: [zh/vllm/blog/performance/struct-decode.md](../../../../zh/vllm/blog/performance/struct-decode.md)  
Source: https://vllm.ai/blog/2025-01-14-struct-decode-intro

2025-01-14. Guest post: BentoML and Red Hat. Study extract, not an official reprint. **“V1 upcoming” in the post is historical** — most of that roadmap later landed. JSON mode, function calling, and agent tool arguments are usually this underneath. Landing [PR #10785](https://github.com/vllm-project/vllm/pull/10785). Roadmap issue then: [#8779](https://github.com/vllm-project/vllm/issues/8779).

**TL/DR:**

- Structured decoding controls output *format* while sampling stays sampling
- vLLM then supported both [outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar)
- Recent XGrammar integration: up to **5×** better TPOT under load
- Then-planned V1: performance + **scheduler-level** mask broadcasting for mixed batches

[vLLM](https://blog.vllm.ai/2023/06/20/vllm.html) is the high-throughput inference engine. This post walks an annotated history of language models, the then-current state of structured decoding in vLLM, the [XGrammar](https://github.com/vllm-project/vllm/pull/10785) landing, and a then-tentative roadmap.

The authors also ask you to read it philosophically: structured decoding as a shift in how we treat LLM outputs, and as a piece of complex agentic systems.

More: then-current [vLLM docs](https://docs.vllm.ai/en/latest/).

Local figures (copyright remains with the original site; study copies):

![shogoth gpt](../../../../assets/vllm/blog/performance/struct-decode/01-shogoth-gpt.png)

**Caption (original).** Shogoth as GPTs. In a sense, RLHF or any post-training injects rules (a GOFAI move) into a large compound AI system.

![mermaid intro](../../../../assets/vllm/blog/performance/struct-decode/02-mermaid-intro.svg)

**Caption (original).** Top-level view of structured decoding.

![constrained json fsm](../../../../assets/vllm/blog/performance/struct-decode/03-constrained-json-fsm.webp)

**Caption (original).** Constrained-JSON FSM. Courtesy of [LMSys, 2024](https://lmsys.org/blog/2024-02-05-compressed-fsm/).

![vllm new xgrammar](../../../../assets/vllm/blog/performance/struct-decode/04-vllm-new-xgrammar.png)

![vllm xgrammar decode time per output token](../../../../assets/vllm/blog/performance/struct-decode/05-vllm-xgrammar-decode-time-per-output-token.png)

**Caption (original).** XGrammar vs Outlines; decode time per output token. Courtesy of Michael Goin (Red Hat).

## Language models: A brief historical context

In 1950 Alan Turing argued that a high-speed digital computer, programmed with rules, could show emergent intelligence (Turing, 1950). Two lineages followed:

1. **Good Old-Fashioned AI (GOFAI):** 1950s expert systems meant to copy a human specialist’s decisions (symbolic reasoning). Haugeland’s name (Haugeland, 1997). Semantic representation did not scale to general tasks; funding collapsed — the “AI Winter” (Hendler, 2008). Footnote: Newell and Simon at RAND showed computers could simulate important aspects of intelligence. In medicine, Stanford’s 1970s MYCIN advised on blood infections (Shortliffe, 1974) and used “rule traces” so humans could follow the reasoning.

2. **New-Fangled AI (NFAI):** Donald Norman’s Parallel Distributed Processing group (Rumelhart et al., 1986) added **hidden layers** to Rosenblatt’s perceptron (Rosenblatt, 1958) to extrapolate responses from training. These connectionist nets sat on statistical methods. Data plus Moore’s Law compute later dominated research and production — especially **decoder-only** transformers for **text generation**. Most modern transformer variants are treated as **NFAI**.

Statistical footnote: 1990s IBM alignment models for machine translation; around 2001 Bag-of-words variants on 0.3B tokens were SOTA (Mikolov et al., 2013) — statistical models captured corpus patterns better than symbols. 2017’s “Attention Is All You Need” (Vaswani et al., 2023) built Transformers on attention (Bahdanau et al., 2016); OpenAI’s scaling laws (Kaplan et al., 2020) started the foundation-model race. Before attention, seq-to-seq used RNNs / LSTM (Hochreiter & Schmidhuber, 1997) for longer context, but gradients and long-range memory were weak. Attention encodes positional data; the paper also proposed encoder–decoder, yet most text-generation models today are decoder-only (better zero-shot). Transformers scale and are hardware-aware — you cannot just stack LSTM blocks and hope for long-term retention.

In short:

- GOFAI is **deterministic** and rule-based: intentionality is programmed
- NFAI is often the black box (in: input — out: some output), data-driven, internal representations grown by the network

## Why do we need structured decoding?

LLMs are good at this heuristic: given a blob of text, emit the most probable continuation. A Wikipedia article should continue like the rest of that article.

The assumption: the prompt is coherent and well structured around the user’s goal. In other words, LLMs can be unpredictable when you need a specific format. Ask for JSON — without guidance you may get readable text that breaks the JSON spec.

Few-shot (“give me JSON like …”) still samples; invalid JSON is allowed. A JSON-finetuned specialist is expensive to train, monitor, and eval.

Structured decoding is the fence: the model follows a desired structure while **keeping the system’s non-determinism** — sampling stays sampling.

Companies like OpenAI shipped [JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode) to constrain output format. If you have built agents, function calling, or coding assistants on those APIs, you were likely already using structured decoding.

**Structured / constrained / guided decoding** are the same mechanism: a format so the model **samples under a structure**.

> Guided decoding is to LLMs what **validation** is to APIs — a guarantee that what comes out matches what you expect. Structure lets developers wire LLMs into applications.

Dottxt also claimed it can sometimes [improve](https://blog.dottxt.co/coalescence.html) native decode performance.

## Structured decoding and vLLM

In simple terms, structured decoding gives the LLM a “template.” Users provide a schema that “influences” the output so it stays compliant.

Technically, an inference engine can bias next-token probabilities (often via logit masks) from a schema. [outlines](https://github.com/dottxt-ai/outlines) proposed guided generation via a finite-state machine (FSM) (Willard & Louf, 2023): track state during decode, apply logit bias, filter illegal tokens.

In vLLM: pass a JSON schema in **sampling params** (Python SDK or HTTP).

### Previous limitations in vLLM

Then-current Outlines backend pain:

1. **Slow decoding.** The FSM is built token-level: one state transition per step, so one token at a time.
2. **Batch bottlenecks.** Implementation leaned on a logit processor ([then-current `outlines_logits_processors.py`](https://github.com/vllm-project/vllm/blob/80c751e7f68ade3d4c6391a0f3fce9ce970ddad0/vllm/model_executor/guided_decoding/outlines_logits_processors.py)) on the sampling hot path. Compiling the FSM **per request** and computing the mask **synchronously** blocks **every** request in the batch → high TTFT, lower throughput. FSM compile itself was expensive, a large TTFT contributor. HuggingFace’s [logits-processor zoo](https://huggingface.co/blog/logits-processor-zoo) is the general valve.
3. **CFG mode performance.** JSON was relatively fast; CFG was much slower and could [crash the engine](https://github.com/vllm-project/vllm/issues/10081).
4. **Limited advanced features.** [Jump-forward decoding](https://lmsys.org/blog/2024-02-05-compressed-fsm/) was not possible: it wants to prefill *k* already-determined tokens; a logit processor only sees the *next* token.

### Integration with XGrammar

[XGrammar](https://github.com/mlc-ai/xgrammar) batches constrained decoding via a pushdown automaton (PDA). Think of a PDA as a “collection of FSMs, each a context-free grammar (CFG).” Recursion lets it take multiple state transitions. Extra grammar-compile [optimizations](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

This hits **limitation (1)**: compilation moved out of Python into C via `pthread`. It also lays groundwork for **limitation (4)**. Figures 4–5: XGrammar vs Outlines; up to **5×** TPOT under load.

In v0, XGrammar was still a [logit processor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/xgrammar_decoding.py) with a tokenizer-data cache. Gains were real; they still expected more.

Then-current XGrammar v0 gaps vs full feature parity (keep labeled as of the post):

- Grammars other than **GBNF** not yet ([vLLM PR](https://github.com/vllm-project/vllm/pull/10870))
- **Regex** not yet
- Complex JSON with regex patterns or numeric ranges not yet ([vLLM #10899](https://github.com/vllm-project/vllm/pull/10899), upstream [xgrammar #106](https://github.com/mlc-ai/xgrammar/pull/106))

> vLLM then had basic XGrammar by default. When XGrammar could not serve the request, it fell back to Outlines.
>
> Also in-tree: lm-format-enforcer. Their tests: some long-context cases failed to enforce correct outputs; performance not up to Outlines.

## Tentative plans for v1

With [v1](https://github.com/vllm-project/vllm/issues/8779) then on the horizon, a tentative structured-decoding plan:

1. Move guided decoding to the **scheduler**:
   - The scheduler knows which requests are structured, so they should **not block** others in the batch (limitation **(2)**). Off the hot path.
   - More natural vertical integration with jump-forward (limitation **(4)**).
2. Compute the bitmask **in one process**, then **broadcast** to GPU workers instead of repeating per worker.
   - Bandwidth of broadcasting per-sample masks for guided requests needed careful analysis.
3. Shared baseline for **speculative decoding** and **tool-use**:
   - XGrammar planned tool-use so they could leave Python [tool parsers](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers).
   - Tree scoring in spec decode could share the jump-forward API (depends on scheduler-level guided decoding).

_NOTE: suggestions welcome. Slack then: `#feat-structured-output` (see the post’s [vLLM slack](https://www.notion.so/bentoml/slack.vllm.ai) pointer)._

## Acknowledgements

The vLLM team, the XGrammar team, [Aaron Pham (BentoML)](https://github.com/aarnphm), [Michael Goin (Red Hat)](https://github.com/mgoin), [Chendi Xue (Intel)](https://github.com/xuechendi), and [Russell Bryant (Red Hat)](https://github.com/russellb) for bringing XGrammar to vLLM and continuing structured-decoding work.

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2016). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv preprint arXiv:1409.0473
- Haugeland, J. (1997). *Mind Design II: Philosophy, Psychology, and Artificial Intelligence*. The MIT Press. <https://doi.org/10.7551/mitpress/4626.001.0001>
- Hendler, J. (2008). Avoiding Another AI Winter. *IEEE Intelligent Systems*, *23*(2), 2–4. <https://doi.org/10.1109/MIS.2008.20>
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv preprint arXiv:2001.08361
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, *65*(6), 386–408. <https://doi.org/10.1037/h0042519>
- Rumelhart, D. E., McClelland, J. L., & Group, P. R. (1986). *Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition: Foundations*. The MIT Press. <https://doi.org/10.7551/mitpress/5236.001.0001>
- Shortliffe, E. H. (1974). *MYCIN: A Rule-Based Computer Program for Advising Physicians Regarding Antimicrobial Therapy Selection* (Technical Report STAN-CS-74-465). Stanford University.
- Statistical Machine Translation. (n.d.). *IBM Models*. <http://www2.statmt.org/survey/Topic/IBMModels>
- Turing, A. M. (1950). i.—Computing Machinery And Intelligence. *Mind*, *LIX*(236), 433–460. <https://doi.org/10.1093/mind/LIX.236.433>
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762
- Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv preprint arXiv:2307.09702

Anatomy’s Structured Output Manager is that room once V1 existed. Read with [speculative-decoding.md](../../features/speculative-decoding.md): that post is “guess the next tokens”; this one is “guessed tokens still have to fit the schema.”
