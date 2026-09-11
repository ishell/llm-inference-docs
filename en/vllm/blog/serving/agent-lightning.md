---
source: https://vllm.ai/blog/2025-10-22-agent-lightning
lang: en
fetched: 2026-09-04
---

# `return_token_ids`: stop retokenizing agent RL

Chinese: [zh/vllm/blog/serving/agent-lightning.md](../../../../zh/vllm/blog/serving/agent-lightning.md)

2025-10-22. **The Agent Lightning (AGL) Team.** vLLM **≥ 0.10.2**. PR into core: [#22587](https://github.com/vllm-project/vllm/pull/22587). Docs: [OpenAI-compatible server (v0.10.2)](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html#api-reference). Project: [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning), [docs](https://microsoft.github.io/agent-lightning/latest/). Weight pause / sync that the trainer still needs: [native-rl.md](native-rl.md). Study note.

**TL;DR.** Agents call LLMs through OpenAI-compatible endpoints that used to return **strings only**. In **agent RL** that becomes **Retokenization Drift**: detokenize at inference, retokenize at training; IDs can differ while the strings match. Ask vLLM to return the **exact token IDs** for prompt and completion: `"return_token_ids": true` on `/v1/chat/completions` or `/v1/completions` → `prompt_token_ids` and `token_ids` beside the text. Agent Lightning treats each model call as its own sample — no stitching a trajectory — and logs those IDs.

## Why token IDs matter for Agent RL

LLM RL trains on token sequences, so the trainer needs the IDs the behavior policy actually sampled. Single-turn used to be easy: vLLM’s low-level `generate` already returns tokens.

Agent stacks prefer OpenAI-style `chat.completions` / `completions` over raw `generate`: chat templating and roles (system / user / assistant), tool / function calling, structured outputs. Those APIs historically returned **strings only**. Stored text had to be retokenized for training. That is unstable in practice: **retokenization drift**.

Symptoms: wiggly learning curves, and a hard-to-debug gap between the data you think you optimized and what the model sampled.

Local figures (copyright remains with the original site; study copies):

![1 rewards](../../../../assets/vllm/blog/serving/agent-lightning/01-1_rewards.png)

Red and blue: same recipe — store text, retokenize at train time. Yellow: keep tokens from the inference engine.

Three usual forks:

**Non-unique “HAVING”.** A word is sampled as two tokens (`H` + `AVING`) and retokenized later as another split (`HAV` + `ING`). Same text, different IDs; the learner optimizes the wrong sequence.

![2 having](../../../../assets/vllm/blog/serving/agent-lightning/02-2_having.png)

The word “HAVING” maps to different tokens.

**Tool-call serialization.** Generated text like `<tool_call>{ "name": ... }</tool_call>` is parsed into the object the chat-completion API wants, then rendered back and retokenized. Parsing / re-rendering changes whitespace and formatting. The parser may even **auto-correct JSON errors**, hiding the model’s real mistakes so they never train away.

**Chat template difference.** Templates are not unique. One LLaMA can have several templates in [vLLM examples](https://github.com/vllm-project/vllm/tree/1d165d6d859d3c50720f0c07209db2363c4fd33b/examples) and another on [HuggingFace](https://huggingface.co/meta-llama). Inference and training on different frameworks → different IDs for the whole sequence. One extra space is enough.

Those three produce retokenization drift, then training instability: inference and training disagree, so updates are **off-policy**. On-policy is load-bearing for stable RL; this off-policy gap is **not even at token level**, so token-level importance sampling cannot patch it.

The alternative is to save the IDs the model emitted, as single-turn already does. That requires talking to the engine at token level. Most agents — LangChain stacks especially — only speak OpenAI-compatible APIs and cannot tokenize / detokenize themselves. Longer discussion: [Token IDs and why they matter](https://microsoft.github.io/agent-lightning/stable/deep-dive/serving-llm/#token-ids-and-why-they-matter).

## Solution and new feature

An **OpenAI-compatible API that returns token IDs**. Agent Lightning and vLLM landed this in [vLLM core](https://github.com/vllm-project/vllm/pull/22587). From **vLLM v0.10.2**, [`return_token_ids`](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html#api-reference) is a first-class request field. Set it `true` and the response grows two fields:

- `prompt_token_ids` — input IDs **after** chat-template processing
- `token_ids` — completion IDs, via `completion.choices`

Everything else stays OpenAI-compatible; existing clients keep working.

## Introduction to Agent Lightning (v0.2)

[Agent Lightning](https://github.com/microsoft/agent-lightning) (AGL) v0.1 already sold a flexible RL trainer for **any** agent:

- Integration with existing agents with **almost zero code change**.
- Any agent framework (LangChain, OpenAI Agent SDK, Microsoft Agent Framework, …) or **none** (plain Python).
- No constraint on what goes into the LLM: summarization, multi-agent collaboration, other orchestration.

At first release they shipped an [instrumented vLLM server](https://github.com/microsoft/agent-lightning/blob/v0.1/agentlightning/instrumentation/vllm.py) that **monkey-patched** vLLM’s OpenAI server to return token IDs. Now AGL **automatically adds** `return_token_ids` on each request. [Tracing](https://microsoft.github.io/agent-lightning/latest/tutorials/traces/) then collects what the trainer needs, including those IDs.

## The middleware for agent optimization

From v0.2 the role is sharper: a durable middleware layer and standardized data protocols for agent optimization, especially agent RL.

![3 agl](../../../../assets/vllm/blog/serving/agent-lightning/03-3_agl.png)

Conceptual overview of Agent Lightning.

Modular pieces, talking through those protocols:

- **Agent Runner** — runs agents on assigned tasks. Receives tasks, delegates, collects results and intermediate data, reports into the store. Separate from the LLM side, so it can live on **CPU** and scale out many concurrent agents.
- **Algorithm (Model Trainer)** — hosts the LLMs for inference and training. Owns the RL loop: task sampling, rollout management, model updates from experience. Typically **GPU**; talks to the Runner asynchronously through the shared schemas.
- **[Data Store](https://microsoft.github.io/agent-lightning/latest/how-to/write-first-algorithm/#the-central-hub-the-lightningstore)** — central hub. Standardized interfaces and unified schemas so heterogeneous pieces interoperate. Algorithm and Runner communicate **indirectly**. Example: Algorithm delegates via [`rollouts`](https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/#rollout); the Runner reports traces as [`spans`](https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/#span).

![4 tasks spans loop](../../../../assets/vllm/blog/serving/agent-lightning/04-4_tasks-spans-loop.svg)

The training loop: tasks out, spans back.

Under a store-centric design, every training iteration is two steps: collect agent-running data (spans) into the store; pull what the algorithm needs and train.

That split buys algorithmic flexibility. Collection can use [various tracers](https://microsoft.github.io/agent-lightning/latest/tutorials/traces/) or [emit custom messages](https://microsoft.github.io/agent-lightning/latest/tutorials/write-agents/#emitting-rewards-messages-and-more) — different rewards, any intermediate variable. The algorithm side [queries spans](https://microsoft.github.io/agent-lightning/latest/deep-dive/birds-eye-view/?h=query#putting-it-all-together-a-reinforcement-learning-example-verl) and runs them through [adapters](https://microsoft.github.io/agent-lightning/latest/deep-dive/birds-eye-view/#adapter).

Same frame covers [algorithm customizations](https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/verl/#customization) (credit assignment, auxiliary models on partial data, training-time data adjustment) and other algorithms: [automatic prompt tuning (APO)](https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/), [filter high-reward data and fit with Unsloth](https://microsoft.github.io/agent-lightning/latest/how-to/unsloth-sft/).

Second claim: modular boundaries cut system complexity and let components use **different** resources. An agent RL stack already mixes agent frameworks (LangChain, MCP), inference (vLLM), and trainers (Megatron-LM). Coupled, that heterogeneity is a tax. Decoupled: the agent side can take CPU; inference and training take GPU; each scales horizontally on its own.

More on the page:

- [Full docs](https://microsoft.github.io/agent-lightning/latest/)
- [Birds-eye view](https://microsoft.github.io/agent-lightning/latest/deep-dive/birds-eye-view/)
- [Train a SQL agent (multi-agent) with verl](https://microsoft.github.io/agent-lightning/latest/how-to/train-sql-agent/)
- [Train a room-selector agent with APO](https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/), prompts via [POML](https://github.com/microsoft/poml/)
- [Train a math agent (OpenAI Agents SDK + MCP) with Unsloth](https://microsoft.github.io/agent-lightning/latest/how-to/unsloth-sft/)

## Acknowledgements

vLLM maintainers: [Kaichao You](https://github.com/youkaichao), [Nick Hill](https://github.com/njhill), [Aaron Pham](https://github.com/aarnphm), [Cyrus Leung](https://github.com/DarkLight1337), [Robert Shaw](https://github.com/robertgshaw2-redhat), [Simon Mo](https://github.com/simon-mo). Agent Lightning is an open-source project from Microsoft Research; thanks to MSR for backing that exploration. [Yuge Zhang](https://github.com/ultmaster) is named as the primary contributor.
