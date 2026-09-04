---
source: https://vllm.ai/blog/2024-10-17-spec-decode
lang: en
fetched: 2026-09-04
---

# How Speculative Decoding Boosts vLLM Performance by up to 2.8×

Chinese: [zh/vllm/blog/performance/spec-decode.md](../../../../zh/vllm/blog/performance/spec-decode.md)  
2024-10-17 Office Hours write-up. Flags and class names below are **that vintage** — today’s surface is [speculative-decoding.md](../../features/speculative-decoding.md) (`--speculative-config`). V1’s January 2025 alpha listed spec decode as **not yet supported** in V1; this post is the V0-era integration. Slides: [Google deck](https://docs.google.com/presentation/d/1wUoLmhfX6B7CfXy3o4m-MdodRL26WvY3/edit#slide=id.p1). Recording: [YouTube](https://youtu.be/eVJBFajJRIU). Office hours signup was via Neural Magic’s community page.

Local figures (copyright remains with the original site; study copies):

![figure8](../../../../assets/vllm/blog/performance/spec-decode/01-figure8.png)

![figure1](../../../../assets/vllm/blog/performance/spec-decode/02-figure1.png)

![figure9](../../../../assets/vllm/blog/performance/spec-decode/03-figure9.png)

![figure2](../../../../assets/vllm/blog/performance/spec-decode/04-figure2.png)

![figure3](../../../../assets/vllm/blog/performance/spec-decode/05-figure3.png)

![figure4](../../../../assets/vllm/blog/performance/spec-decode/06-figure4.png)

![figure5](../../../../assets/vllm/blog/performance/spec-decode/07-figure5.png)

![figure6](../../../../assets/vllm/blog/performance/spec-decode/08-figure6.png)

![figure7](../../../../assets/vllm/blog/performance/spec-decode/09-figure7.png)

![figure10](../../../../assets/vllm/blog/performance/spec-decode/10-figure10.png)

## Mechanism (Leviathan et al., 2023)

Paper: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). Small **draft** model proposes tokens; large **target** model verifies the sequence in **one** forward pass, keeps a matching prefix, and **corrects the first miss**. Lossless vs the target distribution.

Why latency can drop: vanilla Decode is autoregressive — T1, T2, T3 each need their own forward. Speculative decoding proposes several tokens and verifies them together.

Three beats:

1. **Draft model** proposes tokens one by one (cheap).
2. **Target** verifies that proposed span in a single forward; confirms hits, rewrites the first miss.
3. Several tokens commit per target pass when the draft is accurate enough that the verify forward amortizes.

Worked example (figure 08): draft proposes `["I", "like", "cooking", "and", "traveling"]`. Target says the third token should be `"playing"`. This step emits `["I", "like", "playing"]` — five drafted tokens, one target forward, three accepted (the miss is replaced, not dropped).

## How it sits in vLLM

Continuous batching stays: different requests share one batch.

Two runners:

- **Draft Runner** — small model, proposes candidates.
- **Target Runner** — large model, verifies.

Two engine pieces had to change:

1. **Scheduler** — multiple token slots in one forward, so propose + verify can share a step.
2. **Memory manager** — **two KV caches** (draft and target).

## Three draft styles (then-supported)

### 1. Separate draft model

Most common. Example they give: **Llama 68M** drafting for **Llama 2 70B**. Draft must be small enough not to dominate, accurate enough to pay for itself.

Hard constraint: draft and target **must share a vocabulary**. Llama 3-class models often had no well-matched small draft because of vocab mismatch — that is why the next two methods exist (no second weight file, or heads on the target).

### 2. Prompt lookup / n-gram

Also called n-gram matching. Works when the answer **copies the prompt** (summarization, some QA).

Figure 03: build all **2-grams** in the prompt as lookup keys; values are the **three tokens** that follow each key. At generation, if the current 2-gram hits a key, propose those following tokens. No second model.

### 3. Medusa / EAGLE / MLPSpeculator

Extra heads (or layers) on the **target** itself, proposing several future positions in one forward. No separate draft weights.

Figure 04 (from [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)): three heads, last transformer block as input. Head 1 proposes `["is", "'", "the"]` for position 1; head 2 `["difficult", "is", "'"]` for position 2; head 3 `["not", "difficult", "a"]` for position 3. They called this **preliminary**, more promising as kernels improved.

## When it helps, when it hurts

**Low QPS** (they plot **QPS = 1**), Llama-3-70B, **4×H100**:

- ShareGPT + draft `turboderp/Qwama-0.5B-Instruct`: up to **1.5×**.
- CNN/DailyMail + n-gram: up to **2.8×**.

**High QPS**, same 70B / 4×H100:

- ShareGPT: **1.4× slower**.
- CNN/DailyMail: **1.8× slower**.

When serving is already **compute-bound**, paying for propose + verify adds work to a full GPU. The extra forwards are a tax, not a gift.

## Roadmap then: dynamic speculative decoding

To make spec decode safe under load: **dynamic speculative decoding** ([arXiv:2406.14066](https://arxiv.org/abs/2406.14066); also [RFC #4565](https://github.com/vllm-project/vllm/issues/4565)). They called it an active vLLM research direction.

Rule of thumb in the post: **shorten the proposed length when system load is high**; shorten **less** when the average token **acceptance rate** is high (figure 10). Goal: auto-tune speculation degree every step so users can turn spec decode on without first guessing their QPS. **Not shipped in this post.**

## Then-current offline API

Docs of that vintage: [v0.6.0 spec_decode](https://docs.vllm.ai/en/v0.6.0/models/spec_decode.html). Constructor kwargs below — not today’s `--speculative-config` JSON.

**Draft model**, 5 speculative tokens:

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="facebook/opt-125m",
    num_speculative_tokens=5,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

**n-gram** (`speculative_model="[ngram]"`):

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
    ngram_prompt_lookup_min=1,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

**Draft TP smaller than target TP** — less communication on the cheap model. Example: target `tensor_parallel_size=4`, draft `speculative_draft_tensor_parallel_size=1`, draft `ibm-fms/llama3-70b-accelerator`:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_model="ibm-fms/llama3-70b-accelerator",
    speculative_draft_tensor_parallel_size=1,
)
outputs = llm.generate("The future of AI is")

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

They expected later work (same paper + RFC) to pick `num_speculative_tokens` automatically.

## Conclusion in the post

Spec decode: large wins at **low QPS**. Dynamic length was the bet for **high QPS**. This note’s job is the inequality: **low QPS looks like magic, high QPS looks like tax** — until the proposal length can move with load and acceptance rate.
