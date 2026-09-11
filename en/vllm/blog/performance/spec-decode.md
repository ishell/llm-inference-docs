---
source: https://vllm.ai/blog/2024-10-17-spec-decode
lang: en
fetched: 2026-09-05
---

# How Speculative Decoding Boosts vLLM Performance by up to 2.8×

Chinese: [zh/vllm/blog/performance/spec-decode.md](../../../../zh/vllm/blog/performance/spec-decode.md)  
Source: https://vllm.ai/blog/2024-10-17-spec-decode

2024-10-17. **vLLM Team.** Study extract, not an official reprint. From biweekly Office Hours. Flags and class names below are **that vintage** — today’s surface is [speculative-decoding.md](../../features/speculative-decoding.md) (`--speculative-config`). V1’s January 2025 alpha listed spec decode as **not yet supported** in V1; this post is the V0-era integration. Slides: [Google deck](https://docs.google.com/presentation/d/1wUoLmhfX6B7CfXy3o4m-MdodRL26WvY3/edit#slide=id.p1). Recording: [YouTube](https://youtu.be/eVJBFajJRIU). Signup then: [Neural Magic community office hours](https://neuralmagic.com/community-office-hours/?utm_campaign=vLLM%20Office%20Hours&utm_source=vllm-blog).

Speculative decoding in vLLM uses a small model and a large model together to accelerate token generation. This post: how it works in vLLM, and what performance it bought.

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

## An Introduction to Speculative Decoding

Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) cuts latency during token generation. Smaller models handle simpler predictions; larger models verify or adjust. Faster generation without sacrificing accuracy — **lossless** vs the target distribution.

**Why can latency drop?** Vanilla LLMs generate autoregressively: given a prompt, T1, T2, T3 each need their own forward. Speculative decoding proposes several tokens and verifies them in one forward.

Three beats:

1. **Draft Model:** a smaller, cheaper model proposes tokens one by one.
2. **Target Model Verification:** the larger model verifies that span in **one** forward. Confirms hits, corrects the first miss.
3. **Multiple Tokens in One Pass:** not one token per pass; several tokens can commit from one forward, so latency falls.

**Figure 08.** Draft proposes `["I", "like", "cooking", "and", "traveling"]`, forwarded to the target for parallel verification. The third token `"cooking"` should be `"playing"`. This step emits `["I", "like", "playing"]` — five drafted tokens, one target forward, three accepted (the miss is replaced, not dropped).

They framed this as useful for both small-scale and large-scale deployments.

## How Speculative Decoding Works in vLLM

Continuous batching stays: different requests share one batch so throughput can rise. Two components:

- **Draft Runner** — small model, proposes candidates.
- **Target Runner** — large model, verifies.

Wired so speculative decoding runs with continuous batching.

**Figure 01.** How draft and target runners meet inside vLLM batching.

Two engine pieces had to change:

1. **Scheduler** — multiple token slots in one forward, so propose + verify can share a step.
2. **Memory Manager** — **two KV caches** (draft and target).

**Figure 09.** System architecture of speculative decoding in vLLM.

## Types of Speculative Decoding Supported in vLLM

Three types then, for different workloads.

### Draft Model-Based Speculative Decoding

Most common: smaller model predicts the next tokens, larger model verifies. Example: **Llama 68M** drafting for **Llama 2 70B**. Draft must be small enough not to dominate, accurate enough to pay for itself. Choosing the draft is the efficiency lever.

Choosing is hard. Llama 3-class models often had no well-matched small draft: draft and target **must share a vocabulary**. Vocab mismatch can block this path. That is why the next two methods exist (no second weight file, or heads on the target).

**Figure 02.** The separate-draft path.

### Prompt Lookup Decoding

Also called n-gram matching. Works when the answer **copies the prompt** (summarization, some QA). Instead of a small model, speculate from information already in the prompt. Especially useful when the large model repeats parts of the prompt in its answers.

**Figure 03.** Build all **2-grams** in the prompt as lookup keys; values are the **three tokens** that follow each key. At generation, if the current 2-gram hits a key, propose those following tokens.

### Medusa / EAGLE / MLPSpeculator

Extra layers (or heads) on the **target** itself, proposing several future positions in one forward. No separate draft weights — the large model’s own capacity for parallel token generation. They called this **preliminary**, more promising as kernels improved.

**Figure 04** from [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa): three heads, last transformer block as input. Head 1 proposes `["is", "'", "the"]` for position 1; head 2 `["difficult", "is", "'"]` for position 2; head 3 `["not", "difficult", "a"]` for position 3.

## Speculative Decoding Performance Insights: Speedups and Trade-offs

Large wins at **low QPS**. On ShareGPT, draft-model speculative decoding: up to **1.5×** token generation. On summarization (CNN/DailyMail), prompt lookup: up to **2.8×**.

**Figures 05–06.** **QPS = 1**, Llama-3-70B, **4×H100**: ShareGPT + draft `turboderp/Qwama-0.5B-Instruct` up to **1.5×**; CNN/DailyMail + n-gram up to **2.8×**.

At **high QPS**, the extra compute to propose and verify can hurt — the system is already **compute-bound**, request rate rises, the tax outweighs the gift.

**Figure 07.** High QPS: ShareGPT **1.4× slower**; CNN/DailyMail **1.8× slower** (same 70B / 4×H100).

## On the Roadmap: Dynamic Adjustments for Better Performance

To make spec decode safe under load: **dynamic speculative decoding**. Paper: [arXiv:2406.14066](https://arxiv.org/abs/2406.14066). They called it an active vLLM research direction. Adjust the number of speculative tokens from system load and draft accuracy. Rule of thumb: **shorten the proposed length when system load is high**; shorten **less** when the average token **acceptance rate** is high (figure 10).

Later: auto-tune speculation degree every step so users can turn spec decode on without first guessing their QPS. **Not shipped in this post.**

## How to Use Speculative Decoding in vLLM

They said: when launching the vLLM server, include flags for the speculative model, token count, and tensor parallel size. The three snippets below are **offline** `LLM(...)` constructors — not today’s `--speculative-config` JSON. Docs of that vintage: [v0.6.0 spec_decode](https://docs.vllm.ai/en/v0.6.0/models/spec_decode.html).

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

Sometimes the draft should use a smaller tensor-parallel size: fewer resources, less communication, heavy work left to the target. Example: target `tensor_parallel_size=4`, draft `speculative_draft_tensor_parallel_size=1`, draft `ibm-fms/llama3-70b-accelerator`:

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

Later work (same paper + [RFC #4565](https://github.com/vllm-project/vllm/issues/4565)) would pick `num_speculative_tokens` automatically.

Docs pointer then: the spec_decode page above. Questions and feedback: biweekly office hours.

## Conclusion: The Future of Speculative Decoding in vLLM

Spec decode: large wins at **low QPS**. Once dynamic length landed, they wanted it useful at high QPS too — lower latency, higher efficiency, a tool you leave on.

This note’s job is the inequality: **low QPS looks like magic, high QPS looks like tax** — until the proposal length can move with load and acceptance rate.
