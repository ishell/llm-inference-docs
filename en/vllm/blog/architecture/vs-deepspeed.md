---
source: https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed
lang: en
fetched: 2026-09-04
---

# Notes on vLLM vs DeepSpeed-FastGen

2023-11-14 reply to DeepSpeed’s “2× vs vLLM” post. Snapshot, not a 2026 bake-off. Chinese: [zh/vllm/blog/architecture/vs-deepspeed.md](../../../../zh/vllm/blog/architecture/vs-deepspeed.md)

The DeepSpeed team published a [FastGen blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen) claiming **2× throughput** over vLLM via **Dynamic SplitFuse**. vLLM’s note is a public counter: they are glad the community is shipping new tricks; they also say the SplitFuse win is **narrow**, and that on most workloads vLLM is faster or tied.


Local figures (copyright remains with the original site; study copies):

![s1](../../../../assets/vllm/blog/architecture/vs-deepspeed/01-s1.png)

![s2](../../../../assets/vllm/blog/architecture/vs-deepspeed/02-s2.png)

## TL;DR

- vLLM matches DeepSpeed-FastGen in common scenarios and **surpasses it when outputs are long**.
- FastGen only outperforms vLLM on **long prompt + short output**, because of Dynamic SplitFuse. That optimization was **on vLLM’s roadmap** in this post (the later everyday name is **chunked prefill**).
- vLLM’s mission, as written then: the fastest and easiest-to-use **open-source** LLM inference and serving engine. **Apache 2.0**, community-owned, broad model and optimization support.

## Two performance differences they named

1. **FastGen’s KV allocation is conservative / suboptimal.** Waste shows up when output lengths are large: reserved KV that PagedAttention would have given back to the batch.
2. **Dynamic SplitFuse scheduling speeds up only when prompt lengths are much greater than output lengths** (ISL ≫ OSL). Prefill is chopped and fused into the decode stream so a long prompt cannot stall the batch; that is the same idea as today’s `max_num_batched_tokens` / chunked prefill.

Consequence they draw: FastGen looks good when the workload is *consistently* long prompt and short output. In other scenarios, vLLM is faster.

Hardware for the published bars: **NVIDIA A100-80GB**, **LLaMA-7B**.

### Scenario 1: long prompt, short output

This is where SplitFuse is *expected* to shine. The gain they measure is **not** the advertised 2×.

Read from the published bar chart (`prompt_len=2600`):

| output_len | vLLM (reqs/s) | DeepSpeed-FastGen (reqs/s) |
|---|---|---|
| 60 | 3.52 | 3.7 |
| 128 | 2.68 | 2.76 |
| 200 | 2.13 | 2.13 |

FastGen is slightly ahead at the shortest outputs; the gap is gone by `output_len=200`. Both systems slow down as generation length grows — that is decode + KV, not a SplitFuse miracle.

### Scenario 2: other cases

Here vLLM is up to **1.8×** faster. Published bars (`prompt_len=500`):

| output_len | vLLM (reqs/s) | DeepSpeed-FastGen (reqs/s) | approx. ratio |
|---|---|---|---|
| 150 | 10.03 | 7.42 | ~1.35× |
| 500 | 3.43 | 1.97 | ~1.74× |
| 1024 | 1.49 | 0.81 | ~1.84× |

The longer the output, the more FastGen’s conservative KV reservation hurts, and the closer the ratio gets to the “up to 1.8×” claim.

Benchmark code they pointed at then: [`benchmarks/benchmark_throughput.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py). Questions and suggestions: the [vLLM GitHub](https://github.com/vllm-project/vllm). These numbers are a **November 2023** snapshot.

## Community mission (the paragraph that is not a score)

Coming out of UC Berkeley **Sky Computing Lab**, they want vLLM to take the community’s best models, optimizations, and hardware. Priorities named in this post:

- system performance
- new features: **LoRA**, **speculative decoding**, better **quantization**
- hardware collaborations: **AMD**, **AWS Inferentia** (the post misspells it “Inferenetia”), **Intel Habana**

Specifically for Dynamic SplitFuse: they were “actively investigating the proper integration.” Read after the [PagedAttention launch](paged-attention.md): the 2023 fight was no longer “whether to page KV,” but **whether to split Prefill**. That split is `max_num_batched_tokens` / chunked prefill in today’s [optimization](../../../optimization/optimization.md) page, and it is default in V1 ([v1-alpha.md](v1-alpha.md), [anatomy.md](anatomy.md)).

## Appendix: feature comparison (Nov 2023 capsule)

DeepSpeed-FastGen then offered **basic** functionality: **three** model types, and it lacked popular features such as **stop strings** and **parallel sampling** (e.g. beam search). The vLLM authors expected FastGen to catch up and welcomed the competition.

|  | vLLM | DeepSpeed-FastGen |
|---|---|---|
| Runtime | Python/PyTorch | Python/PyTorch |
| Model implementation | HuggingFace Transformers | Custom implementation + converter for HF models |
| Server frontend | Simple FastAPI server for demo purposes | Custom gRPC-based server |
| Scheduling | Continuous batching | Dynamic SplitFuse |
| Attention kernel | PagedAttention & FlashAttention | PagedAttention & FlashAttention |
| Custom kernels (for LLaMA) | Attention, RoPE, RMS, SILU | Attention, RoPE, RMS, SILU, Embedding |
| KV Cache allocation | Near-optimal | Suboptimal/conservative |
| Supported models | 16 different architectures | LLaMA, Mistral, OPT |
| Sampling methods | Random, parallel, beam search | Random |
| Stop criterion | Stop strings, stop tokens, EOS | EOS |

Both already ran PagedAttention & FlashAttention. The remaining argument was scheduling (continuous batching vs SplitFuse) and how greedily you pack KV. Do not use this table as a 2026 feature matrix for either project.
