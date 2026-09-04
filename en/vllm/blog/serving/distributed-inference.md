---
source: https://vllm.ai/blog/2025-02-17-distributed-inference
lang: en
fetched: 2026-09-04
---

# Distributed Inference with vLLM

Chinese: [zh/vllm/blog/serving/distributed-inference.md](../../../../zh/vllm/blog/serving/distributed-inference.md)

2025-02-17. Same map as a TensorRT-LLM sharding chapter: **communication** is the constraint. Two answers to CUDA OOM: lower precision (FP8 / low-bit — accuracy and scale still bite past hundreds of billions of parameters), or **cut the model across GPUs/nodes**. Quantization alone does not save 100B+ models.

Inference is not training: shapes change, latency matters, and you still have to house KV, speculative decoding, and the Prefill→Decode handoff. Training can hide communication inside huge static steps; inference may admit a new batch every step. vLLM's knives in this post are mainly **in-node tensor parallelism (TP)** and **cross-node pipeline parallelism (PP)**, plus communication kernels and a control plane that tries not to let CPU stall GPU. Expert parallelism is a **forward look** here; it becomes the main course in [large-scale.md](large-scale.md).

Local figures (copyright remains with the original site; study copies):

![tp strategies](../../../../assets/vllm/blog/serving/distributed-inference/01-tp_strategies.png)

![column row parallel](../../../../assets/vllm/blog/serving/distributed-inference/02-column_row_parallel.png)

![tensor parallelism](../../../../assets/vllm/blog/serving/distributed-inference/03-tensor_parallelism.png)

![kv cache effects](../../../../assets/vllm/blog/serving/distributed-inference/04-kv_cache_effects.png)

## Tensor parallelism

When a model will not fit one GPU, TP **shards weights** so several GPUs compute concurrently. Lineage: [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053), adapted for inference.

Two primitives:

1. **Column parallelism** — split weight columns, concatenate after the matmul.
2. **Row parallelism** — split rows, **sum / all-reduce** after.

Llama MLP postcard: column-parallel **up-projection** → elementwise **SILU** on shards → row-parallel **down-projection** + **all-reduce**. Splitting weights multiplies memory bandwidth, which helps memory-bound Decode — if the corridor is fast (**NVLink / InfiniBand**). A slow corridor eats the win.

Tensor-parallelism figure source named on the page: [Sebastian Raschka, 2023](https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html).

MLA-style models later make naive TP a bad idea (duplicated latents). That is the Wide-EP argument: EP + DP Attention instead of more TP.

## Pipeline parallelism

When even a multi-GPU **node** is not enough (DeepSeek R1, Llama 3.1 405B), PP **cuts contiguous layers** across nodes. Activations **send/recv** once per stage — cheaper than TP all-reduce. PP **does not inherently cut latency**; idle bubbles are filled with **pipeline scheduling / micro-batches** so every GPU stays busy.

## Combining TP and PP

Rule of thumb (almost the TRT-LLM sentence):

- Slow inter-node fabric → **TP inside the node, PP between nodes**.
- Fast NVLink / IB → **TP may cross nodes**.
- Combine them to avoid paying the wrong communication tax, not to fill a slide with every letter.

## Super-linear KV rooms

2 GPUs is not 2×. After the split, KV room per GPU can grow **faster than linear**, so batches can grow (better locality, higher util). Figure in the post: **TP=1 → TP=2**, KV blocks ~**13.9×**, token throughput ~**3.9×** — not 2×. Same story as PagedAttention: what you buy first is rooms. `optimization.md` says the same under preemption: raise `tensor_parallel_size` so KV fits; weigh it against sync cost.

## Extra inference headaches

KV has to move with the parallel plan (later P/D, Mooncake, NIXL). Speculative decoding makes “tokens this step” not equal 1. The control plane decides who participates, how micro-batches are cut, and how failure is collected — CPU starvation shows up as idle GPUs.

Conclusion in the original also names **chunked prefill** next to TP/PP as part of serving large models, plus upcoming **expert parallelism for MoE** and more quantization.

## Further reading (named then)

- Megatron-LM (Shoeybi et al., 2019)
- [Orca (Yu et al., 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

Figures: Sangbin Cho (xAI) originated some of them. The original page also advertised bi-weekly office hours — skip that chrome in the notes.
