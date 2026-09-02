---
source: https://vllm.ai/blog/2025-02-17-distributed-inference
lang: en
fetched: 2026-08-31
---

# Distributed Inference with vLLM

2025-02-17. https://vllm.ai/blog/2025-02-17-distributed-inference  
Same map as the TensorRT-LLM sharding chapter: communication is the constraint. Quantization alone does not save 100B+ models. 

Inference is not training: shapes change, latency matters, and you still have to house KV, speculative decoding, and the prefill→decode handoff. vLLM’s knives in this post are mainly **in-node TP** and **cross-node PP**, plus communication kernels and a control plane that tries not to let CPU stall GPU. Expert parallelism is a forward look here; it becomes the main course in [large-scale.md](large-scale.md).


Local figures (copyright remains with the original site; study copies):

![tp strategies](../../../../assets/vllm/blog/serving/distributed-inference/01-tp_strategies.png)

![column row parallel](../../../../assets/vllm/blog/serving/distributed-inference/02-column_row_parallel.png)

![tensor parallelism](../../../../assets/vllm/blog/serving/distributed-inference/03-tensor_parallelism.png)

![kv cache effects](../../../../assets/vllm/blog/serving/distributed-inference/04-kv_cache_effects.png)

## Tensor parallelism

Megatron-LM lineage (Shoeybi et al., 2019), adapted for inference. Column-parallel splits weight columns and concatenates; row-parallel splits rows and reduces.

Llama MLP postcard: column-parallel up-proj → SILU on shards → row-parallel down-proj + **all-reduce**. Splitting weights multiplies memory bandwidth, which helps memory-bound decode — if the corridor is fast (NVLink / IB). A slow corridor eats the win.

MLA-style models later make naive TP a bad idea (duplicated latents). That is the Wide-EP argument: EP + DP Attention instead of more TP.

## Pipeline parallelism

When a stack of layers will not fit one multi-GPU node (DeepSeek R1, Llama 3.1 405B), cut **contiguous layers**. Activations send/recv once per stage. Cheaper than TP all-reduce; does **not** inherently cut latency. Idle bubbles are filled with pipeline scheduling / micro-batches.

Rule of thumb (almost the TRT-LLM sentence): slow inter-node → TP inside the node, PP between nodes. Fast NVLink/IB → TP may cross nodes. Combine them to avoid paying the wrong communication tax.

## Super-linear KV

2 GPUs is not 2×. After the split, KV room per GPU can grow faster than linear, so batches can grow. Figure in the post: TP=1→2, KV blocks ~**13.9×**, token throughput ~**3.9×**. Same story as PagedAttention: what you buy first is rooms. `optimization.md` says the same under preemption: raise `tensor_parallel_size` so KV fits; weigh it against sync cost.

## Extra inference headaches

KV has to move with the parallel plan (later P/D, Mooncake, NIXL). Speculative decoding makes “tokens this step” not equal 1. The control plane decides who participates, how micro-batches are cut, and how failure is collected — CPU starvation shows up as idle GPUs.

Further reading named then: Megatron-LM, Orca, DeepSpeed, FasterTransformer. EP and more quantization were already on the horizon.
