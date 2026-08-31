---
source: https://vllm.ai/blog/2025-02-17-distributed-inference
lang: en
fetched: 2026-08-31
---

# Distributed Inference with vLLM

2025-02-17. Same map as the TRT-LLM sharding chapter: communication is the constraint. Quantization alone does not save 100B+ models.

**TP** (Megatron-LM lineage): column/row splits. Llama MLP: column up-proj → SILU on shards → row down-proj + all-reduce. Needs NVLink/IB. Multiplies memory bandwidth → lower latency.

**PP**: contiguous layer stacks across nodes; send/recv activations once per stage. Lower comm than TP; does **not** inherently cut latency. vLLM keeps GPUs busy with pipeline scheduling / micro-batches.

Rule of thumb: slow inter-node → TP inside node, PP between. Fast NVLink/IB → TP may cross nodes.

Super-linear KV: TP=1→2 can grow KV blocks ~**13.9×** and token throughput ~**3.9×**, not 2× — more cache → larger batches.
