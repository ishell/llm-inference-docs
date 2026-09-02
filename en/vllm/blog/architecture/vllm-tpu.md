---
source: https://vllm.ai/blog/2025-10-16-vllm-tpu
lang: en
fetched: 2026-09-01
---

# vLLM TPU: PyTorch and JAX on one XLA path

Chinese: `../../zh/vllm/blog/architecture/vllm-tpu.md`  
`pip install vllm-tpu`.

Gen-1 used PyTorch/XLA + MPMD and, by Cloud Next, ~**3.6×** Llama 3.1-8B on v6e-1 and ~**2.1×** 70B on v6e-8. This generation is **tpu-inference**: Torchax and JAX both lower JAX→XLA. Same `llama.py`, new lowering, about **+20%** throughput. Default: TPU-optimized code in tpu-inference if present, else upstream PyTorch via Torchax. Native JAX rewrites exist because GPU-shaped logic hurts TPUs, not because JAX is magically faster.

**RPA v3**: arbitrary head dim / quant / TP; KV scatter fused into attention; three compiled subkernels (prefill / decode / mixed); ~**+10%** vs v2 on Trillium (v6e). Default **SPMD** (compiler shards, inserts comms) instead of GPU-style multi-worker.

Verified then: Trillium / v5e; prefix cache, chunked prefill, multimodal (tpu-inference models), ngram spec, weight quant. Experimental: v5p, Torchax multimodal, multi-LoRA, tree Eagle-3, single-host P/D. XL MoE / MLA / vision encoders still landing. Read with [hardware plugin](hardware-plugin.md): TPU is a plugin, not a fork.

Local figures (copyright remains with the original site; study copies):

![vllm tpu](../../../../assets/vllm/blog/architecture/vllm-tpu/01-vllm-tpu.png)

![whats new](../../../../assets/vllm/blog/architecture/vllm-tpu/02-whats-new.png)

![vllm serve model](../../../../assets/vllm/blog/architecture/vllm-tpu/03-vllm-serve-model.png)

![llama3 8b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/04-llama3-8b-throughput-progress.png)

![llama3 70b throughput progress](../../../../assets/vllm/blog/architecture/vllm-tpu/05-llama3-70b-throughput-progress.png)
