---
source: https://docs.vllm.ai/en/stable/
lang: en
fetched: 2026-09-05
---

# Welcome to vLLM

Chinese: [zh/vllm/getting-started/index.md](../../../zh/vllm/getting-started/index.md)  
Hub: https://docs.vllm.ai/en/stable/  ·  rolling: https://docs.vllm.ai/en/latest/

Docs landing page. Logos on the original are not copied. Study notes, not official docs. **Easy, fast, and cheap LLM serving for everyone.**

Originally Sky Computing Lab, UC Berkeley; now a large community project (the page says dozens of institutions and companies, **2000+** contributors). Where to start:

- Run open-source models → [Quickstart](quickstart.md)
- Build applications → official User Guide (not mirrored here)
- Build vLLM → official Developer Guide (not mirrored here)

Also named: [roadmap.vllm.ai](https://roadmap.vllm.ai), [GitHub releases](https://github.com/vllm-project/vllm/releases).

## Fast with

- State-of-the-art serving throughput
- **PagedAttention** for KV ([launch post](../blog/architecture/paged-attention.md))
- Continuous batching, chunked prefill, prefix caching
- Piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and more
- Attention kernels: FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, Triton
- GEMM/MoE kernels: CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding: n-gram, suffix, EAGLE, DFlash
- `torch.compile` kernel generation and graph transforms
- Disaggregated prefill, decode, and encode

## Flexible with

- Hugging Face models
- Parallel sampling, beam search, and other decoding algorithms
- Tensor, pipeline, data, expert, and context parallelism
- Streaming outputs
- Structured outputs (xgrammar / guidance)
- Tool calling and reasoning parsers
- OpenAI-compatible API server, Anthropic Messages API, gRPC
- Multi-LoRA for dense and MoE
- NVIDIA / AMD GPUs, x86 / ARM / PowerPC CPUs, plus plugins: Google TPU, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more

**200+** Hugging Face architectures named: decoder-only (Llama, Qwen, Gemma), MoE (Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS), hybrid attention / SSM (Mamba, Qwen3.5), multimodal (LLaVA, Qwen-VL, Pixtral), embedding/retrieval (E5-Mistral, GTE, ColBERT), reward/classification (Qwen-Math). Full list lives on the official supported-models page.

Further reading they list: PagedAttention blog, [vLLM paper (SOSP 2023)](https://arxiv.org/abs/2309.06180), Anyscale continuous-batching write-up, meetups.

## Local notes in this repo

- Install / offline / `vllm serve`: [quickstart.md](quickstart.md)
- Perf-related server flags (not the generated CLI page): [serve.md](serve.md)
- Tuning order: [optimization.md](../optimization/optimization.md)
- Client ruler: [cli.md](../benchmarking/cli.md) · batch grid: [auto-tune.md](../benchmarking/auto-tune.md)
- Clock indoors: [/metrics](../metrics/production-metrics.md) · how they are computed: [design-metrics.md](../metrics/design-metrics.md)
- Features: [APC](../features/prefix-caching.md) · [ledger](../features/prefix-caching-design.md) · [spec decode](../features/speculative-decoding.md) · [V1](../features/v1-guide.md)
