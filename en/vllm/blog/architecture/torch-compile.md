---
source: https://vllm.ai/blog/2025-08-20-torch-compile
lang: en
fetched: 2026-08-31
---

# torch.compile in vLLM

2025-08-20. Office-hours write-up.  This is what `-O0`…`-O3` and `--enforce-eager` sit on.

JIT: Dynamo captures Tensor-only `torch.fx` graphs (graph-breaks on unsupported Python); Inductor fuses, autotunes, picks matmul backends, CUDA-graphs the safe parts. TorchBench cite: **1.8–2×** geomean as a *baseline*, not a FlashAttention replacement.

vLLM default-on in V1. Off: `-O0` / `--enforce-eager`. Artifacts in `~/.cache/vllm/torch_compile_cache` (disable with `VLLM_DISABLE_COMPILE_CACHE=1`); copy across identical machines for autoscaling. Default: one dynamic-batch graph. `compile_sizes: [1,2,4]` specializes. Piecewise CUDA graphs skip unsafe ops (cascade attention).

Custom FX passes keep model files declarative: SiLU+quant fusion (up to ~**+8%** on Llama 3.1 405B FP8 / 8×MI300; later obsolete on some torch-op paths); Sequence Parallelism + Async TP (GEMM+collective, up to ~**+10%**, two passes instead of editing every model). Other fusions: RMSNorm/SiLU-Mul/Attention + FP8 quant, AllReduce+RMSNorm (~+15%), FP4 variants, no-op reshape elimination. Add passes via `PostGradPassManager` / `--compilation-config`.

Then: migrate off private compile APIs (torch 2.8+), cut startup for autoscale, keep `-O*` as the startup-vs-speed dial. Experimental: MPK/Mirage megakernel.

Read after [v1-alpha.md](v1-alpha.md) and [mrv2.md](mrv2.md).

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/architecture/torch-compile/01-figure1.png)

![figure2](../../../../assets/vllm/blog/architecture/torch-compile/02-figure2.png)

![figure3](../../../../assets/vllm/blog/architecture/torch-compile/03-figure3.png)

![figure4](../../../../assets/vllm/blog/architecture/torch-compile/04-figure4.png)

![figure5 a](../../../../assets/vllm/blog/architecture/torch-compile/05-figure5_a.png)

![figure5 b](../../../../assets/vllm/blog/architecture/torch-compile/06-figure5_b.png)

![figure6](../../../../assets/vllm/blog/architecture/torch-compile/07-figure6.png)

![figure7](../../../../assets/vllm/blog/architecture/torch-compile/08-figure7.png)

![figure8](../../../../assets/vllm/blog/architecture/torch-compile/09-figure8.png)
