---
source: https://vllm.ai/blog/2025-08-20-torch-compile
lang: en
fetched: 2026-09-04
---

# Introduction to torch.compile and How It Works with vLLM

Chinese: [zh/vllm/blog/architecture/torch-compile.md](../../../../zh/vllm/blog/architecture/torch-compile.md)

2025-08-20. Office-hours write-up from the biweekly forum hosted by Red Hat with vLLM committers and the UC Berkeley team (deep dive + Q&A; recordings on their YouTube playlist). This is what `-O0`…`-O3` and `--enforce-eager` sit on. Study note, not a dump of the page.

**torch.compile** is PyTorch’s JIT: wrap a function or `nn.Module`, capture tensor ops, emit fused kernels. For vLLM it is not a bolt-on accelerator. The bet is that **model files stay declarative and optimizations happen at compile time**, so hundreds of architectures do not each pay a hand-tuned kernel tax.

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

## What torch.compile is

Applying it is a decorator (or wrapper) on a function, `nn.Module`, or other callable. Figure 1: it produces a single fused kernel for the pointwise ops in `fn`. Capture and compile are JIT; a change in capture conditions (input shapes, …) can trigger a recompile.

You can use it as a **kernel generator** (compile one function) or compile a full model / submodule. Where to put the wrapper depends on model structure and compile-time budget; PyTorch’s troubleshooting docs are the then-current advice.

## Why bother

Hand-written CPU/CUDA ops can hit the ceiling, but every model × platform pays once. torch.compile is meant to get **a decent fraction of peak with almost no extra kernel engineering**. The post cites PyTorch’s open-source [TorchBench](https://hud.pytorch.org/benchmark/compilers) suite: **1.8–2× geomean** on **80+** models. That is a *baseline*, not a FlashAttention replacement. Figure 2 is the “save development time” slide.

## Two-stage pipeline

Details in the [PyTorch 2 paper](https://docs.pytorch.org/assets/pytorch2-2.pdf).

### Frontend: TorchDynamo (graph capture)

A custom bytecode interpreter traces Python and extracts straight-line [`torch.fx`](https://docs.pytorch.org/docs/stable/fx.html) graphs of **Tensor ops only**. Unsupported Python does not crash the whole compile: it **graph-breaks** — end the current graph, run the unsupported statement, start a new graph. Each traced graph goes to the backend.

Figure 3: `torch.save` is disk I/O; compile of `f` is equivalent to compiling the compute before and after that call, not teaching the compiler how to write files.

### Backend: TorchInductor (optimize + emit kernels)

Graph passes, then lowering to C++, Triton, or other kernels. What the post lists:

- Fuse pointwise and reduction ops
- Autotune kernel configs (block sizes, …)
- Choose matmul backends (**cuBLAS / Triton / CUTLASS**) and fuse prologue/epilogue
- CUDA Graphs to cache and replay launches

CUDA Graphs need static input addresses and CUDA-only ops. The compiler **splits** at unsupported ops into smaller CUDA-graph-safe graphs and manages static input buffers.

## How vLLM wires it

vLLM **V1** turns torch.compile on by default for **online and offline**. Disable with `-O0` or `--enforce-eager`. Then-current design notes: [vLLM torch.compile docs](https://docs.vllm.ai/en/latest/design/v1/torch_compile.html).

### Compilation cache

Cold start compiles and writes artifacts (FX graphs, Triton kernels) to `~/.cache/vllm/torch_compile_cache` by default. Warm start reads them back. Disable with `VLLM_DISABLE_COMPILE_CACHE=1` or by deleting the directory.

Artifacts reuse across machines with the **same environment**. Autoscaling: generate the cache once, share it among instances. Figure 4.

### Dynamic batch sizes and specialization

Default: **one graph with a dynamic batch size** covering all batch sizes. If you only run 1, 2, or 4, specialize:

```text
compile_sizes: [1, 2, 4]
```

That compiles those **static** sizes and can autotune more aggressively. Figure 5.

### Piecewise CUDA Graphs

Not every op is CUDA-graph compatible — [cascade attention is not](https://docs.vllm.ai/en/latest/design/v1/torch_compile.html#full-cudagraph-capture). vLLM splits the captured graph into CUDA-graph-safe and unsafe regions and runs them separately. This is the piecewise path behind `-O1` PIECEWISE and `-O2` `FULL_AND_PIECEWISE` in later flag revamps. Figure 6.

## Custom compiler passes

Inductor already fuses a lot. vLLM adds **custom FX passes** so model authors keep modular layers while peak performance fuses **across** submodules.

Passes:

- Fuse memory-bound custom ops (activations, quantization)
- Optimizations Inductor does not have (extra no-op elimination)

### Example: SiLU + Quantize

Quantized MLP: SiLU, then a quantized down-proj (quant op + quantized matmul). Alone, SiLU and quant are memory-bound. `ActivationFusionPass` uses Inductor’s pattern matcher to replace them with one fused kernel. Throughput up to about **+8%**.

Figure 7: Llama 3.1 **405B** FP8 on **8× AMD MI300** — `fusion` vs `default` (torch RMSNorm/SiLU + custom FP8 quant kernel) vs `custom` (unfused custom kernels). Figure 8: if **all** quantization overhead (~**8%**) disappeared via fusion, that would be the theoretical ceiling; some points hit it.

**Then-current caveat (after the office hours):** they added a **torch-op** quantization path. Compiled by Inductor, it was **faster than the custom CUDA/ROCm kernel**, and Inductor can fuse those torch ops with SiLU automatically — so **SiLU+quant and RMSNorm+quant passes are obsolete on some paths**. Fusions that involve **custom ops** (attention, collectives, sub-byte quant) still need custom passes. The SiLU+Quant example is kept for slide consistency; other fusion passes look similar.

### Example: Sequence Parallelism + Async TP

Under Tensor Parallelism, a linear layer shards weights, produces incomplete GEMMs, then synchronizes. Separate compute and communication kernels leave GPUs idle on network latency.

Overlap via fused **GEMM+collective** kernels (GEMM+`reduce_scatter`, `all_gather`+GEMM). That needs decomposing `all_reduce` into `reduce_scatter` + `all_gather`, and **postponing** the `all_gather` until after layernorm so it can fuse with the next GEMM.

Doing this in model definitions would touch every architecture vLLM supports. Two compile passes + CLI flags apply it to all models. Cited speedup: up to about **+10%**. Implemented in full by community member [@cascade812](https://github.com/cascade812); Async TP background on the [PyTorch blog](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487).

### Passes then available

**Fusion:**

- RMSNorm + Quant (FP8)
- SiLU-Mul + Quant (FP8)
- Attention + Quant (FP8) — up to **~7%**
- AllReduce + RMSNorm — up to **~15%**
- AllReduce + RMSNorm + Quant (FP8) — up to **~8%**
- AllReduce + RMSNorm + Quant (FP4) — up to **~10%**
- Sequence Parallelism & Async TP — up to **~10%**

**Other:**

- **No-op Elimination** — drop or simplify redundant reshapes
- **Fix Functionalization** — reinplace `auto_functionalized` ops to avoid extra copies / memory

**Then “coming soon” (PRs at posting):**

- Attention + Quant (FP4): [#22703](https://github.com/vllm-project/vllm/pull/22703)
- SiLU-Mul + Quant (FP4): [#22448](https://github.com/vllm-project/vllm/pull/22448)

Add passes via `PostGradPassManager`, CLI `--compilation-config`, or an offline config object — custom graph transforms without editing vLLM source.

## Future work (then: next six months)

**Stability.** The integration used many **private** (underscore-prefixed) torch.compile APIs because the public API could not give “fast serving and **no recompiles during serving**.” That caused odd cache bugs and needing to disable vLLM’s compile cache for some models. The PyTorch compiler team was upstreaming inference features and migrating vLLM onto stable APIs. Many of those were already in **torch 2.8**, landing in vLLM via [#20358](https://github.com/vllm-project/vllm/pull/20358) (“soon” at posting).

**Startup time.** Cold and warm start both hurt autoscaling. Track [startup-ux](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3Astartup-ux) and Slack `#feat-startup-ux`. Planned UX: revamp `-O` ([#20283](https://github.com/vllm-project/vllm/issues/20283)). `-O<n>` with `n` in **0–3**: `-O0` almost no opts, fastest spin-up; `-O3` slowest start, best performance.

**Custom pass mechanism:**

- Compile **multiple dynamic-shape** `torch.fx` graphs — specialize the forward graph by batch size without a static compile per size. [RFC](https://github.com/vllm-project/vllm/issues/23113).
- Pattern-match **torch implementations of custom ops**. Custom ops (`rms_norm`, quant, …) currently must be enabled for fusion matching, but leftover unfused custom ops (quant can run **4× per layer**) are slower than torch equivalents. Working prototype claimed further gains.

**Experimental backend:** MPK/Mirage — a precision-scheduling **megakernel** compiler: one kernel for the whole forward, less CPU and launch overhead than CUDA Graphs. [RFC](https://github.com/vllm-project/vllm/issues/22201).

**Other then-WIP:**

- Better [FlexAttention](https://github.com/vllm-project/vllm/issues/19765) — attention variants without a custom kernel per variant; torch.compile produces a Triton template.
- [Full CUDA Graphs](https://github.com/vllm-project/vllm/pull/20059) for Flash Attention v2 and FlashInfer — less overhead than piecewise, for high-overhead settings.

## Close

Caching, dynamic shapes, CUDA Graphs, and custom passes are the serving stack around compile. Docs: [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [vLLM design](https://docs.vllm.ai/en/latest/design/v1/torch_compile.html). Slack: `#sig-torch-compile`.

Read after [v1-alpha.md](v1-alpha.md) and [mrv2.md](mrv2.md): V1 made compile the default; MRV2 moved runner bookkeeping onto the GPU; this post is why fusion should not rewrite every model file.
