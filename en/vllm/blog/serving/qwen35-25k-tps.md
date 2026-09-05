---
source: https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
lang: en
fetched: 2026-09-04
---

# vLLM Reaches 25K Total TPS/GPU on Qwen3.5

Chinese: [zh/vllm/blog/serving/qwen35-25k-tps.md](../../../../zh/vllm/blog/serving/qwen35-25k-tps.md)

2026-08-06. **vLLM Team**. GB200 NVL72. Model: [`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4). ISL/OSL=**8192/1024**. Hybrid-attention ancestor: [qwen3-next.md](qwen3-next.md). Later Max-class day-0: [qwen38.md](qwen38.md). Heterogeneous cache / dual descriptors: [hybrid-ssm.md](hybrid-ssm.md). Decode-side relative: [../performance/dcp.md](../performance/dcp.md). NIXL P/D roadmap: [issue #33702](https://github.com/vllm-project/vllm/issues/33702). Study note. **System TPS/GPU ≠ per-user TPS.** They sweep the **left** Pareto (aggregate throughput), concurrency **64–5120**, not 1–32.

Decode fixed **1×DEP8**; prefill **4–8×DEP2**. Peak **25,000 tok/s/GPU**. GSM8K **88%** on all five topologies, matching aggregated. Reproduce with Docker `vllm/vllm-openai:nightly-d223c90`, Dynamo `1.2.0.dev20260526`, srt-slurm `v1.0.32`. Recipes: [srt-slurm-recipes …/Qwen3.5/GB200/8k1k/vllm/disagg](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg).

Local figures (copyright remains with the original site; study copies):

![pareto curves by prefill endpoints](../../../../assets/vllm/blog/serving/qwen35-25k-tps/01-pareto-curves-by-prefill-endpoints.png)

![pareto frontier qwen35 nvfp4](../../../../assets/vllm/blog/serving/qwen35-25k-tps/02-pareto-frontier-qwen35-nvfp4.png)

## Introduction

Qwen3.5 (early 2026) is a hybrid: full-attention layers interleaved with **Gated Delta Network (GDN)**. Disaggregated serving then has two extra jobs: Blackwell GDN kernels, and moving **heterogeneous** attention/GDN state between Prefill and Decode. Community work made that path mature. This post is how **you** get over **25K total TPS/GPU** on GB200 NVL72 — contributions, numbers, recipes. Not a 1–32 user latency sweep.

## Challenges and key optimizations

SSM P/D was driven through the [NIXL disaggregation roadmap](https://github.com/vllm-project/vllm/issues/33702). Layout / logical vs physical blocks / TP state transfer: [hybrid-ssm.md](hybrid-ssm.md). Three cuts matter for Qwen3.5.

### 1. Blackwell-optimized GDN Prefill

[FlashInfer PR #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001) — Blackwell GDN prefill kernel. Versus previous FLA/Triton: about **1.02×–5.78×** across Qwen3.5 sizes, TP, sequence lengths, batch shapes.

Enabled on vLLM Prefill: [vLLM PR #40717](https://github.com/vllm-project/vllm/pull/40717). On **8×B200**, Qwen3.5-397B-A17B-NVFP4:

- Up to **5.92×** GDN kernel in microbenchmarks
- **1.13×** end-to-end Prefill throughput on Prefill-only (ISL/OSL = **8192/1**)
- **−12%** mean TTFT on that same 8K/1 Prefill-only load

On supported Blackwell, `auto` picks FlashInfer. Explicit:

```text
--gdn-prefill-backend flashinfer
```

### 2. Hybrid cache and GDN-state transfer

P/D for hybrid SSM-attention sits on [[Core][KVConnector] Support HMA+NixlConnector #35758](https://github.com/vllm-project/vllm/pull/35758) plus the connector stack in [hybrid-ssm.md](hybrid-ssm.md). Necessary prerequisite: map HMA logical blocks onto the right physical regions so NIXL transfers only the cache that belongs to each layer type. Descriptors **4284 → 1650**; up to ~**7%** throughput on a small-scale intra-node H100 setup. **HMA alone is not enough** for Mamba-style state (layout, size, transfer semantics).

Main hybrid SSM-FA P/D PR: [[PD][Nixl] Add support for hybrid SSM-FA models #36687](https://github.com/vllm-project/vllm/pull/36687) — dual descriptor views + homogeneous-TP, so Prefill and Decode can move full-attention KV **and** Mamba-style SSM state over NIXL. Same stack:

- [[Kernel] Mamba support different layout for Conv state #37416](https://github.com/vllm-project/vllm/pull/37416)
- [[NIXL][Mamba][3/N] Heterogeneous TP: 3-read conv state transfer #37635](https://github.com/vllm-project/vllm/pull/37635)
- [[SSM/Mamba] Follow-up: N-1 prefill for P/D disaggregation #37310](https://github.com/vllm-project/vllm/pull/37310)

Qwen3.5-specific GDN: [PD disagg with NIXL Connector: GDN support (Qwen3.5) #41869](https://github.com/vllm-project/vllm/pull/41869).

### 3. Race-free async scheduling

Two patches. Without them, `--async-scheduling` was unusable — **accuracy collapsed to zero**. Async scheduling is one of the features behind crossing 25K tok/s/GPU, so both races had to land first.

- [[KV Connector] Fix PD async scheduling race condition for hybrid attn models #48481](https://github.com/vllm-project/vllm/pull/48481)
- [[Bugfix] Defer block freeing until in-flight steps finish under async scheduling + PD KV consumer #45357](https://github.com/vllm-project/vllm/pull/45357)

## Performance

### Environment

GB200 cluster, NVLink72. ISL/OSL = **8192/1024**. Model [`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4). Decode topology **fixed**: one endpoint, **DEP8** (DP+EP across 8 GPUs). Prefill: **4 to 8** endpoints, each **DEP2**.

Reproduce: latest `vllm/vllm-openai:nightly-d223c90` ([Hub layer](https://hub.docker.com/layers/vllm/vllm-openai/nightly-d223c900d85224c02f2162ee2c757a769e99f519/images/sha256-987393f42c48b8a649961a3484d95d400db184b64e4e1bb7f77cb91536d0f05e)), [Dynamo](https://github.com/ai-dynamo/dynamo) `1.2.0.dev20260526`, [srt-slurm](https://github.com/NVIDIA/srt-slurm) `v1.0.32`. Recipes in [srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes).

### Accuracy

GSM8K on every serving topology, via srt-slurm:

```yaml
benchmark:
  type: "gsm8k"
```

All **five** configurations: **88%**, matching the aggregated Qwen3.5 run. If this is not ~88, the P/D path is wrong — do not read the Pareto.

### Recipe settings (measurement)

Fixed ISL/OSL, random dataset, `random_range_ratio=0.8`. Flags below.

### Results

**Figure 1.** Pareto curves for disaggregated Qwen3.5 with different numbers of Prefill instances.

**Figure 2.** Combined Pareto frontier, Qwen3.5 NVFP4.

Total TPS per GPU reaches **25,000** tok/s. Concurrency **64…5120**. They **did not** measure 1–32: the goal is the **left** Pareto (max total TPS/GPU), not per-user Gen TPS. Past 5120 they ran out of KV on the **single** 8×GB200 Decode endpoint they froze. Higher concurrency is possible; it needs more Decode GPUs.

## Recipes and best practices

Launch:

```shell
srtctl run --file <recipe>.yaml
```

Naming: `NxDEP2-1xDEP8` — N Prefill endpoints at DEP2 against one DEP8 Decode. Five bases: **4×DEP2 … 8×DEP2**. Each has three derived files: base sweeps sa-bench over cc **64…3072**; `-acc` runs GSM8K **five** times on the same topology; `-cc4096` / `-cc5120` are single high-cc points with Decode `--max-cudagraph-capture-size` raised to **640** and **768**.

Call-outs (most other settings are shared boilerplate):

- `VLLM_SSM_CONV_STATE_LAYOUT=DS` — **mandatory** for SSM models in P/D; conv-state transfer does not work without it. Recipes also passed `--no-disable-hybrid-kv-cache-manager`; HMA is default in later vLLM, so that flag is no longer required.
- `--async-scheduling` — key to 25K tok/s/GPU. Needs a build that already contains the two race fixes.
- `--mamba-ssm-cache-dtype bfloat16` — raises effective Decode KV capacity.
- `--language-model-only` — Qwen3.5 is multimodal; for text-only this disables multimodal **and** unlocks fused QK-norm + RoPE + gate.
- `--max-num-batched-tokens 16384` on Prefill = **2×ISL**. With fewer Prefill endpoints ({4,5,6}×DEP2) Prefill starved Decode; batching two full prompts per step ≈ **+8%** total TPS/GPU at high cc.
- `--max-cudagraph-capture-size` on Decode — `cc/8 + 128` at the two highest points (640 @ cc=4096, 768 @ cc=5120); 8 = DP ranks on Decode. Default cap **512** is enough through cc=3072. They are **not sure** this was required for the Pareto numbers; precaution.
- Prefix caching **off** — buys nothing on a random dataset.
- `--stream-interval 100` — cuts frontend at high cc; **buffers 100-token chunks**, so it moves ITL/TPOT. Skip if you are optimizing per-token latency, not aggregate TPS.

Practical ops:

`--api-server-count 1` while investigating. On a DP endpoint vLLM defaults API server count to DP size; more than one API server **disables default stats** so it does not print incomplete numbers. Forcing 1 brings logging back every 10 s (`VLLM_LOG_STATS_INTERVAL`): prompt/generation throughput and KV utilization. Without that they say they would not have found per-config bottlenecks.

Three env vars: `DYN_LOG=error`, `DYN_SDK_DISABLE_ANSI_LOGGING=1`, `VLLM_LOGGING_COLOR=0`. First cuts Dynamo noise; the other two strip some (not all) ANSI. Otherwise logs are unreadable.

## What's next

This work squeezed the **left** Pareto (total TPS/GPU). Next they want PD configs that maximize **Gen TPS per user**. That regime shifts off DEP toward **TEP** (TP+EP) or plain TP. More GPUs is the other lever. Later architecture reuse at 2.4T: [qwen38.md](qwen38.md). Earlier hybrid GDN/full interleave: [qwen3-next.md](qwen3-next.md).

## Acknowledgements

Artem Perevedentsev (NVIDIA), Vadim Gimpelson (NVIDIA), Jiangyun Zhu (Inferact), Nicolò Lucchesi (Mistral), Zhanqiu Hu (Red Hat), Nick Hill (Inferact), Linxuan Li (Alibaba), JingZe Cui (NVIDIA), Cyrus Chang (NVIDIA), Xin Li (NVIDIA).
