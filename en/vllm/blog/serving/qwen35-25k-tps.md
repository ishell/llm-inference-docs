---
source: https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
lang: en
fetched: 2026-09-04
---

# vLLM Reaches 25K Total TPS/GPU on Qwen3.5

Chinese: [zh/vllm/blog/serving/qwen35-25k-tps.md](../../../../zh/vllm/blog/serving/qwen35-25k-tps.md)

2026-08-06. **vLLM Team**. GB200 NVL72. Model: [Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4). ISL/OSL = **8192/1024**. Earlier hybrid: [qwen3-next.md](qwen3-next.md). Heterogeneous cache transfer: [hybrid-ssm.md](hybrid-ssm.md). Later 3.8: [qwen38.md](qwen38.md). Decode-side long context: [dcp.md](../performance/dcp.md). Study note. **System TPS/GPU ≠ per-user TPS.** They sweep the **left** Pareto (aggregate throughput), concurrency **64–5120**, not 1–32.

Qwen3.5’s hybrid attention (full attention + Gated Delta Network) makes disaggregated serving both harder and more interesting. Community work matured the P/D path. This post: major contributions, GB200 NVL72 numbers, recipes so **you** can get over **25K total TPS/GPU**.

Local figures (copyright remains with the original site; study copies):

![pareto curves by prefill endpoints](../../../../assets/vllm/blog/serving/qwen35-25k-tps/01-pareto-curves-by-prefill-endpoints.png)

![pareto frontier qwen35 nvfp4](../../../../assets/vllm/blog/serving/qwen35-25k-tps/02-pareto-frontier-qwen35-nvfp4.png)

## Challenges and Key Optimizations

Two problems: accelerate GDN on Blackwell, and transfer heterogeneous attention/GDN state correctly between Prefill and Decode workers.

SSM support for P/D: [NIXL disaggregation roadmap](https://github.com/vllm-project/vllm/issues/33702). Layouts, logical/physical mapping, TP state transfer: [hybrid SSM disaggregation](hybrid-ssm.md).

### 1. Blackwell-Optimized GDN Prefill

[FlashInfer #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001). Versus FLA/Triton: about **1.02×–5.78×** across Qwen3.5 sizes, TP, sequence lengths, batch shapes.

Enabled on Prefill in [vLLM PR #40717](https://github.com/vllm-project/vllm/pull/40717). On **8×B200**, Qwen3.5-397B-A17B-NVFP4:

- Up to **5.92×** GDN kernel in the tested microbenchmarks
- **1.13×** e2e Prefill throughput on a Prefill-only workload (ISL/OSL = 8192/1)
- **12%** lower mean TTFT (same Prefill-only 8K/1)

On supported Blackwell, vLLM picks FlashInfer when the GDN backend is `auto`. Explicit:

```
--gdn-prefill-backend flashinfer
```

### 2. Hybrid Cache and GDN-State Transfer

Builds on [[Core][KVConnector] Support HMA+NixlConnector #35758](https://github.com/vllm-project/vllm/pull/35758) and the stack in [hybrid-ssm.md](hybrid-ssm.md). HMA logical blocks map onto the correct physical regions so NIXL transfers only the cache that belongs to each layer type: descriptors **4,284 → 1,650**, throughput up to ~**7%** in a small-scale intra-node H100 setup. Mamba-style state still differs enough that HMA alone would not have been enough for correct or efficient P/D.

Main PR: [[PD][Nixl] Add support for hybrid SSM-FA models #36687](https://github.com/vllm-project/vllm/pull/36687) — dual descriptor views, homogeneous-TP, Prefill and Decode transfer both full-attention KV and Mamba-style SSM over NIXL. Follow-ups: [#37416](https://github.com/vllm-project/vllm/pull/37416) (conv-state layout), [#37635](https://github.com/vllm-project/vllm/pull/37635) (heterogeneous TP, 3-read conv state), [#37310](https://github.com/vllm-project/vllm/pull/37310) (N-1 Prefill for P/D).

Qwen3.5 specifically: [PD disagg with NIXL Connector: GDN support (Qwen3.5) #41869](https://github.com/vllm-project/vllm/pull/41869).

### 3. Race-Free Async Scheduling

Two patches. Without them, `--async-scheduling` drove **accuracy to zero**. Async scheduling was one of the keys to crossing 25K tok/s/GPU.

- [[KV Connector] Fix PD async scheduling race for hybrid attn models #48481](https://github.com/vllm-project/vllm/pull/48481)
- [[Bugfix] Defer block freeing until in-flight steps finish under async scheduling + PD KV consumer #45357](https://github.com/vllm-project/vllm/pull/45357)

## Performance

### 1. Environment Setup

GB200 cluster, NVLink72. ISL/OSL = 8192/1024. Decode: **one** endpoint, **DEP8**. Prefill: **4–8** endpoints, each **DEP2**.

Reproduce: latest vLLM image `vllm/vllm-openai:nightly-d223c90` (digest on the page), [Dynamo](https://github.com/ai-dynamo/dynamo) `1.2.0.dev20260526`, [srt-slurm](https://github.com/NVIDIA/srt-slurm) `v1.0.32`. Recipes: [srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes).

### Accuracy Results

GSM8K on **all five** topologies: **88%**, matching the aggregated Qwen3.5 run.

```yaml
benchmark:
  type: "gsm8k"
```

### 2. Comments on Recipe Settings Choice

Fixed ISL/OSL, random dataset, `random_range_ratio=0.8`. Settings that matter: next section.

### 3. Performance Results

**Figure 1.** Pareto curves by number of Prefill instances.

**Figure 2.** Combined Pareto frontier.

Total TPS per GPU reaches **25,000** tok/s. Concurrency **64 → 5120**. They did **not** measure 1–32: the goal was the left Pareto — maximize total TPS/GPU. They stopped at 5120 because Decode KV capacity ran out on a single 8×GB200 endpoint. Higher concurrency is possible; it needs more Decode GPUs.

## Recipes & best practices

[srt-slurm-recipes … Qwen3.5/GB200/8k1k/vllm/disagg](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg). One command:

```shell
srtctl run --file <recipe>.yaml
```

Naming: `NxDEP2-1xDEP8` — N Prefill endpoints on DEP2 against one DEP8 Decode. Five bases, **4×DEP2** through **8×DEP2**. Each has three derived variants: base sweeps sa-bench over cc 64…3072; `-acc` runs GSM8K five times on the same topology; `-cc4096` / `-cc5120` capture one high-concurrency point with Decode `max-cudagraph-capture-size` raised to **640** and **768**.

Flags worth calling out:

- `VLLM_SSM_CONV_STATE_LAYOUT=DS` — **mandatory** for SSM models in disaggregated serving; conv-state transfer does not work without it. Recipes also passed `--no-disable-hybrid-kv-cache-manager`; HMA has since been default, that flag is no longer required.
- `--async-scheduling` — key to 25K tok/s per GPU. Needs a build with the race fixes above.
- `--mamba-ssm-cache-dtype bfloat16` — raises effective KV capacity on Decode.
- `--language-model-only` — Qwen3.5 is multimodal; for text-only this disables multimodal **and** unlocks fused QK-norm + RoPE + gate.
- `--max-num-batched-tokens 16384` on Prefill = **2× ISL**. With fewer Prefill endpoints ({4, 5, 6}×DEP2) Prefill starved Decode; batching two full prompts per Prefill step was worth about **+8%** total TPS/GPU at high concurrency.
- `--max-cudagraph-capture-size` on Decode — `cc/8 + 128` at the two highest points (640 at cc=4096, 768 at cc=5120); 8 = DP ranks on Decode. Default caps captured graphs at 512, enough through cc=3072. They are not sure this was required for the reported Pareto; set as precaution.
- Prefix caching **off**: buys nothing on a random dataset.
- `--stream-interval 100` — cuts frontend overhead at high concurrency. Buffers streamed output in 100-token chunks, so it **does** affect measured per-token latency. Skip if you are optimizing ITL/TPOT rather than aggregate throughput.

Practical:

`--api-server-count 1` while investigating a config. On a DP endpoint vLLM defaults API-server count to DP size; more than one API server **disables** default stats logging so incomplete numbers are not printed. Forcing 1 brings logging back: every 10 s (`VLLM_LOG_STATS_INTERVAL`) prompt and generation throughput plus KV utilization. Without those metrics they would not have found per-config bottlenecks.

Also: `DYN_LOG=error`, `DYN_SDK_DISABLE_ANSI_LOGGING=1`, `VLLM_LOGGING_COLOR=0`. First cuts Dynamo logs; the other two suppress some (not all) ANSI. Otherwise log files are often unreadable.

## What's next

So far: left Pareto, total TPS/GPU. Next: PD configs that maximize **Gen TPS per user**. That regime shifts from DEP toward **TEP** or plain **TP**, which as a rule deliver better per-user performance. More GPUs is another lever.

## Acknowledgements

Artem Perevedentsev (NVIDIA), Vadim Gimpelson (NVIDIA), Jiangyun Zhu (Inferact), Nicolò Lucchesi (Mistral), Zhanqiu Hu (Red Hat), Nick Hill (Inferact), Linxuan Li (Alibaba), JingZe Cui (NVIDIA), Cyrus Chang (NVIDIA), Xin Li (NVIDIA).
