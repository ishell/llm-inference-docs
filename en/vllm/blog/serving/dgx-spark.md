---
source: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
lang: en
fetched: 2026-09-05
---

# vLLM on DGX Spark: Architecture, Configuration, and Local Evaluation

Chinese: [zh/vllm/blog/serving/dgx-spark.md](../../../../zh/vllm/blog/serving/dgx-spark.md)

2026-06-01. **Inferact**. Desk-side **GB10** / `sm_121`, not a datacenter GPU. Working example: [Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4). Day-0 model note: [nemotron-3-super.md](nemotron-3-super.md). Docker Model Runner later named Spark as incoming: [docker-model-runner.md](docker-model-runner.md). Demo numbers on one office Spark — not a leaderboard.

**TL;DR from the page:**

- CPU, GPU, OS, container runtime, weights, and KV share **one 128 GB** unified pool. `--gpu-memory-utilization` must leave headroom; recipe uses **0.85**.
- `--max-num-seqs 4`. Above four concurrent decode streams the per-token bandwidth tax can outweigh continuous batching; TTFT spikes.
- Fit **~100–130B NVFP4 MoE** with **~10–15B** active parameters. Dense / high-concurrency serving can run; it is less aligned.
- Official image `vllm/vllm-openai:cu130-nightly` is a **compatibility track**, not a pin. Validate a digest.
- Five-scenario median after warmup: Decode **22.7–23.7 tok/s**. Prefill **140 → ~1,900 tok/s** as prompts grow. First-request Inductor/FlashInfer JIT ~**25 s**. Safetensor load **10–15 min**.

![office dgx spark](../../../../assets/vllm/blog/serving/dgx-spark/01-office-dgx-spark.jpg)

*vLLM running Nemotron-3-Super on the DGX Spark for a demo at the Inferact office.*

Local figures (copyright remains with the original site; study copies):

![dgx spark vllm serving architecture](../../../../assets/vllm/blog/serving/dgx-spark/02-dgx-spark-vllm-serving-architecture.svg)

![gb10 unified memory sm121 map](../../../../assets/vllm/blog/serving/dgx-spark/03-gb10-unified-memory-sm121-map.svg)

![dgx spark model fit decode rate](../../../../assets/vllm/blog/serving/dgx-spark/04-dgx-spark-model-fit-decode-rate.svg)

![spark vllm config stability performance slider](../../../../assets/vllm/blog/serving/dgx-spark/05-spark-vllm-config-stability-performance-slider.svg)

![vllm spark game demo flow](../../../../assets/vllm/blog/serving/dgx-spark/07-vllm-spark-game-demo-flow.svg)

![dgx spark vllm benchmark sweep](../../../../assets/vllm/blog/serving/dgx-spark/08-dgx-spark-vllm-benchmark-sweep.svg)

## Technical summary

vLLM on Spark is a **local OpenAI-compatible endpoint**: memory, batching, KV-cache, and Prometheus controls for large NVFP4 models. The Nemotron-3-Super recipe uses the [official OpenAI-compatible server image](https://docs.vllm.ai/en/latest/deployment/docker/) plus Spark-specific flags.

Architecture drives config: `sm_121` consumer Blackwell, unified CPU+GPU pool, Spark memory bandwidth. Continuous batching, paged KV, NVFP4 kernels, and `/metrics` are the relevant controls.

`--gpu-memory-utilization` is a fraction of the **unified** pool. `--max-num-seqs` stays low: Spark is small-batch, not high-concurrency. Current builds should keep **CUDA graphs on** unless a deployment has a reason to disable them. Tuned throughput (newer FP4 kernels, async scheduling, MTP speculative decoding) is **model- and release-specific**.

## DGX Spark architecture and memory model

GB10 Grace Blackwell SoC. Three properties feed the rest of the post.

**Unified memory expands local model size.** More of the box can sit in inference than a fixed dedicated GPU pool would allow. The page says it is practical to load larger NVFP4 models with **up to 200 billion parameters** on a single Spark, depending on architecture and runtime. vLLM knobs: `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`, paged KV. Multi-Spark: ConnectX “low latency and high bandwidth” for distributed inference — no Gbps figure on the page.

**`sm_121` validation.** Use builds, image tags, and flags validated for Spark. Adapting a larger-GPU config is an **engineering checklist** (kernels, memory behavior), not a performance expectation.

**NVFP4 MoE is the strong fit.** NVFP4 cuts memory pressure and helps Prefill / model-fit; Decode is still shaped by **active** parameter count and the kernel path in the current build. ~10–15B-active NVFP4 MoE is the sweet spot. Dense models and high-concurrency serving are less aligned with bandwidth and the unified pool.

**Figure 2** (local `03-…svg`): CPU, GPU, OS, weights, KV share the 128 GB pool.

## vLLM capabilities relevant to DGX Spark

Focus: local small-batch on one Spark; multi-node when Sparks are linked. Relevant: paged KV, dynamic scheduling, OpenAI-compatible serving, metrics, `sm_121` image/runtime.

### Paged KV cache for Spark's unified memory budget

Classical lockstep batching waits on the longest request. Continuous batching admits/evicts at every Decode step. With paged KV, Spark can keep a useful number of in-flight requests without excessive fragmentation.

On a Spark serving a **120B NVFP4 MoE**, KV-cache utilization in their tests: typically **below 5%** single-user, **below 30%** under small-batch demo traffic.

### OpenAI-compatible streaming for local Spark endpoints

Same client code as a hosted OpenAI-compatible API, pointed at `http://localhost:8000/v1`. Datacenter GPUs may win Decode throughput; `stream=true` still makes the desk box feel interactive. Perceived latency matters as much as total generation time for chat, coding, agents.

**Figure 1** (local `02-…svg`): clients hit `/v1` and `/metrics` on a local official vLLM image.

### Spark serving metrics through Prometheus

On one Spark, observability means: Prefill is quick, Decode is steady, the unified pool has headroom. No extra sidecar required. Demo telemetry can poll `/metrics` on the same machine.

Signals they name: `vllm:kv_cache_usage_perc`, prompt and generation token counters, **TTFT** and inter-token-latency histograms. Healthy agentic run: first turn spends time in Prefill; later turns grow KV but Prefill should **not** spike if the conversation prefix is cached. Generation throughput and ITL settle near the expected Decode rate. Compact the conversation before KV nears the context limit.

### Official vLLM image for DGX Spark

Their Nemotron-3-Super run: CUDA 13 nightly [`vllm/vllm-openai:cu130-nightly`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=cu130-nightly) with Spark-specific parser, FP4, scheduling, and memory settings. Nightly tags move — treat as a **track**. For a deployment, pin a release tag, commit-specific nightly, or digest.

Spark does not need a bespoke serving API. The Spark-specific work is the **recipe, image, and flags** for GB10 `sm_121`.

## Runtime configuration and environment variables

### Recipes and docs to check first

Start at [vLLM Recipes](https://recipes.vllm.ai/), then the generated [`vllm serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/) and [Docker docs](https://docs.vllm.ai/en/latest/deployment/docker/). Keep [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and [production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) nearby. NVIDIA Spark guides remain source of truth for Spark-specific recipes, parser plugins, and kernel settings.

### Model selection

Largest lever on Spark, **before** flag tuning. **Figure 3** (local `04-…svg`) is **directional** model-fit guidance, not a performance table: 100–130B MoE NVFP4 with ~10–15B active is the strong local interactive fit. Nemotron-3-Super-120B-A12B-NVFP4 is the concrete example. Other Spark-sized NVFP4 MoE: same principles, start from **that** model's recipe.

### Pre-staging the weights

Do not let the first `vllm serve` also download the model. Pre-stage into a host-mounted Hugging Face cache; mount the same cache into the long-running container. “Download once, mount everywhere.”

### Flags that matter for `vllm serve`

Example: `vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` plus:

**`--gpu-memory-utilization`.** Fraction of GPU-visible memory. On Spark that is the unified pool: OS, kernel page cache, container runtime, KV growth, other processes. Start from the recipe; tune on observed headroom and concurrency.

**`--max-model-len 131072`.** Max prompt + completion. 131K because system prompts, tool schemas, files, and history can exceed **20K** tokens. Raise toward the model max or lower for a constrained demo. Not a fixed worst-case KV reservation for every in-flight request — vLLM schedules on **active** context.

**`--max-num-seqs 4`.** Max in-flight sequences. Current Nemotron NVFP4-on-Spark recipe keeps this low. Above four concurrent decode streams, per-token bandwidth tax can beat continuous-batching gains; TTFT spikes.

**Automatic prefix caching.** [Prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) is **on by default in vLLM V1**; the example does not pass `--enable-prefix-caching`. Useful for a long shared system prompt. The app must stay correct at **zero** hits.

**Tool and reasoning parsers.** Follow the **model recipe**, not a hardware default. Reasoning parser only if the model emits supported blocks; `--enable-auto-tool-choice` plus a tool-call parser only if the client needs tools. Current builds: Nemotron-3 can use built-in `--reasoning-parser nemotron_v3`. Older Spark recipes may still name the external `super_v3` plugin.

Evaluate, do not copy blindly:

- [`--kv-cache-dtype fp8`](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) can shrink KV but may hurt predictability and can carry a **noticeable performance cost on Spark** for some workloads. Skip unless memory pressure requires it and quality checks pass.
- [`--speculative-config`](https://docs.vllm.ai/en/latest/features/speculative_decoding/): for this model, MTP. Re-measure.
- `--tensor-parallel-size 2` is only meaningful if **two Sparks** are linked through ConnectX-7. Not a single-node tuning flag.

### When to override vLLM defaults

On single-GPU Spark: recipe + defaults first. Explicit overrides only when intentional for the exact model, image, and hardware you validated.

**Backend selection.** Leave quantized linear and MoE backends on `auto` unless the tested recipe pins one. The right FP4 path changes with release and architecture; recent **FlashInfer CUTLASS** paths are much stronger than older Spark guidance. If you pin, prefer `--linear-backend` and `--moe-backend`. Older env vars for this path are **deprecated**.

**Version-specific workarounds.** Compatibility env vars in some Spark recipes are tag-specific, not general vLLM requirements. Example: a FlashInfer allreduce backend override is **not** needed for a single-Spark command with no tensor parallelism.

**Checkpoint quantization.** vLLM reads quantization from the model config. For a pre-quantized NVFP4 checkpoint, leave `--quantization` **unset**. Set it only to apply a method at load time on purpose.

### Pre-warming the JIT

Cold-start depends on model, kernels, image, request path. In their Nemotron-3-Super setup, the first request after `vllm serve` boots triggers Inductor and FlashInfer JIT and can take ~**25 s**. Do not send that path to an end user. Fire a small `ping` at startup on the **same** client path (`chat_template_kwargs`, `max_tokens=3`). Once warm, that short prompt path returns in **under 0.5 s** in their setup.

Weight load is separate. If the **10–15 min** safetensor load matters, evaluate [fastsafetensors](https://docs.vllm.ai/en/latest/models/extensions/fastsafetensor/) or [InstantTensor](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/) against the exact model, image, and storage.

### Predictability and throughput tuning

For the measurements in the post: `--kv-cache-dtype` unset, speculative decoding **off**, CUDA graphs **on**. Recipe choices for this model/image/workload, not universal Spark defaults. Throughput-oriented runs can still try FP8 KV, async scheduling, speculative decoding, explicit backends — re-validate.

They optimize here for a public demo: predictable local serving, clear telemetry, stable responses.

**Figure 4** (local `05-…svg`): slider from straightforward demo settings to tuned throughput (FP4 backend, async scheduling, speculative decoding).

## Example workload: vllm-spark-game

[vllm-spark-game](https://github.com/zlxi02/vllm-spark-game): live 20-Questions against the local endpoint; a companion stats view polls vLLM and GPU telemetry on the same Spark. Exercises OpenAI-compatible chat, streaming, Prefill, Decode, KV, live metrics. Commands in the [project README](https://github.com/zlxi02/vllm-spark-game/blob/master/README.md).

![spark demo crowd](../../../../assets/vllm/blog/serving/dgx-spark/06-spark-demo-crowd.jpg)

*vllm-spark-game demo at the Inferact booth during MLSys, May 2026.*

### The Docker invocation

```bash
docker run -d --name vllm --ipc=host --restart unless-stopped \
  --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

`cu130-nightly` is the tested track; pin a digest for a runbook. First load **10–15 min** on default safetensors. Readiness: `curl -sS http://localhost:8000/v1/models | jq -r '.data[0].id'` should return `nemotron-3-super`.

### Deployment shape

**Figure 5** (local `07-…svg`): game → `/v1`; `spark-stats` polls `/metrics` and NVML on the same endpoint.

### Single-Spark evaluation results

Five application-oriented scenarios on one Spark hosting Nemotron-3-Super-120B-A12B-NVFP4. Methodology, not a leaderboard. Decode stayed **22.7–23.7 tok/s**. Each row is the **median of three runs after one warmup**. Token counts from `stream_options.include_usage`, not chunk counts.

| Scenario | Prompt tok | Gen tok | TTFT | Total latency | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| typical judge call (real 20Q, noisy 2-token gen) | 58 | 2 | 0.42 s | ~0.53 s | 140 | ~23 |
| medium prompt, short gen | 1,834 | 32 | 1.12 s | ~2.47 s | 1,636 | 23.7 |
| long prompt, short gen | 7,234 | 32 | 3.85 s | ~5.26 s | 1,877 | 22.7 |
| medium prompt, long gen | 1,834 | 108 | 1.12 s | ~5.74 s | 1,639 | 23.4 |
| long prompt, long gen | 7,234 | 124 | 3.84 s | ~9.26 s | 1,884 | 22.9 |

*Table 2 on the page. There is no Table 1 in the fetched markdown.*

**Figure 6** (local `08-…svg`): the same sweep — TTFT, total latency, Prefill, Decode in that 22.7–23.7 band.

### Evaluation interpretation

**Prefill scales near-linearly with prompt length.** TTFT roughly triples when the prompt grows four times. Prefill climbs from **140** to nearly **1,900 tok/s** as the prompt amortizes per-request overhead. Prefill is compute-bound and parallelizable across the full prompt.

**Decode stays in a narrow 22.7–23.7 tok/s band.** The judge call’s user-facing latency matters more than its Decode rate (two tokens). Decode still depends on active parameters, FP4 path, CUDA graphs, and the exact image. Recipe-specific for Nemotron-3-Super on **one** Spark — not a universal Spark or vLLM ceiling.

Report image tag, context length, CUDA graph status, backend, and scheduling with any reproduced eval.

**Live 20-Questions.** Typical turn: ~**1,000-token** prompts (system + facts + secret + question). Perceived latency dominated by TTFT and short Decode bursts. For **5–15** output tokens, Decode is roughly **0.2–0.7 s** inside that tok/s band. KV-cache utilization **rarely tops 2%** during play. Telemetry: `prompt_tps` spikes at turn start, then `gen_tps` holds in-band while the answer streams.

## Operational takeaways

Pick the model class first: 100–130B NVFP4 MoE matches capacity and active-parameter profile; dense is usually less aligned with interactive local Decode. Official image + Spark-tested recipe beats a source build unless you need custom kernels. Tune `--gpu-memory-utilization` for the shared pool. Pre-warm JIT. `/metrics` gives KV utilization and TTFT histograms.

## Concluding thoughts

Spark is a local inference system for development, demos, and small-batch serving. Different profile from a datacenter GPU server. Workload tuning matters: unified memory, `sm_121`, model-specific FP4, local Decode. With a validated model, image, and flags, applications still get OpenAI-compatible APIs, streaming, continuous batching, paged KV, and Prometheus.

*Written by [Inferact](https://inferact.ai) on a Spark they keep running at the office.*
