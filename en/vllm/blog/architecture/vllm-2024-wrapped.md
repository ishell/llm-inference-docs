---
source: https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision
lang: en
fetched: 2026-09-04
---

# vLLM 2024 Retrospective and 2025 Vision

Chinese: [zh/vllm/blog/architecture/vllm-2024-wrapped.md](../../../../zh/vllm/blog/architecture/vllm-2024-wrapped.md)

2025-01-10. **vLLM Team**. Study note of a then-vision doc; V1 / MRV2 / Wide-EP landed later. Based on the 16th bi-weekly [Office Hours](https://hubs.li/Q02TFDTT0); [recording](https://www.youtube.com/watch?v=xmz8lHsrbGM). Usage site: https://2024.vllm.ai. V1 rewrite: [v1-alpha.md](v1-alpha.md). Earlier governance/perf: [lfai-roadmap.md](lfai-roadmap.md). Later runner: [mrv2.md](mrv2.md). Pluggable doors: [plugin-system.md](plugin-system.md), [hardware-plugin.md](hardware-plugin.md). Spec / structured output that became defaults in the talk: [spec-decode.md](../performance/spec-decode.md), [struct-decode.md](../performance/struct-decode.md).

Fits: reading 2024 growth and the 2025 verbal roadmap (single-node GPT-4o-class, battery-included serving, V1). Does not fit: treating this page as the current architecture — it is a wrap, not the later landings.

## Growth (page numbers)

- GitHub stars **14,000 → 32,600** (**2.3×**)
- Contributors **190 → 740** (**3.8×**)
- Monthly downloads **6,000 → 27,000** (**4.5×**)
- GPU hours ~**10×** over the last six months (as of the post)

Production examples named: Amazon Rufus, LinkedIn AI. Bi-monthly meetups with IBM, AWS, NVIDIA. Aim stated: universal serving for the open-source AI ecosystem.

## 2024: models, hardware, features

### Community

![vllm contributor groups](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/01-vllm-contributor-groups.png)

**Figure.** Main contributor groups by commits (study copy).

- **15+** full-time contributors across **6+** organizations
- **20+** active organizations as stakeholders / sponsors
- UC Berkeley, Neural Magic, Anyscale, Roblox, IBM, AMD, Intel, NVIDIA, plus individuals
- Ecosystem of model creators, hardware vendors, optimization developers
- Bi-weekly office hours

### Models

![model architecture serving usage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/02-model-architecture-serving-usage.png)

**Figure.** Usage by model architecture in serving (study copy).

Start of 2024: a handful of models. Year-end: almost [**100 architectures**](https://docs.vllm.ai/en/latest/models/supported_models.html) — prominent open LLMs, multimodal (image / audio / video), encoder-decoder, speculative decoding, classification, embedding, reward. Production support for **state-space** language models.

### Hardware

![gpu hours by vendor](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/03-gpu-hours-by-vendor.png)

**Figure.** GPU hours by vendor (study copy).

From NVIDIA A100 as the first target:

- **NVIDIA:** first-class H100; V100 and newer
- **AMD:** MI200, MI300, Radeon RX 7900; MI300X adoption growing
- **Google TPU:** v4, v5p, v5e, v6e
- **AWS Inferentia / Trainium:** trn1 / inf2
- **Intel Gaudi (HPU) and GPU (XPU)**
- **CPU:** x86, ARM, PowerPC

Path claimed: all models on all hardware, with optimizations on.

### Features

![quantization deployment percentage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/04-quantization-deployment-percentage.png)

**Figure.** Share of deployments using quantization (study copy).

- **Weight and activation quantization:** FP8+INT8 activation quant; Marlin+Machete for GPTQ/AWQ/wNa16; FP8 KV cache; AQLM, QQQ, HQQ, bitsandbytes, GGUF. **>20%** of deployments used quantization.
- **Automatic prefix caching**
- **Chunked prefill** (stabler ITL for interactive)
- **Speculative decoding:** draft models, n-gram in the prompt, MLP speculators (Medusa / EAGLE)
- **Structured outputs** (JSON, pydantic)
- **Tool calling** via chat templates
- **Distributed inference:** pipeline parallelism, disaggregated prefill

## 2025 vision (as spoken then)

Open models catching proprietary ones; distillation making them smaller and more deployable. Inference-time scaling and pretraining scale both pushing.

### GPT-4o-class on one GPU / node

Verbal target: GPT-4o-level on a **single GPU**, GPT-4o on a **single node**, next-gen scale on a modest cluster. Three frontiers:

- KV / attention: sliding windows, cross-layer attention, native quantization
- MoE: shared experts, many fine-grained experts
- Long context: alternative architectures such as state-space models

Verticals named: reasoning (custom tokens, flexible steps), coding (FIM, prompt lookup), agents (tree-based caching), creative (beam variants, contrastive decode). Post-training: John Schulman named as a signal; tighter data-curation / post-training integration.

### Thousands of production clusters

Quantization, prefix caching, speculative decoding as **defaults**, not options. Structured output as standard. Recipes for routing, caching, auto-scaling. Stable interfaces for cluster-level solutions; robust defaults per model/hardware; a community pushing efficiency.

### Open architecture / V1

Ground-up **V1** rearchitecture: model architectures, scheduling, memory, sampling — meant to be modified in research and private forks. Pluggable architectures for models, hardware, extensions. First-class `torch.compile` (custom fusion). Flexible components for private extensions with a stable core. Recruit a core team; celebrate ecosystem projects. Extensibility over lock-in.

## Reflection (page themes)

**Bridges.** Model creators, hardware vendors, optimization people use vLLM as an amplifier: new accelerators get an application ecosystem; new techniques get a production demo. Contribution ↔ amplification.

**Growth vs excellence.** 2024 velocity also meant codebase complexity. H2 2024 invested in a core redesign → **V1**, so the platform stays maintainable.

**Sponsored-volunteer org.** Not funded by one company. Multiple orgs contribute code, resources, direction. Coordination across org boundaries was still being invented.

**Mission sentence:** world’s fastest and easiest-to-use open-source LLM inference and serving engine.

## Usage data

Metrics from vLLM’s [usage system](https://github.com/vllm-project/vllm/blob/main/vllm/usage/usage_lib.py). Each instance a UUID; technical fields:

- Hardware (GPU count/type, CPU arch, memory)
- Model config (architecture, dtype, TP degree)
- Runtime (quantization type, prefix caching)
- Context (cloud, platform, vLLM version)

Local file: `~/.config/vllm/usage_stats.json`. Opt out: `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, or create `~/.config/vllm/do_not_track`. Schema: [usage stats docs](https://docs.vllm.ai/en/latest/serving/usage_stats.html).

## Join (from the page)

Contribute (RFCs still open); feedback via GitHub / Slack / Discord / events; build with vLLM. [Developer Slack](https://slack.vllm.ai/). Office Hours as above.
