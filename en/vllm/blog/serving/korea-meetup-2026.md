---
source: https://vllm.ai/blog/2026-04-14-vllm-korea-meetup-2026
lang: en
fetched: 2026-09-04
---

# Korea Meetup 2026: V1, playground, NPU plugins, Omni pipeline split

Chinese: [zh/vllm/blog/serving/korea-meetup-2026.md](../../../../zh/vllm/blog/serving/korea-meetup-2026.md)

2026-04-14 wrap-up of **2026-04-02** Seoul. **vLLM Team**. Hosted by vLLM KR Community; support: Rebellions, SqueezeBits, Red Hat APAC, PyTorch Korea. Community notes, not a kernel paper. 2025 debut: [korea-meetup-2025.md](korea-meetup-2025.md). Named cousins: [v1-alpha.md](../architecture/v1-alpha.md), [playground.md](playground.md), [vllm-omni.md](vllm-omni.md), [semantic-router.md](semantic-router.md), [production-stack.md](production-stack.md), [hardware-plugin.md](../architecture/hardware-plugin.md). Survey ~**75%** response; high satisfaction claimed. No SLA numbers except NAVER’s ~**3×** on their Omni decoder path.

![banner](../../../../assets/vllm/blog/serving/korea-meetup-2026/01-banner.jpg)

Field engineers on production LLM serving. Framing: inference is now infrastructure, cloud to enterprise; vLLM as the common serving layer.

## Intro: ecosystem expansion

![networking](../../../../assets/vllm/blog/serving/korea-meetup-2026/02-networking.jpg)

Dr. Hongseok Kim (Rebellions) and Li Ming (Red Hat APAC).

Kim: six months since the first meetup — Steering Group governance, regular meetups and workshops. Technical: full **v0 → v1** (simpler, more modular). Internal: async scheduling, Model Runner. Features named: streaming API, semantic router, vLLM-Omni.

![Li Ming](../../../../assets/vllm/blog/serving/korea-meetup-2026/03-intro_liming.jpg)

Li Ming: [vllm-playground](playground.md) as a GUI over **140+** knobs — shorter time-to-first-run, CPU and macOS, performance visualization.

Message they record: serving is no longer “which framework”; it is running efficiently across unlike environments.

## Accelerators + vLLM

Rebellions `vllm-rbln` plugin for their NPUs. Already: paged attention, continuous batching. Incoming: speculative decoding, distributed KV, Prefill/Decode disaggregation. Next-gen NPU **Rebel100™** named for large inference clusters. Broader shift: not siloed per-chip stacks — vLLM as the common layer. Plugin door: [hardware-plugin.md](../architecture/hardware-plugin.md).

## Production stack

![Hongseok](../../../../assets/vllm/blog/serving/korea-meetup-2026/04-intro_hongseok.jpg)

Taesoo Kim (CTO, SqueezeBits): what the [production stack](production-stack.md) offers in real ops, how it grew, where it is headed. Theme: past “just serve a model”; toward operational features production actually needs.

## Two tracks

From mid-event: Track 1 Open Source, Track 2 Business. Two sessions each.

![production stack](../../../../assets/vllm/blog/serving/korea-meetup-2026/05-production_stack.jpg)

### Track 1-1 — XCENA: memory and KV as the serving problem

Juho Lee (XCENA, CXL 3.0 intelligent memory). LLM serving as a **cluster efficiency** problem: how KV is stored and reused sets both performance and cost. LMCache tiering + routing to cut accelerator-HBM dependence; CXL as a large cache expansion tier. Implication: past compute, into data movement and memory hierarchy.

### Track 1-2 — Upstage: open-weight model → production service

Inseo Song (Upstage / Solar). After training, the hard part. Chat templates for OpenAI-compatible APIs, multi-turn, reasoning, function calling, structured outputs; token-level state parsers. Parsers and logits processors in vLLM for fine-grained generation control. Takeaway: “serving stably” is harder than “a good model.”

### Track 2-1 — Samsung: air-gap enterprise

Sungsu Kim (Samsung Electronics), “Protecting Sensitive Data with vLLM.” Security first: no external SaaS. Private LLM API on internal GPUs, air-gapped. **4000+** employees via OpenWebUI, OpenAI-compatible APIs, Dify, Claude Code. Task-separated RAG agents with access control; minimize custom code by leaning on open-source. Performance is only part of the design.

### Track 2-2 — NAVER: HyperCLOVA Omni pipeline

Jaeeun Gil (NAVER Cloud). Omni-modal (text / image / audio): autoregressive stack + diffusion decoder — heterogeneous, poor fit for a single conventional serve. Disaggregated: encoder / LLM / decoder as separate stages. **Vision decoder** dominated e2e latency. Sequence parallelism + kernel work: **>3×**. Serving as multi-component pipeline optimization, not one-model execution. (This is NAVER’s Omni model, not the vLLM-Omni project.)

## Closing

![closing](../../../../assets/vllm/blog/serving/korea-meetup-2026/06-closing.jpg)

Through-line: diverse models, heterogeneous hardware, complex pipelines, at scale. Hardware vendors, clouds, AI service companies, and end users all building around vLLM.
