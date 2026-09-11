---
source: https://vllm.ai/blog/2025-09-16-vllm-meetup
lang: en
fetched: 2026-09-04
---

# First Korea meetup: 350+ signups, plugins, TPU, quant eval

Chinese: [zh/vllm/blog/serving/korea-meetup-2025.md](../../../../zh/vllm/blog/serving/korea-meetup-2025.md)

2025-09-16 wrap-up of **2025-08-19** Seoul. **vLLM Team**. Hosts: Rebellions + Red Hat. Support: PyTorch Korea User Group, SqueezeBits. Community notes, not a kernel paper. Follow-up: [korea-meetup-2026.md](korea-meetup-2026.md). Cousins: [paged-attention.md](../architecture/paged-attention.md), [vllm-tpu.md](../architecture/vllm-tpu.md), [hardware-plugin.md](../architecture/hardware-plugin.md), [plugin-system.md](../architecture/plugin-system.md). Headline numbers: **350+** signups, **75+** companies, **80%** industry (and 80% of those software engineers / researchers). NPUs named as newly in scope.

![group](../../../../assets/vllm/blog/serving/korea-meetup-2025/01-image-3.png)

Local developers, researchers, AI infra engineers. Theme: efficient LLM inference, hardware-friendly serving.

## Nicolo Lucchesi — vLLM + llm-d, TPU

![Nicolo](../../../../assets/vllm/blog/serving/korea-meetup-2025/02-vllm_meetup_nicolo.jpg)

Nicolò Lucchesi (Senior ML Engineer, Red Hat). Origin story: KV cache + dynamic batching via PagedAttention. Line they quote: “modern problems require traditional solutions” — scheduling and memory already solved in OS paging; vLLM applies that to inference.

**llm-d:** Kubernetes-native orchestration of many vLLM instances with autoscaling — “vLLM meeting Kubernetes.”

Close: ongoing Google TPU integration, more accelerators. TPU note: [vllm-tpu.md](../architecture/vllm-tpu.md).

## Daniele Trifirò — build, test, contribute

![Daniele](../../../../assets/vllm/blog/serving/korea-meetup-2025/03-vllm_meetup_Daniele.png)

Daniele Trifirò (Senior Software Engineer, Red Hat). Weekly releases, growing contributor base, large diffs. Hardware makes local builds hard; practical tips for new contributors. Hardware-specific compilation: memory can spike by target (CUDA / ROCm / TPU). New **hardware plugin** system so devices stop forking core — more device-agnostic serving.

## Hong-seok Kim — Rebellions NPU

![Hong-seok](../../../../assets/vllm/blog/serving/korea-meetup-2025/04-vllm_meetup_HSkim.png)

Hong-Seok Kim (Chief Software Architect, Rebellions). Why vLLM matters to accelerator startups; how they contribute. Plugin path: deploy on custom silicon with a near-GPU experience. With vLLM: MoE on Rebellions NPU, plus parallelism and continuous batching, without a bespoke integration. Door to next-gen accelerators.

## Hyungjun Kim — quant and eval

![Hyungjun](../../../../assets/vllm/blog/serving/korea-meetup-2025/05-vllm_meetup_HJKim.jpg)

Hyungjun Kim (SqueezeBits). Quant as part of deploy, two vLLM paths: load a pre-quantized checkpoint, or quantize then serve. [LLM Compressor](https://github.com/vllm-project/llm-compressor) as the open subproject that wires quant into the pipeline. **Fits on Chips** (SqueezeBits toolkit): compare TPS / latency / accuracy / hardware inside vLLM to pick a serving config.

## Looking ahead

![workshop](../../../../assets/vllm/blog/serving/korea-meetup-2025/06-image-2.png)

Regular Korea meetups with PyTorch Korea User Group and Python Korea: workshops, developer meetups, small groups. Framing: early OSS contributions were more evenly spread; LLMs + accelerators made hands-on experience harder for individuals and academics. Community infra as a sustainable learning environment; volunteers wanted.

![closing](../../../../assets/vllm/blog/serving/korea-meetup-2025/07-image-6.png)

First gathering: practical, scalable real-world serving. Rebellions, Red Hat, and local engineers committed to more events and upstream work.
