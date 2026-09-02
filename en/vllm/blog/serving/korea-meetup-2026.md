---
source: https://vllm.ai/blog/2026-04-14-vllm-korea-meetup-2026
lang: en
fetched: 2026-09-01
---

# Korea Meetup 2026: V1, playground, NPU plugins, Omni pipeline split

Chinese: `../../zh/vllm/blog/serving/korea-meetup-2026.md`  
Seoul, 2026-04-02. Rebellions / SqueezeBits / Red Hat APAC.

V0→V1, async scheduling, streaming, Semantic Router, vLLM-Omni. Li Ming on playground (GUI over 140+ knobs). Rebellions: `vllm-rbln` already paged attention / continuous batch; spec-decode, distributed KV, P/D still incoming. SqueezeBits on production-stack. XCENA: LMCache + CXL as a KV expansion tier. Upstage: chat templates / parsers are what make it shippable. Samsung: internal GPUs + air-gap, 4000+ employees. NAVER HyperCLOVA Omni: split encoder/LLM/decoder; vision decoder dominates latency; sequence parallel + kernels ~**3×**. Community notes, not a kernel paper. 2025: [korea-meetup-2025](korea-meetup-2025.md).

Local figures (copyright remains with the original site; study copies):

![banner](../../../../assets/vllm/blog/serving/korea-meetup-2026/01-banner.jpg)

![networking](../../../../assets/vllm/blog/serving/korea-meetup-2026/02-networking.jpg)

![intro liming](../../../../assets/vllm/blog/serving/korea-meetup-2026/03-intro_liming.jpg)

![intro hongseok](../../../../assets/vllm/blog/serving/korea-meetup-2026/04-intro_hongseok.jpg)

![production stack](../../../../assets/vllm/blog/serving/korea-meetup-2026/05-production_stack.jpg)

![closing](../../../../assets/vllm/blog/serving/korea-meetup-2026/06-closing.jpg)
