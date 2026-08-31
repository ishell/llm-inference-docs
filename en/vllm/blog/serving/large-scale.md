---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: en
fetched: 2026-08-31
---

# Large Scale Serving: DeepSeek @ 2.2k tok/s/H200

2025-12-17. V0 gone in v0.11.0. Coreweave H200 + IB CX7: ~**2.2k tok/s per H200** (was ~1.5k). Async scheduling, DBO, P/D, `FULL_AND_PIECEWISE` CUDA graphs, DeepGEMM, DeepEP, EPLB, SiLU kernel.

**Wide-EP:** DeepSeek-R1 activates 37B/671B. MLA hates naive TP (duplicated latents). `--enable-expert-parallel` + DP (`mp`/`ray`). DeepEP (and other) all-to-all.

**DBO:** `--enable-dbo` — overlap MoE dispatch/combine across two microbatch workers. Helps when EP comm is fat.

**EPLB:** `--enable-eplb` — sliding-window expert load, shuffle weights without restart.

**P/D:** one fat prefill can stall the whole EP group’s combine. DistServe (2024).

**Deploy:** llm-d (K8s well-lit path), Dynamo, Ray Serve LLM.
