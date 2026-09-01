---
source: https://vllm.ai/blog/2026-02-13-gb300-deepseek
lang: en
fetched: 2026-09-01
---

# DeepSeek-V3.2 on GB300: deployment validation, not a peak hunt

Chinese: `../../zh/vllm/blog/performance/gb300-deepseek.md`  
vLLM 0.14.1, CUDA 13.0. GB300/B300 288GB. Reproducible **baseline**, not tuned-to-the-wall.

`VLLM_USE_FLASHINFER_MOE_FP4=1`. Two GPUs hold NVFP4. V3.2 TP2 prefill-only **7360 TGS**/GPU; ISL=2k/OSL=1k output **2816 TGS**. R1 NVFP4+EP2 two-GPU prefill **22476 TGS** (ISL=2k/OSL=1/batch=256); mixed **3072 TGS**. vs Hopper: prefill ~**8×**, mixed ~**10–20×**. NVFP4+TP2 beats FP8 — TP4 thins per-GPU work so Tensor Cores starve. On R1, EP fits prefiller; colocated P+D with large ISL / small OSL prefers TP2 so attention does not crowd decode.

MTP (`num_speculative_tokens 1`) helps decode at modest concurrency and context; high concurrency or tiny decode share cannot amortize it. V3.2 prefill ~**⅓** of R1: Indexer/Sparse MLA, DSA layer ~**2.7×** MLA kernel time. DSA’s TPOT win shows around **10k–20k** context. 1P1D / 3P1D beat colocated on throughput and TPOT slope; ISL 2k→8k starves decode on 1P1D — add P. v0.14.1 P/D still needed #32698 by hand.
