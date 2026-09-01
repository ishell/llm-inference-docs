---
source: https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
lang: en
fetched: 2026-09-01
---

# DeepSeek-R1 on GB200: Wide-EP’s second scoreboard

Chinese: `../../zh/vllm/blog/serving/gb200-wideep.md`  
Follows [Wide-EP](wide-ep.md) on H200 (~2.2k tok/s/H200). Demo numbers, not a promise for your box.

On GB200 NVL72 they reported **26.2K prefill TPGS** and **10.1K decode TPGS** (2K/2K in/out). Topology: 4×(2 GPU) prefill + 1×(8 GPU) decode. NVFP4 GEMM, FP8 MLA, NVFP4 dispatch, fusion, weight offload v2. Some chunking is **off** on GB200 — interconnect and kernel shapes differ from H200; do not paste H200 flags blindly.

Not [EPD](epd.md): this is **text prefill/decode split + wide EP**, not visual-encoder split.
