---
source: https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
lang: en
fetched: 2026-09-01
---

# DeepSeek-R1 on GB200: Wide-EP’s second scoreboard

Chinese: `../../zh/vllm/blog/serving/gb200-wideep.md`  
Follows [Wide-EP](large-scale.md) on H200 (~2.2k tok/s/H200). Demo numbers, not a promise for your box.

On GB200 NVL72 they reported **26.2K prefill TPGS** and **10.1K decode TPGS** (2K/2K in/out). Topology: 4×(2 GPU) prefill + 1×(8 GPU) decode. NVFP4 GEMM, FP8 MLA, NVFP4 dispatch, fusion, weight offload v2. Some chunking is **off** on GB200 — interconnect and kernel shapes differ from H200; do not paste H200 flags blindly.

Not [EPD](epd.md): this is **text prefill/decode split + wide EP**, not visual-encoder split.

Local figures (copyright remains with the original site; study copies):

![topline comparison](../../../../assets/vllm/blog/serving/gb200-wideep/01-topline_comparison.png)

![decode throughput various](../../../../assets/vllm/blog/serving/gb200-wideep/02-decode_throughput_various.png)

![rope quant fusion timeline](../../../../assets/vllm/blog/serving/gb200-wideep/03-rope_quant_fusion_timeline.png)

![mla trtllm ragged prefill prefill](../../../../assets/vllm/blog/serving/gb200-wideep/04-mla_trtllm_ragged_prefill_prefill.png)

![moe flashinfer trtllm nvfp4 prefill](../../../../assets/vllm/blog/serving/gb200-wideep/05-moe_flashinfer_trtllm_nvfp4_prefill.png)

![nccl all gather](../../../../assets/vllm/blog/serving/gb200-wideep/06-nccl_all_gather.png)

![nccl reduce scatter](../../../../assets/vllm/blog/serving/gb200-wideep/07-nccl_reduce_scatter.png)

![layer group](../../../../assets/vllm/blog/serving/gb200-wideep/08-layer_group.png)

![onloading trace](../../../../assets/vllm/blog/serving/gb200-wideep/09-onloading_trace.png)
