---
source: https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DeepSeek-R1 上 GB200：Wide-EP 第二张成绩单

英文对照：`en/vllm/blog/serving/gb200-wideep.md`  
原文：https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1  
接 [Wide-EP](large-scale.md) 的 H200 线（约 2.2k tok/s/H200）。数字是当时演示，不是你机器的承诺。

GB200 NVL72 上他们报到：**26.2K prefill TPGS**、**10.1K decode TPGS**（2K/2K 输入输出）。拓扑：4×(2 GPU) prefill + 1×(8 GPU) decode。NVFP4 GEMM、FP8 MLA、NVFP4 dispatch、融合、weight offload v2。GB200 上关掉一部分 chunking——那代互联和 kernel 形状跟 H200 不一样，H200 笔记里的开关不要原样粘过来。

和 [EPD](epd.md) 分清：这里是 **文本 prefill/decode 分拆 + 宽 EP**，不是视觉 encoder 分拆。

本地图（原文版权仍归原站；学习对照用）：

![topline comparison](../../../../assets/vllm/blog/serving/gb200-wideep/01-topline_comparison.png)

![decode throughput various](../../../../assets/vllm/blog/serving/gb200-wideep/02-decode_throughput_various.png)

![rope quant fusion timeline](../../../../assets/vllm/blog/serving/gb200-wideep/03-rope_quant_fusion_timeline.png)

![mla trtllm ragged prefill prefill](../../../../assets/vllm/blog/serving/gb200-wideep/04-mla_trtllm_ragged_prefill_prefill.png)

![moe flashinfer trtllm nvfp4 prefill](../../../../assets/vllm/blog/serving/gb200-wideep/05-moe_flashinfer_trtllm_nvfp4_prefill.png)

![nccl all gather](../../../../assets/vllm/blog/serving/gb200-wideep/06-nccl_all_gather.png)

![nccl reduce scatter](../../../../assets/vllm/blog/serving/gb200-wideep/07-nccl_reduce_scatter.png)

![layer group](../../../../assets/vllm/blog/serving/gb200-wideep/08-layer_group.png)

![onloading trace](../../../../assets/vllm/blog/serving/gb200-wideep/09-onloading_trace.png)
