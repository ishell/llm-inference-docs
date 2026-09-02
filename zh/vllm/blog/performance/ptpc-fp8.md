---
source: https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# PTPC-FP8：ROCm 上更贴 BF16 的 FP8

英文对照：`en/vllm/blog/performance/ptpc-fp8.md`  
原文：https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm  
vLLM ≥0.7.3。数字是 MI300X 演示。

`--quantization ptpc_fp8`：HF 权重 **现场** 量化，不必预量化。激活 **按 token** 缩放，权重 **按 channel** 缩放——outlier 常钉在同一通道上，per-tensor 会把多数值挤进很少有效 bit。

朴素路径是 GEMM 再乘两路 scale，来回 HBM。ROCm 走融合 `torch._scaled_mm(..., scale_a=token_scales, scale_b=channel_scales)`，他们相对两步最多约 **2.5×**。Llama-3.1-70B SharedGPT 上吞吐和 per-tensor FP8 几乎一样（约 1.01×）。8B Wikitext word perplexity：BF16 9.4281，PTPC 9.5093（+0.86%），标准 FP8 9.5124。GSM8K 8B strict-match：BF16 73.2%，PTPC 70.8%，标准 FP8 69.2%。70B 上 PTPC 有时略高于 BF16——当噪声，别当免费精度。

当时示例还关着 chunked prefill、开着多 step scheduler；旗标以你那版文档为准。和 [FP8 KV](fp8-kvcache.md) 分清：这篇是 **权重量化**，不是 KV dtype。

本地图（原文版权仍归原站；学习对照用）：

![PTPC121](../../../../assets/vllm/blog/performance/ptpc-fp8/01-PTPC121.png)

![PTPC Diagram](../../../../assets/vllm/blog/performance/ptpc-fp8/02-PTPC-Diagram.png)

![FusedGEMM](../../../../assets/vllm/blog/performance/ptpc-fp8/03-FusedGEMM.svg)

![PTPCReqs](../../../../assets/vllm/blog/performance/ptpc-fp8/04-PTPCReqs.svg)

![PTPCSpeedup](../../../../assets/vllm/blog/performance/ptpc-fp8/05-PTPCSpeedup.svg)

![PerplexityBits](../../../../assets/vllm/blog/performance/ptpc-fp8/06-PerplexityBits.png)

![Perplexitywords](../../../../assets/vllm/blog/performance/ptpc-fp8/07-Perplexitywords.png)

![GSM8K8B](../../../../assets/vllm/blog/performance/ptpc-fp8/08-GSM8K8B.png)

![GSM8K70B](../../../../assets/vllm/blog/performance/ptpc-fp8/09-GSM8K70B.png)
