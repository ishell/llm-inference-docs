---
source: https://vllm.ai/blog/2026-08-12-qwen3.8
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Qwen3.8-2.4T：Max 级开权，引擎不用换骨架

英文对照：[en/vllm/blog/serving/qwen38.md](../../../../en/vllm/blog/serving/qwen38.md)  
原文：https://vllm.ai/blog/2026-08-12-qwen3.8  
2026-08-12。数字是演示。

Qwen 家第一次把 Qwen-Max 级开出来：`Qwen3.8-2.4T-A95B`。骨架仍是 Qwen 3.5——512 expert 的稀疏 MoE，92 层里每 4 层一次 full attention，其余 69 层 linear attention。vLLM 声明 **day-0、无新架构**。这篇是起服配方，不是 PagedAttention 解剖。

## 精度与硬件

官方 FP8 / BF16；Inferact 另放 NVFP4、MXFP4。FP4 用 RTN + activation calibration，routed expert 砍到 4-bit。硬件：至少两台 B300 / MI355X；FP4 单机可试。NVIDIA 侧 Linear Attention（Gated Delta Rule）、GQA、Dense GEMM、MoE routing 有共研 kernel，Attention 走 DP+TP、MoE 走 EP。AMD 侧 AITER-fused Gated DeltaNet decode、hipBLASLt 共享 expert、AITER FusedMoE、Quark MXFP4。

## 怎么起（当时）

NVFP4 示例：`--linear-backend flashinfer_cutedsl`、`--tensor-parallel-size 8`、`--tool-call-parser qwen3_coder`、`--reasoning-parser qwen3`、`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。MXFP4 同套 parser / MTP，不必 `flashinfer_cutedsl`。完整 docker 看 recipes。生成：`temperature=1.0, top_p=0.95, top_k=20`；reasoning 要把 `max_tokens` 开大。

## 数字（演示）

提高 reasoning budget 才能复现评测。GSM8K strict/flexible：FP8 **89.61% / 90.52%**，NVFP4 **90.37% / 91.05%**。AIME25 @3 avg/pass：FP8 **87.78% / 93.33%**，NVFP4 **92.22% / 96.67%**——量化不是这篇的精度故事，预算才是。
