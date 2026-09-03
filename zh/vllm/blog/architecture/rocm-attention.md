---
source: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# ROCm 上的七条 attention 路

英文对照：[en/vllm/blog/architecture/rocm-attention.md](../../../../en/vllm/blog/architecture/rocm-attention.md)  
原文：https://vllm.ai/blog/2026-02-27-rocm-attention-backend  
Triton attention 默认见 [triton-attn](triton-attn.md)；这篇讲 **AITER FA 三路**。

`ROCM_AITER_FA` 把 batch 拆成 Prefill / Extend / Decode：新序列走 `flash_attn_varlen_func`（CDNA 矩阵核）；续写长上下文走切块 + LSE merge（约 32K token/iteration 预算）；decode 走 AITER `pa_fwd_asm`。Model runner 把请求排成 `[decode:extend:prefill]`（`reorder_batch_threshold=1`），三路内核吃连续内存。

KV 预洗牌：

```
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

Decode 零 layout 转换，相对标准 layout 约 **15–20%** decode TPS。Extend 要从洗牌 cache gather 回标准 layout 再 MHA。官方相对其他 ROCm backend 约 **1.2–4.4×** 系统 TPS——硬件和模型不同，别当承诺。软件三路而不是一只万能 kernel：好 debug、MI300X→MI355X 同一套路由。

本地图（原文版权仍归原站；学习对照用）：

![continuous batching](../../../../assets/vllm/blog/architecture/rocm-attention/01-continuous-batching.png)

![ROCm Attention unified attn](../../../../assets/vllm/blog/architecture/rocm-attention/02-ROCm-Attention-unified-attn.png)

![ROCm Attention rocm aiter fa](../../../../assets/vllm/blog/architecture/rocm-attention/03-ROCm-Attention-rocm_aiter_fa.png)

![batch reordering](../../../../assets/vllm/blog/architecture/rocm-attention/04-batch_reordering.png)

![chunked context flow](../../../../assets/vllm/blog/architecture/rocm-attention/05-chunked_context_flow.png)

![mha tpot comparison](../../../../assets/vllm/blog/architecture/rocm-attention/06-mha_tpot_comparison.png)

![mha ttft comparison](../../../../assets/vllm/blog/architecture/rocm-attention/07-mha_ttft_comparison.png)

![mha tps comparison](../../../../assets/vllm/blog/architecture/rocm-attention/08-mha_tps_comparison.png)

![mla tpot comparison](../../../../assets/vllm/blog/architecture/rocm-attention/09-mla_tpot_comparison.png)

![mla ttft comparison](../../../../assets/vllm/blog/architecture/rocm-attention/10-mla_ttft_comparison.png)

![mla tps comparison](../../../../assets/vllm/blog/architecture/rocm-attention/11-mla_tps_comparison.png)

![system stack](../../../../assets/vllm/blog/architecture/rocm-attention/12-system_stack.png)

![innovation attribution](../../../../assets/vllm/blog/architecture/rocm-attention/13-innovation_attribution.png)
