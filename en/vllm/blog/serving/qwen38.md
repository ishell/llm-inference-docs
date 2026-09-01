---
source: https://vllm.ai/blog/2026-08-12-qwen3.8
lang: en
fetched: 2026-09-01
---

# Qwen3.8-2.4T day-0

Chinese: `../../zh/vllm/blog/serving/qwen38.md`  
2026-08-12. Study note; figures on the original page. Demo numbers.

First Qwen-Max-class open weights: `Qwen3.8-2.4T-A95B`. Same Qwen 3.5 skeleton — 512-expert sparse MoE, full attention every 4th of 92 layers, 69 linear-attention layers. **No new engine architecture.** Serving recipe.

**Quant:** official FP8/BF16; Inferact NVFP4/MXFP4 (RTN + activation calibration on routed experts). **HW:** ≥2× B300 / MI355X; FP4 can fit one node. NVIDIA: Gated Delta Rule / GQA / GEMM / MoE kernels, DP+TP attention, EP for MoE. AMD: AITER Gated DeltaNet, hipBLASLt shared experts, AITER FusedMoE, Quark MXFP4.

```bash
vllm serve Inferact/Qwen3.8-2.4T-A95B-NVFP4 \
  --linear-backend flashinfer_cutedsl \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

MXFP4 drops `--linear-backend flashinfer_cutedsl`. Sampling: `temperature=1.0, top_p=0.95, top_k=20`; give reasoning a large `max_tokens`.

Demo (raise reasoning budget): GSM8K strict/flexible FP8 **89.61%/90.52%**, NVFP4 **90.37%/91.05%**. AIME25 @3 avg/pass FP8 **87.78%/93.33%**, NVFP4 **92.22%/96.67%**.
