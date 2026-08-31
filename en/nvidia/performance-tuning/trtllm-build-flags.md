---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html
lang: en
fetched: 2026-08-31
---

# TRT-LLM Useful Build-Time Flags

Set via `BuildConfig`; need a rebuild. Case-study numbers are illustrative only.

| Flag | Recommendation |
|---|---|
| **Multiple profiles** | Always on. Helps all metrics; same prompt may pick different kernels under different load (quality OK, not bit-exact). `plugin_config.multiple_profiles = True` or `--multiple_profiles`. Example: token/s 1564 → 1861, ITL 31 → 20 ms. |
| **Paged context attention** | Enable. Chunks prefill across iterations (long ISL). Worst case ~2% hit in naive benches; needed for chunked scheduling + tuning max_num_tokens. `use_paged_context_fmha=True`. |
| **GEMM plugin** | On for FP16/BF16 (`gemm_plugin='auto'`); usually off for FP8. Example: token/s 1867 → 2033, ITL 19.7 → 15.4 ms, slight TTFT bump. |
| **Reduce-norm fusion** | Llama / Mistral-Mixtral + tensor parallel only. Helps generation-heavy work; benchmark yourself. |
| **PP reduce-scatter** | Large MoE + pipeline parallel. |

Together vs baseline in the 70B/2048/2048 study: token/s +31%, ITL −54%, TTFT ~flat.
