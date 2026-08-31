---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html
lang: zh
fetched: 2026-08-31
---

# TRT-LLM 编译期开关

走 `BuildConfig`，改了要重建引擎。案例数字仅作量级参考。

| 开关 | 建议 |
|---|---|
| **Multiple profiles** | 生产建议开。各指标都涨；同 prompt 在不同负载下可能走不同 kernel（质量一般不变，但不是 bit-exact）。例：token/s 1564→1861，ITL 31→20 ms。 |
| **Paged context attention** | 建议开。把 prefill 切块跨 iteration（长 ISL）。最差 naive 基准约 −2%；后面调 `max_num_tokens` 需要它。 |
| **GEMM plugin** | FP16/BF16 开（`'auto'`）；FP8 通常关。例：token/s 1867→2033，ITL 19.7→15.4 ms，TTFT 略升。 |
| **Reduce-norm fusion** | 仅 Llama / Mistral-Mixtral，且要用 TP。生成阶段重时更有用，自己测。 |
| **PP reduce-scatter** | 大 MoE + 流水线并行。 |

70B / 2048/2048 案例相对基线：token/s +31%，ITL −54%，TTFT 几乎不变。
