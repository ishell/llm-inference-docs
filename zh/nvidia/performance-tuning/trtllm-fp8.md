---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
lang: zh
fetched: 2026-08-31
---

# FP8 量化（TensorRT-LLM）

低精度（int8/fp8）通常提高吞吐、降低延迟，质量可能掉——必须验收。FP8 需要算力 **> 8.9**（Ada、Hopper、Blackwell+）。背景：`mastering-llm-techniques.md`。

## 怎么开

`LLM(...)` 上传 `QuantConfig(quant_algo=QuantAlgo.FP8)`。BF16/FP16 权重还要 `CalibConfig` 做 calibration。已经量化过的 checkpoint 不用。

CLI：先 `examples/quantization/quantize.py` 再 `trtllm-build`。

**量化引擎：关掉 GEMM plugin**（默认就是关）。前面调优页的 multiple profiles、paged context FMHA 继续留着。

## 案例加料（Llama-3.3-70B，TP=4，H100）

官方演示数字（环境相关）：

| 配置 | Token TPS | TTFT ms | ITL ms |
|---|---|---|---|
| FP8 baseline（batch/tokens 已调） | 3389 | 96.2 | 12.4 |
| + FP8 KV cache | 5300 | 97.1 | 12.5 |
| + reduce fusion + user buffers | 5981 | 82.3 | 12.7 |
| + GEMM+SwiGLU fusion | ~5977 | 81.9 | 11.7 |
| + low-latency GEMM plugin | 6049 | 88.0 | 10.8 |

- **FP8 KV：** `kv_cache_quant_algo=QuantAlgo.FP8` 或 `--kv_cache_dtype fp8`。吞吐大涨，质量风险更高。
- **Reduce fusion + user buffers**（Llama）：先 `reduce_fusion=True` 再 `user_buffer=True`。
- **GEMM + SwiGLU：** 仅 Hopper FP8；`gemm_swiglu_plugin='fp8'`。会丢掉一个 scale（PTQ 精度）。小负载/低延迟可试 `low_latency_gemm_swiglu_plugin`。
- **Low-latency GEMM：** `low_latency_gemm_plugin='fp8'`。**不要**同时开普通 GEMM plugin。效果看负载；该案例里必须配合 SwiGLU fusion，否则更差。

相对该指南里调过的 FP16：token TPS 约 **+144%**，TTFT **−40%**，ITL **−26%**。质量仍要自己测。
