---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
lang: en
fetched: 2026-08-31
---

# FP8 Quantization (TensorRT-LLM)

Lower precision (int8/fp8) usually raises throughput and cuts latency; quality can drop — always check. FP8 needs compute capability **> 8.9** (Ada, Hopper, Blackwell+). Background: `mastering-llm-techniques.md`.

## Enable

`QuantConfig(quant_algo=QuantAlgo.FP8)` on `LLM(...)`. BF16/FP16 checkpoints also need `CalibConfig` (calibration dataset / scales). Already-quantized checkpoints skip calibration.

```python
quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
calib_config = CalibConfig(calib_batches=512, calib_batch_size=1,
                           calib_max_seq_length=2048, tokenizer_max_seq_length=4096)
llm = LLM(model=..., tensor_parallel_size=4, build_config=build_config,
          quant_config=quant_config, calib_config=calib_config)
```

CLI: `examples/quantization/quantize.py` then `trtllm-build`.

**Quantized engines: leave GEMM plugin off** (already the default). Keep multiple profiles + paged context FMHA from the earlier tuning pages.

## Case study extras (Llama-3.3-70B TP=4 on H100)

Official demo numbers (environment-specific):

| Setup | Token TPS | TTFT ms | ITL ms |
|---|---|---|---|
| FP8 baseline (tuned batch/tokens, no extra fp8 flags) | 3389 | 96.2 | 12.4 |
| + FP8 KV cache | 5300 | 97.1 | 12.5 |
| + reduce fusion + user buffers | 5981 | 82.3 | 12.7 |
| + GEMM+SwiGLU fusion | ~5977 | 81.9 | 11.7 |
| + low-latency GEMM plugin | 6049 | 88.0 | 10.8 |

- **FP8 KV:** `QuantConfig(..., kv_cache_quant_algo=QuantAlgo.FP8)` or `--kv_cache_dtype fp8`. Big throughput win; quality risk higher.
- **Reduce fusion + user buffers** (Llama): `plugin_config.reduce_fusion = True` then `user_buffer = True`. User buffers need reduce fusion.
- **GEMM + SwiGLU fusion:** Hopper FP8 only; `gemm_swiglu_plugin = 'fp8'`. Can drop a scale (PTQ accuracy). For tiny/latency jobs try `low_latency_gemm_swiglu_plugin`.
- **Low-latency GEMM plugin:** `low_latency_gemm_plugin = 'fp8'`. Do **not** also turn on the regular GEMM plugin. Impact is workload-dependent; in the study it needed SwiGLU fusion or it got worse.

Vs tuned FP16 in that guide: **~+144% token TPS**, TTFT **−40%**, ITL **−26%**. Still verify quality.
