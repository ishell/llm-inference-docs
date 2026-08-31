---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
lang: en
fetched: 2026-08-31
---

# FP8 Quantization

Lower precision (FP8 / int8) usually raises throughput and cuts latency. Quality can drop — always check. Needs compute capability **> 8.9** (Ada, Hopper, Blackwell+). Primer: `mastering-llm-techniques.md`. Case-study numbers are illustrative.

## Enable

Pass `QuantConfig` into `LLM`. Set at least `quant_algo`. Already-quantized checkpoints skip calibration; FP16/BF16 weights need `CalibConfig` for scales.

CLI: `examples/quantization/quantize.py` then `trtllm-build`.

**Leave the GEMM plugin off on quantized engines** (already the default). Keep multiple profiles + paged context FMHA.

```python
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig

def main():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(
        calib_batches=512, calib_batch_size=1,
        calib_max_seq_length=2048, tokenizer_max_seq_length=4096,
    )
    build_config = BuildConfig(max_num_tokens=2048, max_batch_size=512)
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.multiple_profiles = True
    llm = LLM(
        model="/path/to/Llama-3.3-70B",
        tensor_parallel_size=4, pipeline_parallel_size=1,
        build_config=build_config,
        quant_config=quant_config, calib_config=calib_config,
    )
    llm.save("baseline_fp8_engine")

if __name__ == "__main__":
    main()
```

## FP8 “baseline” (tuned batch/tokens, no extra FP8 knobs)

| Metric | Value |
|---|---|
| Token Throughput (tokens/sec) | 3389.5305 |
| Request Throughput (req/sec) | 1.6550 |
| Average TTFT (ms) | 96.1597 |
| Average ITL (ms) | 12.4248 |

## Quantized KV cache

Default KV is not quantized. Quality risk rises; check outputs.

```python
quant_config = QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)
```

CLI: `quantize.py --kv_cache_dtype fp8`

| Metric | Baseline | FP8 KV ON |
|---|---|---|
| Token Throughput (tokens/sec) | 3389.5305 | 5299.6372 |
| Request Throughput (req/sec) | 1.6550 | 2.5877 |
| Average TTFT (ms) | 96.1597 | 97.1287 |
| Average ITL (ms) | 12.4248 | 12.5496 |

## Reduce fusion + user buffers (Llama / Mistral-Mixtral)

User buffers skip an extra copy in the communication kernel. **`user_buffer` requires `reduce_fusion`.**

```python
build_config.plugin_config.reduce_fusion = True
build_config.plugin_config.user_buffer = True
```

CLI: `--reduce_fusion enable --user_buffer enable`

Study retuned max tokens to 16384, batch 512:

| Metric | OFF | ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5299.6372 | 5980.7842 |
| Request Throughput (req/sec) | 2.5877 | 2.9203 |
| Average TTFT (ms) | 97.1287 | 82.2679 |
| Average ITL (ms) | 12.5496 | 12.6975 |

## GEMM + SwiGLU fusion

Two Matmuls + SwiGLU in one kernel. **Hopper FP8 only.** Drops one quantization scale (PTQ accuracy risk). Skip tiny workloads or if quality drops.

```python
build_config.plugin_config.gemm_swiglu_plugin = "fp8"
# or, small-batch latency:
build_config.plugin_config.low_latency_gemm_swiglu_plugin = "fp8"
```

CLI: `--gemm_swiglu_plugin=fp8` **or** `--low_latency_gemm_swiglu_plugin=fp8` (not both).

| Metric | OFF | ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5980.7842 | 5976.7977 |
| Request Throughput (req/sec) | 2.9203 | 2.9184 |
| Average TTFT (ms) | 82.2679 | 81.8841 |
| Average ITL (ms) | 12.6975 | 11.7031 |

Nearly flat alone. Needed as a partner for the next plugin.

## Low-latency GEMM plugin

Do **not** combine with the regular GEMM plugin. Workload-dependent; in the study it **hurt** without SwiGLU fusion (worse kernel on the GEMM before SwiGLU).

```python
build_config.plugin_config.low_latency_gemm_plugin = "fp8"
```

CLI: `--low_latency_gemm_plugin=fp8`

| Metric | OFF | ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5976.7977 | 6049.1625 |
| Request Throughput (req/sec) | 2.9184 | 2.9537 |
| Average TTFT (ms) | 81.8841 | 88.0162 |
| Average ITL (ms) | 11.7031 | 10.8225 |

Throughput and ITL up; TTFT worse. Grid-search combinations if latency is sacred.

## Vs tuned FP16 (previous page)

| Metric | Tuned FP16 | Tuned FP8 | % |
|---|---|---|---|
| Token Throughput (tokens/sec) | 2474.2581 | 6049.1625 | 144.48 |
| Request Throughput (req/sec) | 1.2081 | 2.9537 | 144.49 |
| Average TTFT (ms) | 147.5742 | 88.0162 | 40.36 |
| Average ITL (ms) | 14.6852 | 10.8225 | 26.30 |

Vs this page’s FP8 baseline: token/s **+78%**, TTFT **−8.5%**, ITL **−13%**.

Recommendations: turn on FP8 KV if quality holds; benchmark reduce fusion+user buffers, SwiGLU fusion, and low-latency GEMM (never with regular GEMM plugin).
