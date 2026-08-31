---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html
lang: en
fetched: 2026-08-31
---

# Useful Build-Time Flags

Set via `BuildConfig` / `trtllm-build`. Need a rebuild. Case-study numbers (Llama-3.3-70B, 4×H100, ISL/OSL 2048/2048) are illustrative.

```python
from tensorrt_llm import LLM, BuildConfig

def main():
    build_config = BuildConfig()
    build_config.plugin_config.multiple_profiles = True
    llm = LLM(model="/scratch/Llama-3.3-70B-Instruct",
              tensor_parallel_size=4, build_config=build_config)
    llm.save("build_flags_multiple_profiles")

if __name__ == "__main__":
    main()
```

## Multiple profiles

TensorRT builds engines with optimization profiles (min / optimal / max tensor shapes). TRT-LLM hides that; `max_batch_size` and `max_num_tokens` still influence how profiles are created. Default: **one** profile.

Multiple profiles let the engine pick kernels for the live request load. Longer build, no known runtime downside — **always enable in production**. Same prompt may not be bit-exact across loads (different kernels); quality is expected to hold. Skip only if you need full determinism.

- API: `plugin_config.multiple_profiles = True`
- CLI: `--multiple_profiles`

| Metric | Baseline | Multiple Profiles ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1564.3040 | 1861.0881 |
| Request Throughput (req/sec) | 0.7638 | 0.9087 |
| Average TTFT (ms) | 147.6976 | 145.8958 |
| Average ITL (ms) | 31.3276 | 19.6452 |

## Paged context attention

Default: the whole prompt is one context iteration. This flag chunks prefill across iterations (needed for long ISL and for chunked scheduling). Worst-case naive hit ~<2%. Enable it.

```python
build_config.plugin_config.use_paged_context_fmha = True
```

CLI: `--use_paged_context_fmha`

| Metric | Paged Context OFF | Paged Context ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1861.0881 | 1866.6684 |
| Request Throughput (req/sec) | 0.9087 | 0.9115 |
| Average TTFT (ms) | 145.8958 | 145.4089 |
| Average ITL (ms) | 19.6452 | 19.6523 |

Within run-to-run noise (~10 tok/s, ~2 ms TTFT). Still enable: next page uses it to shrink `max_num_tokens` and give memory back to KV cache.

## GEMM plugin

Custom GEMM via cuBLASLt + custom kernels. **On for FP16/BF16**; **off for FP8**. `'auto'` matches model dtype.

```python
build_config.plugin_config.gemm_plugin = "auto"
```

CLI: `--gemm_plugin auto`

| Metric | GEMM Plugin OFF | GEMM Plugin ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1866.6684 | 2033.2640 |
| Request Throughput (req/sec) | 0.9115 | 0.9928 |
| Average TTFT (ms) | 145.4089 | 147.8307 |
| Average ITL (ms) | 19.6523 | 15.4133 |

Throughput and ITL up; slight TTFT hit.

## Reduce-norm fusion

Fuses ResidualAdd + LayerNorm into the AllReduce kernel. **Llama and Mistral/Mixtral only**, and **tensor parallel only** (PP has no AllReduce). Helps generation-heavy work; check context-heavy jobs yourself.

```python
build_config.plugin_config.reduce_fusion = True
```

CLI: `--reduce_fusion enable`

| Metric | REDUCE FUSION OFF | REDUCE FUSION ON |
|---|---|---|
| Token Throughput (tokens/sec) | 2033.2640 | 2044.2628 |
| Request Throughput (req/sec) | 0.9928 | 0.9982 |
| Average TTFT (ms) | 147.8307 | 146.6628 |
| Average ITL (ms) | 15.4133 | 14.4493 |

Slight win; within variance on reruns.

## PP reduce-scatter

ReduceScatter + AllGather for **large MoE + pipeline parallel**. Not in the Llama case study.

```python
build_config.plugin_config.pp_reduce_scatter = True
```

CLI: `--pp_reduce_scatter`

## Together vs baseline

| Metric | Baseline | Build-Time Flags ON | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 1564.3040 | 2044.2628 | 30.68 |
| Request Throughput (req/sec) | 0.7638 | 0.9982 | 30.69 |
| Average TTFT (ms) | 147.6976 | 146.6628 | 0.70 |
| Average ITL (ms) | 31.3276 | 14.4493 | 53.88 |

**Always:** multiple profiles. **Usually:** paged context FMHA, GEMM plugin on FP16/BF16 (off on FP8). **Benchmark:** reduce fusion, PP reduce-scatter.
