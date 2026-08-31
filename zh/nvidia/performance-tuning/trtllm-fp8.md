---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 第 5 章：FP8 量化

把模型从 FP16/BF16 降到 FP8（或 int8），通常吞吐会涨、延迟会掉。税是质量。许多线上系统靠量化活着，但「可以接受」必须你自己验收，没有人能替你签字。背景仍是 `mastering-llm-techniques.md`。

FP8 需要算力 **> 8.9**：Ada、Hopper、Blackwell 以及更后面的卡。再往前的架构，这扇门是锁的。

数字仍是演示。同一条 70B、四张 H100、2048/2048 的故事继续。

## 怎么开

给 `LLM` 传 `QuantConfig`。至少要设 `quant_algo`（fp8、fp8 per token、int8awq……）。完整枚举在官方 LLM-API Reference。

如果你吃的是已经量化过的 checkpoint，不必再校准。如果你吃的是 FP16/BF16 权重，还要 `CalibConfig` 指定校准数据集，用来估 quantization scale。

CLI：先 `examples/quantization/quantize.py`，再 `trtllm-build`。Llama 例子在 TensorRT-LLM 仓库的 LLaMA examples。

**量化引擎请关掉 GEMM plugin**（默认就是关）。multiple profiles 和 paged context FMHA 继续留着。Reduce fusion 在 FP8 里还有一步「user buffers」，后面单独说。

```python
from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import QuantConfig, QuantAlgo, CalibConfig

def main():
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096,
    )
    build_config = BuildConfig(max_num_tokens=2048, max_batch_size=512)
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.plugin_config.multiple_profiles = True

    llm = LLM(
        model="/path/to/Llama-3.3-70B",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        build_config=build_config,
        quant_config=quant_config,
        calib_config=calib_config,
    )
    llm.save("baseline_fp8_engine")

if __name__ == "__main__":
    main()
```

量化想保住精度，并不保证保住。上线前用你真正在乎的题验收。

## FP8「基线」

下面这组已经带了 multiple profiles、paged context，并且调过 max batch / max tokens——只是还没开那些专为量化准备的加料。官方把它叫做 FP8 baseline，方便对比后面每一勺调料。

| 指标 | Value |
|---|---|
| Token Throughput (tokens/sec) | 3389.5305 |
| Request Throughput (req/sec) | 1.6550 |
| Average TTFT (ms) | 96.1597 |
| Average ITL (ms) | 12.4248 |

对照上一章调完的 FP16（2474 tok/s、TTFT 148 ms）：光是换 FP8，吞吐已经换了一个档。第一个字也从将近 150 ms 掉到大约 96 ms。

## 量化 KV cache

默认 KV **不**量化。把 KV 也打成 FP8，吞吐往往再跳一截。量化越狠，质量风险越高。必须再验一遍输出。

```python
quant_config = QuantConfig(
    quant_algo=QuantAlgo.FP8,
    kv_cache_quant_algo=QuantAlgo.FP8,
)
```

CLI：`quantize.py --kv_cache_dtype fp8`

| 指标 | Baseline | FP8 KV-Cache ON |
|---|---|---|
| Token Throughput (tokens/sec) | 3389.5305 | 5299.6372 |
| Request Throughput (req/sec) | 1.6550 | 2.5877 |
| Average TTFT (ms) | 96.1597 | 97.1287 |
| Average ITL (ms) | 12.4248 | 12.5496 |

吞吐从 3389 到 5300。TTFT / ITL 几乎不动。房子变小了，同面积能住更多请求。

## Reduce-norm fusion + user buffers

FP8 也支持 reduce-norm fusion。另外还有 **user buffers**：少一次从 local buffer 拷到 shared buffer，通信 kernel 更干净。**必须先开 `reduce_fusion`，才能开 `user_buffer`。** 仅 Llama / Mistral-Mixtral。

```python
build_config.plugin_config.reduce_fusion = True
build_config.plugin_config.user_buffer = True
```

CLI：`trtllm-build --reduce_fusion enable --user_buffer enable`

案例在打开之后把 max-num tokens 调到 16384、max-batch 仍是 512：

| 指标 | Fusion+UB OFF | Fusion+UB ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5299.6372 | 5980.7842 |
| Request Throughput (req/sec) | 2.5877 | 2.9203 |
| Average TTFT (ms) | 97.1287 | 82.2679 |
| Average ITL (ms) | 12.5496 | 12.6975 |

吞吐再涨一截，TTFT 从 97 ms 掉到 82 ms。ITL 几乎没动。效果看负载，官方仍要你自己测。

## GEMM + SwiGLU fusion

把 Gated-MLP 里的两次 Matmul 和一次 SwiGLU 收进同一个 kernel。**目前仅 Hopper 上的 FP8。** 融合时会丢掉一个 quantization scale，FP8 PTQ 精度可能略掉。

大模型、Hopper、FP8：值得试。很小的负载，或精度掉到不可接受：别开。

```python
build_config.plugin_config.gemm_swiglu_plugin = "fp8"
```

低延迟、小 batch 可换成：

```python
build_config.plugin_config.low_latency_gemm_swiglu_plugin = "fp8"
```

两个只开一个。CLI：`--gemm_swiglu_plugin=fp8` 或 `--low_latency_gemm_swiglu_plugin=fp8`。

| 指标 | Fusion OFF | Fusion ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5980.7842 | 5976.7977 |
| Request Throughput (req/sec) | 2.9203 | 2.9184 |
| Average TTFT (ms) | 82.2679 | 81.8841 |
| Average ITL (ms) | 12.6975 | 11.7031 |

单独开，这个案例几乎持平（吞吐落在方差里，ITL 略好）。但下一节的 low-latency GEMM **必须**配它，才能摸到峰值——单独开 low-latency GEMM 反而更差。旗标会打架。这就是为什么官方反复说：网格搜，不要只拧一只旋钮。

## Low-latency GEMM plugin

普通 GEMM plugin 对 FP8 **建议关**。低延迟场景另有一颗：`low_latency_gemm_plugin`。**不要**和普通 `gemm_plugin` 一起开。

```python
build_config.plugin_config.low_latency_gemm_plugin = "fp8"
```

CLI：`--low_latency_gemm_plugin=fp8`。若你还在传 `--gemm_plugin=fp8`，拿掉。

案例在 SwiGLU fusion 之后打开，max tokens 16384、batch 512：

| 指标 | Low Latency GEMM OFF | Low Latency GEMM ON |
|---|---|---|
| Token Throughput (tokens/sec) | 5976.7977 | 6049.1625 |
| Request Throughput (req/sec) | 2.9184 | 2.9537 |
| Average TTFT (ms) | 81.8841 | 88.0162 |
| Average ITL (ms) | 11.7031 | 10.8225 |

吞吐略涨，ITL 更好，TTFT 变差——decode 高兴，第一个字多等了一点。官方的解读：没有 SwiGLU fusion 时，这颗 plugin 可能给 SwiGLU 前那次 GEMM 选了更差的 kernel；fusion 把那次 GEMM 接走之后，剩下的 kernel 才比基线好。负载一变，故事会改。性能敏感的服务，值得把组合扫一遍。

## 和调过的 FP16 比

| 指标 | Tuned FP16 | Tuned FP8 | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 2474.2581 | 6049.1625 | 144.48 |
| Request Throughput (req/sec) | 1.2081 | 2.9537 | 144.49 |
| Average TTFT (ms) | 147.5742 | 88.0162 | 40.36 |
| Average ITL (ms) | 14.6852 | 10.8225 | 26.30 |

相对「已经调过 batch/tokens 的 FP8 基线」：

| 指标 | Baseline FP8 | Tuned FP8 | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 3389.5305 | 6049.1625 | 78.47 |
| Request Throughput (req/sec) | 1.6550 | 2.9537 | 78.47 |
| Average TTFT (ms) | 96.1597 | 88.0162 | 8.47 |
| Average ITL (ms) | 12.4248 | 10.8225 | 12.90 |

token/s 相对调过的 FP16 大约 **+144%**，TTFT **−40%**，ITL **−26%**。税还是那句：质量必须你自己测。许多真实部署付得起这点税；付不起的那些，会在第一轮评测里被抓出来。

### 建议

| 选项 | 建议 |
|---|---|
| FP8 KV cache | 通常吞吐大涨。质量过关就开。 |
| Reduce fusion + user buffers | 仅 FP8 Llama / Mistral-Mixtral。先开再测。user buffers 依赖 reduce fusion。 |
| GEMM + SwiGLU | 仅 Hopper FP8、带 SwiGLU 的模型。会丢一个 scale。自己看精度。 |
| Low-latency GEMM | 关普通 GEMM plugin。效果看负载，且会跟其他旗标耦合。能网格搜就搜。 |
