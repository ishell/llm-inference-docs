---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 第 2 章：编译期旗标

这些开关写进引擎的骨头里。改了就要重建。LLM-API 走 `BuildConfig`；CLI 走 `trtllm-build`。完整清单在官方 Command Line Reference。

数字仍然是演示。环境、SKU、互联、负载一变，涨幅会换脸。

上一章留下的那张成绩单——token/s 1564、ITL 31 ms——是这一页的对照物。下面每一项都先讲它做什么，再给出怎么开，再拿同一套 70B / 4×H100 / 2048/2048 去撞一次秒表。

```python
from tensorrt_llm import LLM, BuildConfig

def main():
    build_config = BuildConfig()
    build_config.plugin_config.multiple_profiles = True
    llm = LLM(
        model="/scratch/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4,
        build_config=build_config,
    )
    llm.save("build_flags_multiple_profiles")

if __name__ == "__main__":
    main()
```

## Multiple profiles

TensorRT 用 **optimization profile** 描述输入张量的 min / optimal / max 形状。它为 optimal 优化，同时还能在 min–max 之间活着。TensorRT-LLM 把 profile 的制造藏起来了，但 `max_batch_size` 和 `max_num_tokens`（下一章）会悄悄参与。默认只造 **一个** profile。

线上的请求负载会把形状拧来拧去。开多个 profile，引擎可以按当下的形状选更合适的 kernel。**编译更久，运行时没有已知的副作用。生产建议永远开。**

唯一要记住的：同一句 prompt，在不同负载下可能走进不同的 kernel。输出不必 bit-exact，质量通常不受伤。你若需要完全确定性，就别开。

- LLM-API：`build_config.plugin_config.multiple_profiles = True`
- CLI：`trtllm-build --multiple_profiles`

| 指标 | Baseline | Multiple Profiles ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1564.3040 | 1861.0881 |
| Request Throughput (req/sec) | 0.7638 | 0.9087 |
| Average TTFT (ms) | 147.6976 | 145.8958 |
| Average ITL (ms) | 31.3276 | 19.6452 |

几乎全线变好。ITL 从 31 ms 掉到 20 ms——decode 突然会走路了。

## Paged context attention

默认：一条新请求的 prompt 在一次 iteration 里全部做完（整段 context phase）。打开 paged context attention 之后，prefill 可以被切成块，摊到好几次 iteration。长 ISL 尤其需要它。最差情况，naive 基准大约掉 **不到 2%**，所以可以放心开。下一章会把它和调度器、`max_num_tokens` 绑在一起讲。

在上一节的例子里加一行：

```python
build_config.plugin_config.use_paged_context_fmha = True
```

CLI：`trtllm-build --use_paged_context_fmha`

| 指标 | Paged Context OFF | Paged Context ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1861.0881 | 1866.6684 |
| Request Throughput (req/sec) | 0.9087 | 0.9115 |
| Average TTFT (ms) | 145.8958 | 145.4089 |
| Average ITL (ms) | 19.6452 | 19.6523 |

这次几乎是噪声：官方复跑大约 ±10 tok/s、TTFT ±2 ms，ITL 稳定在 1 ms 以内。有的负载 naive 打开甚至会略慢。仍然建议开——因为 chunked prefill 让你敢于把 `max_num_tokens` 调小，把显存还给 KV。那才是真正的杠杆。

## GEMM plugin

TensorRT 允许用 **plugin** 替换它自己挑的 kernel。TensorRT-LLM 有一堆为支持的模块写的自定义 kernel。GEMM plugin 走 NVIDIA cuBLASLt 和一些定制 GEMM。

- **FP16 / BF16：建议开**，通常更快、显存更省。
- **FP8：建议关**（默认就是关）。

```python
build_config.plugin_config.gemm_plugin = "auto"
```

`'auto'` 表示 GEMM 精度跟模型走。除非你在做混合精度，否则别手填。

CLI：`trtllm-build --gemm_plugin auto`

| 指标 | GEMM Plugin OFF | GEMM Plugin ON |
|---|---|---|
| Token Throughput (tokens/sec) | 1866.6684 | 2033.2640 |
| Request Throughput (req/sec) | 0.9115 | 0.9928 |
| Average TTFT (ms) | 145.4089 | 147.8307 |
| Average ITL (ms) | 19.6523 | 15.4133 |

吞吐和 ITL 明显变好，TTFT 略升——decode 高兴，第一个字稍微多等了一点。

## Reduce-norm fusion（Llama / Mistral）

TensorRT-LLM 默认就有更快的 AllReduce kernel。这个旗标再往前一步：把 AllReduce **后面** 的 ResidualAdd 和 LayerNorm **融进同一个 kernel**。

约束很窄：

- 目前主要是 **Llama**，以及 **Mistral / Mixtral**。
- 必须有 **tensor parallelism**。纯 pipeline parallelism 没有 AllReduce，开了白开。
- 生成阶段重的负载更受益；极端 context-heavy 的工作，开关各跑一次再决定。

```python
build_config.plugin_config.reduce_fusion = True
```

CLI：`trtllm-build --reduce_fusion enable`

| 指标 | REDUCE FUSION OFF | REDUCE FUSION ON |
|---|---|---|
| Token Throughput (tokens/sec) | 2033.2640 | 2044.2628 |
| Request Throughput (req/sec) | 0.9928 | 0.9982 |
| Average TTFT (ms) | 147.8307 | 146.6628 |
| Average ITL (ms) | 15.4133 | 14.4493 |

2048/2048 上略好。官方复跑时，最差情况与不开持平——落在 run-to-run 方差里。自己测。

## Pipeline parallel reduce-scatter

给 **大 MoE + pipeline parallelism** 的优化：ReduceScatter + AllGather。Llama 不是 MoE，案例里没开。

```python
build_config.plugin_config.pp_reduce_scatter = True
```

CLI：`trtllm-build --pp_reduce_scatter`

## 合在一起

| 指标 | Baseline | Build-Time Flags ON | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 1564.3040 | 2044.2628 | 30.68 |
| Request Throughput (req/sec) | 0.7638 | 0.9982 | 30.69 |
| Average TTFT (ms) | 147.6976 | 146.6628 | 0.70 |
| Average ITL (ms) | 31.3276 | 14.4493 | 53.88 |

token/s 大约 **+31%**，ITL 大约 **−54%**，TTFT 几乎不动。第一个字还在等同一场雨，后面的字开始跑起来了。

### 建议（官方原意）

| 旗标 | 建议 |
|---|---|
| Multiple profiles | **永远开。** 编译久一点。负载不同时同 prompt 可能不 bit-exact，质量通常没事。 |
| Paged context attention | **开。** 最差略慢；调 `max_num_tokens` 时几乎是前提。 |
| GEMM plugin | FP16/BF16 开，FP8 关。仍要在你的负载上复核。 |
| Reduce fusion | 仅 Llama / Mistral-Mixtral + TP。自己测。 |
| PP reduce-scatter | 大 MoE + PP。 |

下一章把调度器从黑箱里拖出来：`max_batch_size` 和 `max_num_tokens`。
