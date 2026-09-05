---
source: https://vllm.ai/blog/2024-07-23-llama31
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Llama 3.1：128K 自动 chunked prefill；405B 单机走 FP8，多机优先 PP

英文对照：[en/vllm/blog/serving/llama31.md](../../../../en/vllm/blog/serving/llama31.md)  
原文：https://vllm.ai/blog/2024-07-23-llama31  
2024-07-23。**vLLM Team**。和 Meta 的合作帖。数字是 **早期** 参考——他们自己说几周内还会涨。后来的 Llama 4：[llama4.md](llama4.md)。chunked prefill / 分布式细节在文档里，这里不搬。当时镜像：`vllm/vllm-openai`。安装：`pip install -U vllm`。

Llama 3.1：上下文到 **128K**，参数到 **405B**。vLLM 补上 chunked prefill、FP8、pipeline parallelism，好让更长、更大的 Llama 跑得动。

128K：vLLM **自动打开 chunked prefill**。控显存，也减少长 prompt 打断正在 decode 的请求。

## 405B：他们列的几条路

- **FP8：** 官方 FP8，**8×A100 或 8×H100**
- **Pipeline parallelism：** 官方 BF16，层放到不同节点
- **Tensor parallelism：** 跨节点、跨卡切
- **AMD MI300x 或 NVIDIA H200：** 单机 **8×MI300x**（192 GB）或 **8×H200**（141 GB）
- **CPU offloading：** 最后一招，精度不砍、GPU 紧

当时建议：单机 **FP8**，多机 **pipeline parallelism**。更多量化和 PP 吞吐还在探。

## FP8

H100 / MI300x 有原生 FP8 Tensor Core。当时 vLLM：KV cache、attention、MLP 都能走 FP8。官方 Meta Llama 3.1 405B FP8 经 FBGEMM：MLP 的 up/gate/down **按 channel** 量化，静态 scale；跳过第一层和最后一层；再加静态上界。精度掉得很少——是他们的说法。

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tensor-parallel-size 8
```

他们报的负载：平均输入 **1024**，平均输出 **128**：

| 指标 | 值 |
|---|---|
| 请求 | **2.82 req/s** |
| 输入 | **2884.86 tok/s** |
| 输出 | **291.53 tok/s** |

GSM8K，lm-eval-harness，8-shot CoT，exact match：FP8 **95.38%（±0.56%）**，BF16 官方 **96.8%**。

## Pipeline parallelism

不量化的 405B 用 **16×H100 或 16×A100**。PP 按层组切；点对点通信，不必 all-reduce。节点之间没有 InfiniBand 时特别有用。

PP 可以和 TP 叠。例：16 卡 / 2 节点，**PP2 × TP8**——每节点半个模型，节点内 NVLink all-reduce：

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2
```

有 InfiniBand 就 16-way TP：

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 16
```

本地图（原文版权仍归原站；学习对照用）：

![perf llama3](../../../../assets/vllm/blog/serving/llama31/01-perf_llama3.png)

**16×H100** 上的 serving 吞吐，合成平均 1024 / 128。

**没有 InfiniBand** 时，PP2+TP8 对 16-way TP 大约 **6.6×**。**有 InfiniBand** 则接近。

他们指向的文档：[distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)、[CPU offload 例子](https://docs.vllm.ai/en/latest/getting_started/examples/cpu_offload.html)。

## 致谢

Meta 预发布。Neural Magic（FP8）、CentML + Snowflake AI Research（PP）、Anyscale（chunked prefill）。评测跑在 Lambda 1-Click Clusters，带 InfiniBand。
