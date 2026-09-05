---
source: https://vllm.ai/blog/2024-07-23-llama31
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Llama 3.1：128K 自动 chunked prefill；405B 单机走 FP8，多机优先 PP

英文对照：[en/vllm/blog/serving/llama31.md](../../../../en/vllm/blog/serving/llama31.md)  
原文：https://vllm.ai/blog/2024-07-23-llama31  
2024-07-23。署名 **vLLM Team**。数字是 **早期** 参考，文内自己说几周内会再涨。后继 herd：[llama4.md](llama4.md)。Llama Stack 邻居：[llama-stack.md](llama-stack.md)。分布式：[distributed-inference.md](distributed-inference.md)。1024/128 那条负载是他们的盘子，不是你的 SLA。

**原文 TL;DR：**

- 满 **128K** 窗口会打开 chunked prefill：控显存；长 prompt 少打断正在 decode 的请求。
- 405B-Instruct-FP8：`--tensor-parallel-size 8` 上 8×H100 / A100。他们 1024/128 负载：**2.82 req/s**，输入 **2884.86 tok/s**，输出 **291.53 tok/s**。
- GSM8K 8-shot CoT：FP8 **95.38%**（±0.56）vs BF16 官方 **96.8%**。
- 无量化：`--pipeline-parallel-size 2 --tensor-parallel-size 8` 跑 16 卡；没 IB 时 PP+TP 比 16-way TP 约 **6.6×**，有 IB 则接近。

## Introduction

vLLM 和 Meta 一起接 Llama 3.1：上下文更长（128K），模更大（405B），能力更强。点名的增强：chunked prefill、FP8 量化、pipeline parallelism。

当时：40+ 种 LLM，Nvidia / AMD / Inferentia / TPU / Intel / Gaudi。文档：[docs.vllm.ai](https://docs.vllm.ai/)。

128K：vLLM 自动打开 [chunked prefill](https://www.linkedin.com/posts/joinanyscale_recently-weve-contributed-chunked-prefill-activity-7201277641490849792-lGqZ)。显存有界；长 prompt 少打断正在飞的 decode。

当时安装：`pip install -U vllm` 或 `vllm/vllm-openai`。

405B 页上的几条路：

- **FP8：** 官方 FP8，8×A100 或 8×H100
- **Pipeline Parallelism：** BF16 跨机，层放在不同节点
- **Tensor Parallelism：** 跨机、跨卡切
- **AMD MI300x 或 NVIDIA H200：** 一台 8×MI300x（192 GB）或 8×H200（141 GB）
- **CPU Offloading：** 最后一招，满精度、GPU 内存不够时把权重卸到 CPU

推荐：单机走 FP8；多机走 pipeline parallelism。性能仍是初步；当时还在探更多量化和 PP 吞吐。

## FP8

8-bit 浮点。H100 / MI300x：原生 tensor core。当时：KV cache、attention、MLP 都能走 FP8。脚印更小、吞吐更高、延迟更低，声称精度几乎不掉。

官方 Meta Llama 3.1 405B FP8 经 FBGEMM：MLP 的 up/gate/down 做 per-channel 量化，静态 scale。第一层和最后一层跳过；再加静态上界。命令：

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tensor-parallel-size 8
```

负载：平均输入 **1024**、平均输出 **128** → **2.82** requests/s；输入 **2884.86 tok/s**，输出 **291.53 tok/s**。

精度核对（lm-eval-harness，GSM8K 8-shot CoT，exact match）：**95.38%**（±0.56 stddev）vs BF16 官方 **96.8%**。

## Pipeline parallelism

无量化 405B：16×H100 或 16×A100。PP 按层切开、跨机；点对点通信，不必付昂贵的 all-reduce。节点之间没有 InfiniBand 时尤其有用。

PP 和 TP 可以叠。16 卡 / 2 机：2-way PP + 8-way TP——每机一半模型，机内 NVLink all-reduce：

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2
```

有 InfiniBand，页上也写了 16-way TP：

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 16
```

本地图（原文版权仍归原站；学习对照用）：

![perf llama3](../../../../assets/vllm/blog/serving/llama31/01-perf_llama3.png)

**Figure。** 16×H100 上的 serving 吞吐，合成数据（平均输入 1024，平均输出 128）。没 IB：PP+TP 相对 16-way TP 约 **6.6×**。有 IB：接近。

分布式文档：[distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)。CPU offload 例子：[cpu_offload](https://docs.vllm.ai/en/latest/getting_started/examples/cpu_offload.html)。

## Acknowledgements

Meta（预发布合作）。Neural Magic（FP8）。CentML 和 Snowflake AI Research（pipeline parallelism）。Anyscale（chunked prefill）。评测跑在带 InfiniBand 的 [Lambda 1-Click Clusters](https://lambdalabs.com/service/gpu-cloud/1-click-clusters) 上。
