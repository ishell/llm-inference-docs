---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 第 3 章：调 Max Batch Size 和 Max Num Tokens

TensorRT-LLM 真正值钱的部分，往往不是某一颗更亮的 kernel，而是那个 **inflight batching** 调度器：context（prefill）和 generation 可以挤在同一次 iteration 里。谁能挤进来，由两个编译期上限决定——`max_batch_size` 和 `max_num_tokens`。拧它们，吞吐会变一张脸。

数字仍是演示。环境、SKU、互联、负载会改写结局。


本地图（原文版权仍归原站；学习对照用）：

![TRTLLM Scheduler Vis 1](../../../assets/nvidia/performance-tuning/trtllm-max-batch/01-TRTLLM_Scheduler_Vis_1.svg)

![TRTLLM Scheduler Vis 2](../../../assets/nvidia/performance-tuning/trtllm-max-batch/02-TRTLLM_Scheduler_Vis_2.svg)

![TRTLLM Scheduler Vis 3](../../../assets/nvidia/performance-tuning/trtllm-max-batch/03-TRTLLM_Scheduler_Vis_3.svg)

![TRTLLM Scheduler Vis 4](../../../assets/nvidia/performance-tuning/trtllm-max-batch/04-TRTLLM_Scheduler_Vis_4.svg)

![TRTLLM Scheduler Vis Chunked Context 1](../../../assets/nvidia/performance-tuning/trtllm-max-batch/05-TRTLLM_Scheduler_Vis_Chunked_Context_1.svg)

## 调度器在干什么

官方用一组玩具数字把调度器画出来：`max_batch_size = 4`，`max_num_tokens = 12`。每个方块是一个 token，颜色是请求。；这里用文字把同一出戏走一遍。不同请求画在不同行上，只是为了好看，**不是真实显存布局**。

引擎刚醒，门外排着几条还没被调度的请求。

调度器先收下 Request 1 和 Request 2，去做它们的 context phase。两条 prompt 各 5 个 token，合计 10。token 预算还剩 2。剩下的请求 prompt 都长过 2，谁也挤不进来——除非开了 **context chunking**（见本章末尾，以及 paged context attention）。这些 prompt token 在图上标着 **C**。

一次 iteration 跑完：两条请求的 KV 建好了，各自吐出第一个生成 token，标成 **G1**。

调度器**优先排 generation**。两条 G1 只占 2 个 token，12 的预算几乎空着。于是 Request 3、Request 4 的 prefill 可以进来。Request 5 进不来：token 预算还够，但 **batch 已经到 4**。

再一次 iteration。Request 1 的 G2 恰好是 stop token。调度器在下一轮执行前把它踢掉，准备交还给用户。batch 空出一个位子，Request 5 才能进。与此同时 Request 2 的 G1 已经写进它的 KV——KV 会随着生成一起长。

两件事同时发生：旧请求在 decode 里慢慢长大，新请求在抢剩下的 token 预算。上限设歪了，不是「慢一点」，而是一整类请求永远排不上，或者 prefill 把 KV 的房子挤塌。

调度器还会看空闲 KV 显存，以及下一章的运行时策略。这只是为了让你看见 `max_batch_size` 和 `max_num_tokens` 怎么卡脖子。

## 调 Max Batch Size

太小：新请求进不来，吞吐被自己掐死。默认 **2048**。建议扫 2 的幂。

```python
build_config = BuildConfig(max_batch_size=512)
```

CLI：`trtllm-build --max_batch_size <N>`

接上一章已经开了 multiple profiles、GEMM plugin、paged context attention、reduce fusion 的引擎，扫 64 / 512 / 默认 2048：

| 指标 | Max Batch Size 64 | Max Batch Size 512 | Max Batch Size 2048 |
|---|---|---|---|
| Token Throughput (tokens/sec) | 1944.3031 | 2466.7933 | 2044.2628 |
| Request Throughput (req/sec) | 0.9494 | 1.2045 | 0.9982 |
| Average TTFT (ms) | 145.7607 | 147.7876 | 146.6628 |
| Average ITL (ms) | 14.6475 | 14.6554 | 14.4493 |

64 明显堵。**512 是甜区**：相对默认 2048，吞吐大约 **+20%**，延迟几乎不动。默认值不是神谕，只是一个够大的屋顶。

## 调 Max Num Tokens

太小：调度被卡，长 prompt 进不去。太大：长上下文的 prompt token 把显存吃光，KV 没地方住，吞吐掉下去，甚至 OOM。默认 **8192**。建议扫 ≥1024 的 2 的幂。能网格搜索就和 `max_batch_size` 一起扫——它们一起决定「这一拍能装下谁」。

```python
build_config = BuildConfig(max_batch_size=512, max_num_tokens=2048)
```

CLI：`trtllm-build --max_num_tokens <N>`

batch 钉在 512 时：

| 指标 | Max Num Tokens 2048 | Max Num Tokens 8192 | Max Num Tokens 16384 |
|---|---|---|---|
| Token Throughput (tokens/sec) | 2474.2581 | 2466.7933 | 2461.0165 |
| Request Throughput (req/sec) | 1.2081 | 1.2045 | 1.2017 |
| Average TTFT (ms) | 147.5742 | 147.7876 | 147.9623 |
| Average ITL (ms) | 14.6852 | 14.6554 | 14.6769 |

这个负载上 2048 略好，差距不大。有的负载差距会很大。不要默认「8192 就行」——去看你的秒表。

## 为什么总建议开 paged context attention

现在调度器的形状已经清楚了。Paged context attention 打开的是 **context chunking**：一条请求的 prefill 可以拆到好几次 iteration。玩具例子里 Request 3 因为超了 token 预算进不去；有了 chunking，它的**第一块**就可以先进来。

两件好事：

1. 长 prompt 不会被已经在飞的请求永远挡住。生产里最差 TTFT 会好看很多。
2. `max_num_tokens` **不必 ≥ 最长 prompt**。长上下文场景尤其重要：把 `max_num_tokens` 设成天文数字，等于从 KV 手里抢房子。

最差情况，chunked context 对性能几乎无伤；很多场景它是增益。NVIDIA 的原话接近：**永远开。**

## 合在一起

相对上一章开齐编译旗标之后：

| 指标 | Build-Time Flags ON | Tuned Max Batch / Max Num Tokens | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 2044.2628 | 2474.2581 | 21.03 |
| Request Throughput (req/sec) | 0.9982 | 1.2081 | 21.03 |
| Average TTFT (ms) | 146.6628 | 147.5742 | -0.62 |
| Average ITL (ms) | 14.4493 | 14.6852 | -1.63 |

吞吐大约 **+21%**。延迟那点回落在 run-to-run 噪声里。

相对完全没调的 baseline：

| 指标 | Baseline | Flags + Tuned Batch/Tokens | % Improvement |
|---|---|---|---|
| Token Throughput (tokens/sec) | 1564.3040 | 2474.2581 | 58.17 |
| Request Throughput (req/sec) | 0.7638 | 1.2081 | 58.17 |
| Average TTFT (ms) | 147.6976 | 147.5742 | 0.08 |
| Average ITL (ms) | 31.3276 | 14.6852 | 53.12 |

吞吐大约 **+58%**，ITL 大约 **−53%**，TTFT 几乎没动。房子还是那栋房子，门厅被重新安排过了。
