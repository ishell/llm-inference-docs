---
source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# KV Offloading Connector：抢占时把记忆寄存在 CPU

英文对照：[en/vllm/blog/serving/kv-offload.md](../../../../en/vllm/blog/serving/kv-offload.md)  
原文：https://vllm.ai/blog/2026-01-08-kv-offloading-connector

2026-01-08。Or Ozeri、Danny Harnik（IBM Research 的 vLLM 团队）。学习译文，不是官方译本。功能随 **vLLM 0.11.0** 进来；**0.12.0** 因物理块变大，性能跳了一档。和 [Mooncake](mooncake.md) 同一扇 **KVConnector** 门：一个寄存在本机 DRAM，一个放到集群池。这篇把笔墨放在 **CPU DRAM**，以及 host↔device 怎么搬。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/kv-offload/01-figure1.png)

**图注（原文 Figure 1）。** 单条请求的 TTFT（Llama-3.1-8B-Instruct，NVIDIA H100）。

![figure2](../../../../assets/vllm/blog/serving/kv-offload/02-figure2.png)

**图注（原文 Figure 2）。** 并发吞吐（Llama-3.1-8B-Instruct，NVIDIA H100，10000 条 512-token Prefill）。

![figure3](../../../../assets/vllm/blog/serving/kv-offload/03-figure3.png)

**图注（原文 Figure 3）。** 单次 GPU → CPU 传输吞吐（NVIDIA H100，一次搬 1000 块）。

![figure4](../../../../assets/vllm/blog/serving/kv-offload/04-figure4.png)

**图注（原文 Figure 4）。** 单次 CPU → GPU 传输吞吐（NVIDIA H100，一次搬 1000 块）。

![figure5](../../../../assets/vllm/blog/serving/kv-offload/05-figure5.png)

**图注（原文 Figure 5）。** 单条请求 TTFT（Llama-3.2-1B-Instruct，NVIDIA H100）。

![figure6](../../../../assets/vllm/blog/serving/kv-offload/06-figure6.png)

**图注（原文 Figure 6）。** 并发吞吐（Llama-3.2-1B-Instruct，NVIDIA H100，10000 条 512-token Prefill）。

## Motivation

侍候 LLM 计算很重，核心是算出一坨叫 KV 的数据。用户 prompt 进来，第一步是按这段 prompt 算 KV——请求生命周期里的 Prefill。Prefill 贵，要加速器（GPU）才快得起来。

一份 prompt 算出来的 KV，可以被共享同一前缀的其他 prompt 复用，不必重算。缓存并复用 KV，常常换来两件事：

- **压低请求延迟**（读 cache 若快过重算 KV）
- **抬高单机吞吐**（GPU 核空出来，才能同时伺候更多请求）

即便请求之间**没有**共享前缀，KV offload 仍然有用。并发一高，GPU 装不下正在跑的那批 KV，引擎会**抢占**一条正在跑的请求，把它的 KV 从 GPU 丢掉。回头再调度这条请求，KV 就要重算。抢占前先把 KV 卸到更大的一层（例如 CPU DRAM），重算那笔可以不付。

### CPU Offloading

这篇把重点放在卸到 CPU 内存（DRAM）。几条叠在一起，才值得单独讲：

- CPU RAM 几乎处处都有。
- 容量通常大于 GPU 显存，KV cache 才能更大。
- CPU RAM 和 GPU 之间延迟低、吞吐高。叠上容量，CPU offload **最适合伺候抢占**。
- CPU RAM 也是再往外置存储卸的**方便中转**。存储延迟高时尤其如此。

## The New Offloading Connector

### The vLLM Connector API

vLLM 早就有一套读、写 KV 的 API，嵌在请求生命周期里，叫 Connector API。引擎处理任何请求之前会问它：能不能从外部进口 KV。算完新的 KV 之后，再叫它存到外部目标。

早年 Connector API 是**同步**的：对外搬 KV 的时候，引擎卡住，下一批请求进不来。vLLM **0.9.0** 把这条 API 扩成 **异步** 读写。Offloading connector 走的就是这条异步路。

他们引入 **offloading connector**：异步卸、异步装 KV。对外再开一层可插拔 backend API，任何介质都能当卸货处。加新 backend 时，你主要是写一个在介质之间拷 KV 的 transfer function。

自带 CPU backend，vLLM 里就能原生把 KV 卸到 CPU。后文只谈 CPU offload。

### Using the Offloading Connector

CPU offload，给 `vllm serve` 加：

```
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```

这条 CLI 假定 [PR #24498](https://github.com/vllm-project/vllm/pull/24498)，文中指望进 **0.14.0**。

更早的发行版用：

```
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

`num_cpu_blocks` 是给 CPU KV cache 分配的块数。

## Benefits of CPU Offloading via the Offloading Connector

两套微基准。第一套量**单条** Prefill 的 TTFT，看从 CPU 加载相对 GPU 重算快多少。第二套量**并发**吞吐，看 offload 怎么扛更重的活。

第一套：单条 Prefill，CPU cache 加载对 GPU 重算 KV。

Figure 1：从 CPU 加载 KV，相对 GPU 重算，TTFT 降 **2–22×**，随 prompt 变长。完整环境和代码在文末。

卸 KV（GPU → CPU）本身**不挡用户看见的等待**：offload 也是异步的，用户请求不必等这次传输结束。所以 **cache miss 时，这条 connector 几乎不伤 TTFT**。

第二套：并发。提交 **10000** 条互不共享的请求（各 **512** token），按 CPU cache 命中率看吞吐。计时**不计** CPU cache 预热，再折成 token/s。GPU cache 关掉，好把笔墨留在 CPU 命中上。

Figure 2：吞吐随 CPU KV 命中率上升。最多大约 **9×**——尽管这种长度的 TTFT 只降了大约 **2×**。大头是**吞吐最大化**，不是单条延迟。

### vLLM versions of the Offloading Connector

**0.12.0** 性能跳了一大档。例如 Llama-3.1-8B-Instruct + NVIDIA H100：TTFT 最多约 **4×**，吞吐约 **5×**。原因写在后面「物理块大小」那一节。

文中还指望 **0.14.0** 再带两笔（这篇评测已经算进去）：

- 被抢占的请求能从 CPU 请回（[PR #29870](https://github.com/vllm-project/vllm/pull/29870)）
- 修 offload 与模型计算的竞态（[PR #31341](https://github.com/vllm-project/vllm/pull/31341)）

## Evaluating GPU-CPU Transfer Techniques

后半篇是设计时的技术深潜：尽量抬 GPU–CPU 吞吐，同时少占 GPU / CPU 核，好让推理吞吐上去。

给 offloading connector 写 backend，核心是一个 **transfer function**。CPU backend 的这份函数在 GPU 内存和 CPU 内存之间拷（双向）。当时支持 **CUDA 兼容**设备（NVIDIA 与 AMD）。

实现走 `cudaMemcpyAsync`，用的是 GPU 上的 **DMA**（Direct Memory Access）。DMA 就是为设备与主机之间的高吞吐搬运设计的，而且几乎不占 CPU / GPU 核。异步和模型计算重叠时，这一点尤其要紧。

DMA 最喜欢**大块、物理连续**的拷贝。所以 offload 成绩会随 KV 布局变：KV 块更大的模型更占便宜。

DMA 到底有多快？自定义 CUDA kernel 会不会更快？

他们做了微基准 [gpu_cpu_benchmark](https://github.com/orozery/playground/tree/kv-offloading-blog-dec-2025/kvcache/gpu_cpu_benchmark)，比两种拷法：

- **DMA** —— `cudaMemcpyAsync`
- **自定义 CUDA kernel** —— GPU 核用裸指针拷 16 字节。并行度高，但更抢 GPU 核上的主业。

第一项：单次传输 **1000** 块，块大小从 **4 KB 到 16 MB**：

Figure 3 / Figure 4：DMA **只在块大时**好看。块小时，自定义 kernel 吞吐明显更高。不过 kernel 更吵，方差更大。

再测双向：同时发一次读、一次写。块固定 **2 MB**，改两个方向的大小比。两种机制都在两边搬得差不多时到峰值。单向两边都能到大约 **50 GB/s**；双向就不一样了：

- DMA：**83.4 GB/s**
- 自定义 kernel：**68.5 GB/s**

于是还要问两句：

- **vLLM 用的有效块有多大？** 取决于模型和配置。下一节给当时常见模型的答案。
- **两种拷法会怎样干扰 GPU 上的模型计算？** Offloading connector 本来就要和 forward 并行。后面会看每种拷法对整体吞吐的影响。

## Changing vLLM’s Memory Layout

这一节讲他们怎么改 GPU 上的 KV 布局：让传输更好过，同时不牺牲计算速度。

先看默认布局，以及 offload 时实际要拷的碎片有多大——这就是 vLLM 里 KV 传输的有效物理块。

vLLM 按 token 块分配 GPU 内存，默认每块 **16** token。物理布局取决于 attention 后端（FlashAttention、FlashInfer 等）和模型。今天最常见的是均匀模型：多层，每层一份同形状的 KV。vLLM 也支持 hybrid 模型，当时**还没为这条 connector 优化**。均匀模型里，每层各有一份 KV，所以一个逻辑块的 KV 会被撕成 `num_layers` 块。再往下，有的 attention 后端还会按 K / V 再撕成 2 个子块。

这点碎片对计算无所谓，对 KV offload 是灾难：有效块变小。他们为此 [upstream](https://github.com/vllm-project/vllm/pull/27743) 了一份新布局：所有层的 KV 收进**一块连续物理块**。有效物理块大约变成 **`2 × num_layers`** 倍，offloading connector 的吞吐上了一个数量级。

下表是当时常见模型，旧（0.11.0）对新（0.12.0）物理块（假定 vLLM 用 16-token 块）：

| Model | Old block size | New block size |
| :---- | :---- | :---- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B（`tensor_parallel_size=2`） | 16 KB | 2 MB |
| deepseek-ai/DeepSeek-V2-Lite-Chat（GPU block size=64） | 72 KB | 1.9 MB |
| meta-llama/Llama-3.1-8B-Instruct | 32 KB | 2 MB |
| meta-llama/Llama-3.2-1B-Instruct | 16 KB | 0.5 MB |
| meta-llama/Llama-3.1-70B-Instruct | 8 KB | 1.25 MB |
| mistralai/Mistral-7B-Instruct-v0.2 | 32 KB | 2 MB |
| mistralai/Mistral-Small-24B-Instruct-2501 | 32 KB | 2.5 MB |
| Qwen/Qwen2.5-3B-Instruct | 8 KB | 0.56 MB |
| Qwen/Qwen3-0.6B | 32 KB | 1.75 MB |
| Qwen/Qwen2.5-7B-Instruct | 16 KB | 0.87 MB |
| Qwen/Qwen3-4B-Instruct-2507 | 32 KB | 2.25 MB |
| Qwen/Qwen2.5-1.5B-Instruct | 8 KB | 0.44 MB |
| Qwen/Qwen3-8B | 28 KB | 1.97 MB |
| Qwen/Qwen3-1.7B | 32 KB | 1.75 MB |
| Qwen/Qwen3-32B（`tensor_parallel_size=2`） | 16 KB | 2 MB |

新布局常见物理块大约 **0.5–2 MB**，旧的只有几 KB。叠上 GPU–CPU 微基准：DMA 应能打平或只略逊于自定义 kernel，取决于模型。

## End-to-end Evaluation of Copy Methods

用前面两套 vLLM 微基准，对比 offloading connector 的两个变体：

- 上游版本：DMA transfer function
- 打过补丁的版本：微基准里那只自定义 kernel

他们故意挑 **DMA 最亏的场景**：物理块只有 **0.5 MB** 的模型。

Figure 5：单条 TTFT。自定义 kernel **略快**——1K prompt 差不到 **1 ms**，90K prompt 最多大约 **15 ms**。这和 0.5 MB 块上的微基准对得上。块更大的模型，两边差不多。

Figure 6：并发一万条 512 Prefill。**DMA 吞吐更好。** 命中 0% 时大约领先 **5.5%**，命中 80% 时大约 **15%**。

解释：自定义 kernel 和模型计算抢 GPU 核。命中 **0%** 时，自定义 kernel 甚至比**完全不开 CPU offload** 还慢约 **6%**。命中 **100%** 时没有并行计算，差距缩小。

他们强调：这是 DMA 最亏的模型。更常见的模型物理块更大，DMA 更占便宜。以 **Llama-3.1-8B-Instruct** 为例，DMA 吞吐最多大约领先自定义 kernel **32%**，TTFT 打平。

收束：改 GPU 布局，是为了让 DMA 配得上 KV 传输，整体吞吐才更好。

## Evaluation Setup and Benchmark Code

评测环境：

- 单容器 Ubuntu 24.04.1 LTS
- Kernel 5.14.0-427.81.1.el9_4.x86_64
- Intel Xeon Sapphire Rapids 2.1 GHz（限制 **8** 核）
- NVIDIA H100 80GB HBM3
- 500 GB DRAM
- CUDA 12.9
- vLLM commit `2a1776b7ac4fae7c50c694edeafc1b14270e4350`
- Flash Attention backend
- GPU prefix caching 关掉（只评 CPU 命中）
- GPU block size 16 token
- CPU block size 16 token
- De/Tokenization 关掉

脚本：[kv_offload_benchmark.py](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py)。

### What's Next?

下一步里程碑：让 CPU KV cache 当外置存储 offload 的中间层。

正确性和性能仍是第一优先。欢迎试用、报数字、报问题。

讨论：vLLM Slack 的 `#feat-v1-cpu-offloading`（[vLLM Slack](https://vllm-dev.slack.com/archives/C09AYJFFLKD)）。
