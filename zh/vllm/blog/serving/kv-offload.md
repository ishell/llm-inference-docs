---
source: https://vllm.ai/blog/2026-01-08-kv-offloading-connector
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# KV Offloading Connector：抢占时把记忆寄存在 CPU

英文对照：[en/vllm/blog/serving/kv-offload.md](../../../../en/vllm/blog/serving/kv-offload.md)  
原文：https://vllm.ai/blog/2026-01-08-kv-offloading-connector  
2026-01-08。随 **vLLM 0.11.0** 进来；**0.12.0** 因物理块变大，性能跳了一档。和 [Mooncake](mooncake.md) 同一扇 **KVConnector** 门：一个寄存在本机 DRAM，一个放到集群池。这篇把笔墨放在 **CPU DRAM**，以及 host↔device 怎么搬。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/kv-offload/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/kv-offload/02-figure2.png)

![figure3](../../../../assets/vllm/blog/serving/kv-offload/03-figure3.png)

![figure4](../../../../assets/vllm/blog/serving/kv-offload/04-figure4.png)

![figure5](../../../../assets/vllm/blog/serving/kv-offload/05-figure5.png)

![figure6](../../../../assets/vllm/blog/serving/kv-offload/06-figure6.png)

## 动机

Prefill 按 prompt 算 KV——贵，要加速器。共享前缀就能复用：读 cache 若快过重算，**延迟降**；GPU 核空出来，**单机吞吐升**。

没有共享前缀时，offload 仍有用。并发把 GPU KV 挤满，引擎会**抢占**、丢掉 KV，回头再 **RECOMPUTE**。抢占前先卸到更大的一层（CPU DRAM），重算那笔可以不付。

## 为什么是 CPU

- RAM 几乎处处都有
- 容量通常 **大于 HBM**，KV 池子才能更大
- CPU↔GPU 相对外置存储更近——适合伺候**抢占**
- 再往盘上卸时，它是方便的**中转**（存储延迟高时尤其）

## Connector API

vLLM 早就在处理请求前问 Connector（能不能进口 KV），算完再叫它存。早年是**同步**的：搬的时候引擎卡住，下一批进不来。**0.9.0** 起 **异步** 读写。Offloading connector 走这条路。

Backend **可插拔**：实现一个在介质之间拷 KV 的 transfer function。自带 **CPU backend**。后文只谈 CPU。当时 transfer 支持 **CUDA 兼容**设备（NVIDIA 与 AMD）。

## CLI

较新的旗标（文中假定 [PR #24498](https://github.com/vllm-project/vllm/pull/24498)，指望进 **0.14.0**）：

```bash
--kv_offloading_backend native --kv_offloading_size <size_in_GB>
```

更早的写法：

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks": <num_cpu_blocks>}}'
```

`num_cpu_blocks` 是 CPU KV 的块数。

## 成绩（Llama-3.1-8B-Instruct，NVIDIA H100）

**单条 Prefill 的 TTFT**（Figure 1）：从 CPU 加载 KV，相对 GPU 重算，大约 **2–22×**（随 prompt 变长）。GPU→CPU 的卸本身异步，不挡用户可见路径——**cache miss 几乎不伤 TTFT**。

**吞吐：一万条互不共享的 512-token Prefill**，关掉 GPU prefix cache（Figure 2）：不计 CPU cache 预热；token/s 对 CPU 命中率。最多大约 **9×**——尽管这种长度的 TTFT 只降了大约 **2×**。大头是**吞吐**，不是单条延迟。

### 版本

**0.12.0** 跳了一大档（下面的物理块）。同模型同卡：TTFT 最多约 **4×**，吞吐约 **5×**。

指望进 **0.14.0**（这篇评测已经算进去）：

- 被抢占的请求能从 CPU 请回 — [PR #29870](https://github.com/vllm-project/vllm/pull/29870)
- offload 与计算的竞态 — [PR #31341](https://github.com/vllm-project/vllm/pull/31341)

## GPU↔CPU：DMA 对自定义 kernel

CPU backend 用 `cudaMemcpyAsync`——**DMA**，少占 SM 和 CPU，才能和 forward 重叠。DMA 喜欢**大块连续**拷贝，布局（块大小）决定命运。

他们还做了微基准：自定义 CUDA kernel，用裸指针拷 16 字节（并行度高，但**跟计算抢 SM**）。代码：[gpu_cpu_benchmark](https://github.com/orozery/playground/tree/kv-offloading-blog-dec-2025/kvcache/gpu_cpu_benchmark)。

**一次搬 1000 块**，块从 **4 KB 到 16 MB**，H100：

- Figure 3：GPU→CPU。块**大**时 DMA 赢；块**小**时自定义 kernel 赢。Kernel 更吵（方差大）。
- Figure 4：CPU→GPU。同一形状。

**双向**，块固定 **2 MB**，改读写比。两边搬得差不多时峰值最高。单向两边大约 **50 GB/s**；双向：

- DMA **83.4 GB/s**
- 自定义 kernel **68.5 GB/s**

接下来要问两件事：vLLM 的**有效物理块**有多大；两种拷法会怎样干扰 forward。

## 改布局

默认逻辑块 **16 token**。物理布局看 attention 后端（FlashAttention、FlashInfer……）和模型。常见**均匀**模型：每层一份同形状 KV。**混合模型当时还没为这条 connector 优化。**

按层切完，还可能再按 K/V 切开。对计算无所谓，对 DMA 是灾难——有效拷贝只剩几 **KB**。他们 [upstream](https://github.com/vllm-project/vllm/pull/27743) 了**跨层一块连续物理块**。有效大小大约变成 **`2 × num_layers`** 倍。Offload 吞吐上了一个数量级。新布局常见 **0.5–2 MB**，旧的只有几 KB。

| 模型 | 旧块 | 新块 |
| --- | --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B（`tensor_parallel_size=2`） | 16 KB | 2 MB |
| deepseek-ai/DeepSeek-V2-Lite-Chat（GPU block size=64） | 72 KB | 1.9 MB |
| meta-llama/Llama-3.1-8B-Instruct | 32 KB | **2 MB** |
| meta-llama/Llama-3.2-1B-Instruct | 16 KB | **0.5 MB** |
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

（除非另注，都是 16-token 块。）叠上微基准：DMA 应能打平或只略逊于 kernel，取决于模型。

## 端到端：故意挑 DMA 最亏的 0.5 MB

Llama-3.2-1B-Instruct，H100。

Figure 5，单条 TTFT：自定义 kernel **略快**——1K prompt 差不到 **1 ms**，90K 大约 **15 ms**。块更大的模型，两边差不多。

Figure 6，并发一万条 512 Prefill：**DMA 吞吐更好**。命中 0% 大约 **5.5%**，80% 大约 **15%**。命中 **0%** 时，自定义 kernel 甚至比**完全不开 offload** 还慢约 **6%**（跟计算抢 SM）。命中 **100%** 时没有并行计算，差距缩小。

**Llama-3.1-8B-Instruct：** DMA 吞吐最多大约领先 kernel **32%**，TTFT 打平。改布局不是为了好看，是为了让 DMA 配得上这块活。

## 评测环境（原文）

Ubuntu 24.04.1 容器；内核 `5.14.0-427.81.1.el9_4.x86_64`；Intel Xeon Sapphire Rapids 2.1 GHz（限制 **8 核**）；NVIDIA **H100 80GB HBM3**；**500 GB** DRAM；CUDA **12.9**；vLLM `2a1776b7ac4fae7c50c694edeafc1b14270e4350`；Flash Attention；关掉 GPU prefix cache；GPU/CPU 块 **16**；关掉 de/tokenization。脚本：[kv_offload_benchmark.py](https://github.com/orozery/playground/blob/kv-offloading-blog-dec-2025/kvcache/kv_offload_benchmark.py)。

下一步（文中当时）：CPU 当外置存储的中间层。Slack：`#feat-v1-cpu-offloading`。
