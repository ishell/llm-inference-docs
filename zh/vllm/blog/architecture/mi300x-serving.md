---
source: https://vllm.ai/blog/2024-10-23-vllm-serving-amd
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 在 AMD MI300X 上侍候 LLM：当时的最佳实践

英文对照：[en/vllm/blog/architecture/mi300x-serving.md](../../../../en/vllm/blog/architecture/mi300x-serving.md)  
原文：https://vllm.ai/blog/2024-10-23-vllm-serving-amd  
2024-10-23。客座：**Embedded LLM** 与 **Hot Aisle Inc.**。学习译文，不是官方译本。vLLM **0.6.2**（commit `cb3b2b9`）。旗标、镜像、数字都是那一天的；今日请以文档为准。后来的 ROCm attention 路由：[rocm-attention.md](rocm-attention.md)；硬件插件：[hardware-plugin.md](hardware-plugin.md)。CK vs Triton、hipBLASLt、TP vs PP 另见 [Leonard Lin](https://shisa.ai/blog/posts/tuning-vllm-mi300x/)。

**TL;DR。** vLLM 在 AMD MI300X 上相对 Text Generation Inference（TGI）：Llama 3.1 **405B** 吞吐约 **1.5×**、TTFT 约 **1.7×**；Llama 3.1 **70B** 吞吐约 **1.8×**、TTFT 约 **5.1×**。下文拧八个 vLLM 旋钮。只想看当时最优参数，跳到 [Quick Start Guide](#quick-start-guide)。

本地图（原文版权仍归原站；学习对照用）：

![405b1](../../../../assets/vllm/blog/architecture/mi300x-serving/01-405b1.png)
![405b2](../../../../assets/vllm/blog/architecture/mi300x-serving/02-405b2.png)

**图注（原文）。** vLLM vs TGI，Llama 3.1 405B，8× MI300X，BF16，32 QPS。

![70b1](../../../../assets/vllm/blog/architecture/mi300x-serving/03-70b1.png)
![70b2](../../../../assets/vllm/blog/architecture/mi300x-serving/04-70b2.png)

**图注（原文）。** 同上，Llama 3.1 70B。

## Introduction

Meta 当时宣布：线上 Llama 3.1 405B 的流量 **100%** 跑在 AMD MI300X 上——用来说明 ROCm 已经能扛 LLM 推理。这条消息碰上 **ROCm 6.2**：vLLM 在 AMD GPU 上更好接。

ROCm 是 AMD 对 CUDA 的那一套。有人还不熟，但它在长成能打的替代。配上 vLLM，把这股力用起来比以前容易。下文就是当时他们怎么用。

## vLLM v.s. TGI

数字再写一遍。MI300X 上 vLLM 相对 TGI：Llama 3.1 **405B** 吞吐 **1.5×**、TTFT **1.7×**；**70B** 吞吐 **1.8×**、TTFT **5.1×**。

405B 上，TTFT 和吞吐在多种 QPS 里都压着 TGI。优化配置、**16 QPS**：TTFT 平均大约 **3.8×** 快。吞吐：ShareGPT、优化配置、**1000 QPS** 时最高 **5.76** requests/second，TGI 是 **3.55** requests/second。

默认配置也赢。例如 16 QPS：vLLM 默认 **4.05** requests/second，TGI **2.58** requests/second。这个优势跨 QPS 档还在。

![Throughput](../../../../assets/vllm/blog/architecture/mi300x-serving/05-Throughput-Requests-per-Second-.png)
![Mean TTFT](../../../../assets/vllm/blog/architecture/mi300x-serving/06-Mean-TTFT-ms-.png)

**图注（原文）。** vLLM vs TGI，Llama 3.1 405B，8× MI300X，BF16，QPS 16、32、1000；命令见 Appendix。

## How to run vLLM with Optimal Performance

### Key Settings and Configurations

他们在 MI300X 上拧过许多 vLLM 旋钮。学到的是：

- **Chunked Prefill。** 口诀：MI300X 上多数情况先关，吞吐更好。
- **Multi-Step Scheduling。** GPU 利用率和整体表现能明显抬头。`--num-scheduler-steps` 取 **10 到 15**。
- **Prefix Caching。** 和 chunked prefill 叠用，在某些流量上会发光。用户请求的 prefix cache 命中低，不妨把 chunked prefill 和 prefix caching **一起关**。
- **Graph Capture。** 模型支持长上下文时，`--max-seq-len-to-capture` 设 **16384**。再加大不保证更快，有时会因 bucket 变粗而掉速。
- **AMD-Specific Optimizations。** 关掉 NUMA balancing、拧 `NCCL_MIN_NCHANNELS`，还能再挤一点。
- **KV Cache Data Type。** 求最优表现：用默认 KV dtype，它会自动跟上模型精度。
- **Tensor Parallelism。** 追吞吐：用能装下权重和上下文的**最小 TP**，再开多个 vLLM 实例。追延迟：TP 等于节点上的 GPU 数。
- **Maximum Number of Sequences。** `--max-num-seqs` 提到 **512** 或更高，按显存和算力来。短进短出时，利用率和吞吐会好看一截。
- **Use CK Flash Attention。** CK 实现比 Triton 那条快得多。

### Detailed Analysis and Experiments

#### Case 1: Chunked Prefill

Chunked prefill 当时还是实验功能：大 prefill 切成小块，和 decode 请求组在同一个 batch 里。compute-bound 的阅读，和 memory-bound 的说话，叠在一起。打开：LLM 构造器里 `--enable_chunked_prefill=True`，或命令行 `--enable-chunked-prefill`。

他们跑下来：把 chunked prefill 的值拧过，比完全关掉只**略好**一点。拿不准开不开，就先关——一般会好过默认。这句话只对 **MI300X**。

![case1 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/07-Requests-Per-Second.png)
![case1 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/08-Mean-TTFT-ms-.png)
![case1 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/09-Mean-TPOT-ms-.png)

#### Case 2: Number of scheduler steps

_Multi-step scheduling_ 进 vLLM **v0.6.0**，声称更高 GPU 利用率、更好整体表现。同日亲戚：[v0.6 吞吐文](https://vllm.ai/blog/2024/09/05/perf-update.html)（本库：[v0.6-throughput.md](../performance/v0.6-throughput.md)）。魔法是：调度和输入准备做一次，然后让模型连跑若干步、中间不打断 GPU。CPU 开销摊到这几步上，GPU 少空转。

打开：`--num-scheduler-steps` 大于 **1**（1 是默认）。他们发现再往上收益递减，所以上限停在 **15**。

![case2 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/10-Requests-per-Second.png)
![case2 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/11-Mean-TTFT-ms-.png)
![case2 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/12-Mean-TPOT-ms-.png)

#### Case 3: Chunked Prefill and Prefix caching

Chunked prefill 把大 prefill 切块好组 batch；prefix caching 把共享前缀已经算过的 KV 留下来。两件优化。

默认：模型上下文 **超过 32k token** 时，vLLM 会**自动打开** chunked prefill。prefill 切块的最大 token 数默认 **512**。

看图之前先定词。**Fresh Run**：prefix caching 的内存完全空着。**2nd Run**：Fresh Run 之后把同一套 benchmark 再跑一遍。ShareGPT 第二次跑，prefix caching 命中率大约 **50%**。

三句观察：

1. 第 2 根柱（红）对基线（蓝）：性能抬了一大截。
2. 第 3 根（黄）、第 5 根（橙）、第 6 根（青绿）对基线：chunked prefill 好不好，取决于用户 prompt 的长度分布。
3. 实验里第 3 根（黄）和第 4 根（绿）的 prefix caching 命中大约 **0.9%** 和 **50%**。这两根对基线和红柱：请求若没有高命中，**把 chunked prefill 和 prefix caching 一起关**，是当时的好口诀。

![case3 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/13-Requests-per-Second.png)
![case3 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/14-Mean-TTFT-ms-.png)
![case3 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/15-Mean-TPOT-ms-.png)

#### Case 4: Max sequence length to capture

`--max-seq-len-to-capture` 管 CUDA/HIP graph 愿意 capture、再 replay 的最长序列。超过这个长度，系统退回 eager：一步一步执行，通常更慢。普通模型和 encoder-decoder 都适用。

benchmark 里有个别扭的趋势：把 `--max-seq-len-to-capture` 加大，**不总是**更快，有时还会更慢。他们猜，和 vLLM 给不同长度做 bucket 的方式有关。

- **Bucketing。** 相近长度的序列归进同一只桶，每只桶各自优化 graph capture。
- **Optimal Buckets。** 一开始桶很细，例如 `[4, 8, 12, …, 2048, 4096]`，各种长度都能对上还算合适的图。
- **Coarser Buckets。** 加大 `--max-seq-len-to-capture`，桶会变粗，例如 `[4, 8, 12, 2048, 8192]`。
- **Performance Impact。** 真实输入掉进这些更疏的桶里，capture 下来的 CUDA/HIP graph 可能并不贴那次长度，表现就会掉。

所以：让 graph 去 capture 更长的序列，听起来划算，但要秤 bucket。最优的 `--max-seq-len-to-capture` 往往要按自己的负载试，在 capture 效率和桶的粒度之间找平衡。

![case4 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/16-Requests-per-Second.png)
![case4 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/17-Mean-TTFT-ms-.png)
![case4 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/18-Mean-TPOT-ms-.png)

#### Case 5: AMD Recommended Environmental Variables

再挤 MI300X，可以用 AMD 侧的环境变量。

- **Disabling NUMA Balancing。** Non-Uniform Memory Access（NUMA）的自动 balancing 有时会拖 GPU，甚至挂住。[AMD MAD 仓库](https://github.com/ROCm/MAD/blob/develop/benchmark/vllm/README.md) 建议关掉：

```bash
# disable automatic NUMA balancing
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
# check if NUMA balancing is disabled (returns 0 if disabled)
cat /proc/sys/kernel/numa_balancing
0
```

- **Tuning NCCL Communication。** NVIDIA Collective Communications Library（NCCL）管卡间通信。MI300X 上，[AMD vLLM fork 的性能文档](https://github.com/ROCm/vllm/blob/main/ROCm_performance.md) 建议 `NCCL_MIN_NCHANNELS=112`。

这两项一起开，他们测到的是**轻微**提升。这和 ["NanoFlow: Towards Optimal Large Language Model Serving Throughput"](https://arxiv.org/abs/2408.12757) 对得上：拧网络有好处，但 LLM 推理仍主要由 compute-bound 和 memory-bound 的活决定，通信那一刀幅度有限。

增益虽小，把环境变量拧干净，仍是把 AMD 这台机器再挤一毫米。

![case5 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/19-Requests-Per-Second.png)
![case5 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/20-Mean-TTFT-ms-.png)
![case5 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/21-Mean-TPOT-ms-.png)

#### Case 6: KVCache Type Auto/FP8

默认：vLLM 按模型精度自动分配 KV cache 类型。MI300X 上也支持原生 FP8 KV——KV 更瘦，可部署的上下文就能更长。

他们拿 Auto KV 和 FP8 KV 对默认基线。下图：Auto（红）的 requests per second 高于 FP8（黄）。理论上，这可能是 `Llama-3.1-70B-Instruct (bfloat16)` 上的量化税；税看起来不大，若能换来 KV 房间大幅下降，有些场景仍划算。

![case6 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/22-Requests-per-Second.png)
![case6 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/23-Mean-TTFT-ms-.png)
![case6 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/24-Mean-TPOT-ms-.png)

#### Case 7: Performance Difference between TP 4 and TP 8

Tensor parallelism 把单个张量切到多设备上，层内或算子内并行。模型在每张卡上的脚印变小，才能跨卡铺开。

把 TP 度加大，等于多给算力，但加速**不总是线性**：设备越多通信越重，每张卡上的活越少。MI300X 本身就很能打，每卡活太少反而吃不饱，扩展更难看。

所以追吞吐：他们建议**多开 vLLM 实例**，而不是把 TP 拧到头——吞吐更容易接近线性。若第一优先是压延迟，再加大 TP 可能更对。

![case7 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/25-Requests-per-Second.png)
![case7 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/26-Mean-TTFT-ms-.png)
![case7 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/27-Mean-TPOT-ms-.png)

#### Case 8: Effect of Maximum Number of (Parallel) Sequences

`--max-num-seqs`：每个 iteration 最多处理多少条序列。它管一个 batch 里能同时飞多少请求，也管显存和表现。ShareGPT 样本的进出都短，`Llama-3.1-70B-Instruct` 在 MI300X 上每个 iteration 能吞很多请求。实验里，即便 `--max-num-seqs` 已经是 **1024**，它**仍是**限制因素。

![case8 rps](../../../../assets/vllm/blog/architecture/mi300x-serving/28-Request-per-Second.png)
![case8 ttft](../../../../assets/vllm/blog/architecture/mi300x-serving/29-Mean-TTFT-ms-.png)
![case8 tpot](../../../../assets/vllm/blog/architecture/mi300x-serving/30-Mean-TPOT-ms-.png)

## Quick Start Guide

部署设置和用户请求分布都还不清楚时，可以：

- 用 CK Flash Attention（这篇没画对照；他们写 CK 比 Triton 快得多）
  - `export VLLM_USE_TRITON_FLASH_ATTN=0`
- 关掉 chunked prefill：`--enable-chunked-prefill=False`
- 关掉 prefix caching
- 模型支持长上下文：`--max-seq-len-to-capture` 设 **16384**
- `--num-scheduler-steps` 设 **10** 或 **15**
- AMD 环境：
  - `sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'`
  - `export NCCL_MIN_NCHANNELS=112`
- `--max-num-seqs` 提到 **512** 以上，看 GPU 显存和算力

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve meta-llama/Llama-3.1-70B-Instruct --host 0.0.0.0 --port 8000 -tp 4 --max-num-seqs 1024 --max-seq-len-to-capture 16384 --served-model-name meta-llama/Llama-3.1-70B-Instruct --enable-chunked-prefill=False --num-scheduler-steps 15 --max-num-seqs 1024
```

（原文把 `--max-num-seqs 1024` 写了两遍，照抄。）

快速起手：他们把 vLLM **0.6.2**（commit `_cb3b2b9ba4a95c413a879e30e2b8674187519a93_`）打成镜像推进 GitHub Container Registry。

```bash
# v0.6.2 post
docker pull ghcr.io/embeddedllm/vllm-rocm:cb3b2b9
# P.S. We also have compiled the image for v0.6.3.post1 at commit 717a5f8
docker pull ghcr.io/embeddedllm/vllm-rocm:v0.6.3.post1-717a5f8
```

起容器：

```bash
sudo docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /path/to/hfmodels:/app/model \ # if you have pre-downloaded the model weight, else ignore
   ghcr.io/embeddedllm/vllm-rocm:cb3b2b9 \
   bash
```

然后在容器里用他们找到的参数起服务——就是上面那条 `vllm serve`。

## Conclusion

这篇把 vLLM 在 AMD MI300X 上侍候大模型的力气拧过一遍。仔细调 chunked prefill、multi-step scheduling、CUDA graph capture，相对默认配置和别的 serving 方案，吞吐和响应都能抬一截。当时的结论：在 AMD 硬件上部署 LLM，vLLM 是合适的选择。

但要承认：探索主要对着**短进短出的 chatbot**。摘要、长文生成还要另查。Triton 和 CK attention kernel 的差异，也值得再挖。

他们还点名 Leonard Lin 那篇 [MI300X 调优文](https://shisa.ai/blog/posts/tuning-vllm-mi300x/)：hipBLAS vs hipBLASLt、CK Flash Attention vs Triton Flash Attention、Tensor Parallelism vs Pipeline Parallelism 等。

## Acknowledgements

正文由 [Embedded LLM](https://embeddedllm.com/) 起草。感谢 [Hot Aisle Inc.](https://hotaisle.xyz/) 赞助 MI300X 做 benchmark。

## Appendix

### Server Specification

Hot Aisle 那台：

- CPU：2 × Intel Xeon Platinum 8470
- GPU：8 × AMD Instinct MI300X

benchmark 用的模型和软件：

- Model：`meta-llama/Llama-3.1-405B-Instruct` 和 `meta-llama/Llama-3.1-70B-Instruct`
- vLLM（v0.6.2）：commit `cb3b2b9ba4a95c413a879e30e2b8674187519a93`
- Dataset：ShareGPT
- Benchmark script：仓库里的 `benchmarks/benchmark_serving.py`

ROCm 版 vLLM 镜像从仓库 `Dockerfile.rocm` 打（他们把跑 benchmark 的那版推进 GHCR：`docker pull ghcr.io/embeddedllm/vllm-rocm:cb3b2b9`）。

**所有 benchmark 都在该容器实例里跑，4 张 MI300X，CK Flash Attention，`VLLM_USE_TRITON_FLASH_ATTN=0`。**

### Detail Benchmark Configuration

| Configuration | Command |
| --- | --- |
| vLLM Default Configuration | `VLLM_RPC_TIMEOUT=30000 VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Llama-3.1-405B-Instruct -tp 8 --max-num-seqs 1024 --max-num-batched-tokens 1024` |
| TGI Default Configuration | `ROCM_USE_FLASH_ATTN_V2_TRITON=false TRUST_REMOTE_CODE=true text-generation-launcher --num-shard 8 --sharded true --max-concurrent-requests 1024 --model-id Llama-3.1-405B-Instruct` |
| vLLM (This Guide) | `VLLM_RPC_TIMEOUT=30000 VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Llama-3.1-405B-Instruct -tp 8 --max-seq-len-to-capture 16384 --enable-chunked-prefill=False --num-scheduler-steps 15 --max-num-seqs 1024` |
| TGI (This Guide) | `ROCM_USE_FLASH_ATTN_V2_TRITON=false TRUST_REMOTE_CODE=true text-generation-launcher --num-shard 8 --sharded true --max-concurrent-requests 1024 --max-total-tokens 131072 --max-input-tokens 131000 --model-id Llama-3.1-405B-Instruct` |
