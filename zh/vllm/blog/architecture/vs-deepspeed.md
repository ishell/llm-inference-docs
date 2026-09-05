---
source: https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 笔记：vLLM 对 DeepSpeed-FastGen

英文对照：[en/vllm/blog/architecture/vs-deepspeed.md](../../../../en/vllm/blog/architecture/vs-deepspeed.md)  
原文：https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed  
2023-11-14。这是对 DeepSpeed 那篇「我们比 vLLM 快 2×」的公开回应，不是产品手册，更不是 2026 年的对打。

DeepSpeed 团队发了 [FastGen 博文](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)，说靠 **Dynamic SplitFuse** 吞吐能到 vLLM 的两倍。vLLM 这篇写得很克制：社区出新招，他们高兴；但 SplitFuse 真正赢的场合很窄，多数负载上 vLLM 更快，或者打平。


本地图（原文版权仍归原站；学习对照用）：

![s1](../../../../assets/vllm/blog/architecture/vs-deepspeed/01-s1.png)

![s2](../../../../assets/vllm/blog/architecture/vs-deepspeed/02-s2.png)

## TL;DR（原文原意）

- 常见负载上 vLLM 跟 FastGen 打平，**输出变长时 vLLM 更快**。
- FastGen 只在 **prompt 很长、输出很短** 时明显赢，靠的是 **Dynamic SplitFuse**。当时这东西在 vLLM 路线图上——后来日常名字是 **chunked prefill**。
- vLLM 的自我定位：最快、最好用的开源推理与 serving 引擎；**Apache 2.0**，社区的，模型和优化都要接得广。

## 他们看见的两处差别

1. **FastGen 的 KV 分配更保守 / 次优。** 输出一长，浪费就露出来：预留给一条序列的房间，PagedAttention 本可以还回给整桌。
2. **Dynamic SplitFuse 的加速，几乎只在 prompt 远长于输出时成立**（ISL ≫ OSL）。把 Prefill 切开，融进 Decode 的流里，免得一条长 prompt 把整桌卡住——这就是后来 `max_num_batched_tokens` / chunked prefill 要办的事。

所以：负载永远是长问短答，FastGen 好看；其余时候，vLLM 自称最多大约 **1.8×** 更快。

硬件： **NVIDIA A100-80GB**，**LLaMA-7B**。

### 场景 1：长 prompt、短输出

SplitFuse 理应在这里发光。他们测到的优势，没有宣传里的 2× 那么戏剧。

从图里读（`prompt_len=2600`）：

| output_len | vLLM (reqs/s) | DeepSpeed-FastGen (reqs/s) |
|---|---|---|
| 60 | 3.52 | 3.7 |
| 128 | 2.68 | 2.76 |
| 200 | 2.13 | 2.13 |

最短输出时 FastGen 略快；到 `output_len=200`，两条柱一样高。两边都随生成变长而变慢——那是 Decode 和 KV 在要房间，不是 SplitFuse 的魔术失灵。

### 场景 2：其余情形

这里 vLLM 最多大约 **1.8×**。图上的柱（`prompt_len=500`）：

| output_len | vLLM (reqs/s) | DeepSpeed-FastGen (reqs/s) | 大约倍率 |
|---|---|---|---|
| 150 | 10.03 | 7.42 | ~1.35× |
| 500 | 3.43 | 1.97 | ~1.74× |
| 1024 | 1.49 | 0.81 | ~1.84× |

输出越长，FastGen 那种保守的 KV 预留越吃亏，倍率就越靠近「最多 1.8×」那句。

当时公开的基准代码：[benchmarks/benchmark_throughput.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py)。问题与建议走 [vLLM GitHub](https://github.com/vllm-project/vllm)。数字是 **2023 年 11 月** 的快照，不要拿来羞辱 2026 年的任何引擎。

## 社区宣言（比分数更耐读的一段）

从 Berkeley **Sky Computing Lab** 出来，就要把最好的模型、优化、硬件接进来。他们点名正在做的：

- 系统性能
- 新功能：**LoRA**、投机解码、更好的量化
- 和硬件厂商： **AMD**、**AWS Inferentia**（原文拼成 Inferenetia）、**Intel Habana**

SplitFuse 他们说会认真集成。读完 [立项文](paged-attention.md) 再读这一篇，是为了看见 2023 年秋天社区在吵什么：不是「要不要分页」，而是 **Prefill 要不要切开**。切开之后，NVIDIA 调优手册和 vLLM [optimization.md](../../../optimization/optimization.md) 里的 `max_num_batched_tokens`，说的是同一件事。V1 默认的 chunked prefill，是这条路走完以后的日常——见 [V1 alpha](v1-alpha.md) 与 [Anatomy](anatomy.md)。

## 附录：功能对照（2023 年 11 月的时代胶囊）

当时 FastGen 只提供基本能力： **三种** 模型，没有 **stop string**，也没有并行采样（例如 beam search）。vLLM 写得很客气：他们预期 FastGen 会追上，也欢迎市场上的新招。

|  | vLLM | DeepSpeed-FastGen |
|---|---|---|
| Runtime | Python/PyTorch | Python/PyTorch |
| 模型实现 | HuggingFace Transformers | 自研实现 + HF 模型转换器 |
| Server 前端 | 演示用的简单 FastAPI | 自研 gRPC server |
| 调度 | Continuous batching | Dynamic SplitFuse |
| Attention kernel | PagedAttention & FlashAttention | PagedAttention & FlashAttention |
| 自定义 kernel（LLaMA） | Attention, RoPE, RMS, SILU | Attention, RoPE, RMS, SILU, Embedding |
| KV Cache 分配 | Near-optimal | Suboptimal / conservative |
| 支持的模型 | 16 种架构 | LLaMA, Mistral, OPT |
| 采样 | Random, parallel, beam search | Random |
| 停止条件 | Stop strings, stop tokens, EOS | EOS |

两边 attention 当时都已经写着 PagedAttention & FlashAttention。剩下的分歧是调度（continuous batching 对 SplitFuse），以及 KV 房间分得有多狠。不要把这张表当成 2026 年任何一方的功能清单。
