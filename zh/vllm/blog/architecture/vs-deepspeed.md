---
source: https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 笔记：vLLM 对 DeepSpeed-FastGen

英文对照：[en/vllm/blog/architecture/vs-deepspeed.md](../../../../en/vllm/blog/architecture/vs-deepspeed.md)  
原文：https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed  
2023-11-14。这是对 DeepSpeed 那篇「我们比 vLLM 快 2×」的公开回应，不是产品手册。


本地图（原文版权仍归原站；学习对照用）：

![s1](../../../../assets/vllm/blog/architecture/vs-deepspeed/01-s1.png)

![s2](../../../../assets/vllm/blog/architecture/vs-deepspeed/02-s2.png)

## TL;DR（原文原意）

- 常见负载上 vLLM 跟 FastGen 打平，**输出变长时 vLLM 更快**。
- FastGen 只在 **prompt 很长、输出很短** 时明显赢，靠的是 **Dynamic SplitFuse**。当时这东西在 vLLM 路线图上——后来它有了一个更常用的名字：**chunked prefill**。
- vLLM 的自我定位：最快、最好用的开源推理引擎；Apache 2.0，社区的。

## 他们看见的两处差别

1. FastGen 的显存分配更保守：输出一长，浪费就露出来。
2. SplitFuse 的加速，几乎只在 **ISL ≫ OSL** 时成立。

所以：负载永远是长问短答，FastGen 好看；其余时候，vLLM 自称最多大约 **1.8×** 更快。A100-80GB、LLaMA-7B。长问短答那组，他们测到的优势也没有宣传里的 2× 那么戏剧。

基准代码当时公开在 GitHub。数字是 2023 年 11 月的快照，不要拿来羞辱 2026 年的任何引擎。

## 社区宣言（比分数更耐读的一段）

从 Berkeley Sky Computing Lab 出来，就要把最好的模型、优化、硬件接进来。他们点名正在做的：系统性能、LoRA、投机解码、更好的量化；以及 AMD、AWS Inferentia、Intel Habana。SplitFuse 他们说会认真集成——读到这里的人已经知道，V1 默认的 chunked prefill 就是这条路走完以后的日常。

附录里一张功能表，也是时代胶囊：当时 FastGen 三种模型、没有 stop string、没有并行采样；vLLM 16 种架构，随机 / 并行 / beam search 都在。两边 attention 都写着 PagedAttention & FlashAttention。KV 分配：vLLM「接近最优」，FastGen「次优 / 保守」。

读完立项文再读这一篇，是为了看见 2023 年秋天社区在吵什么：不是「要不要分页」，而是 **prefill 要不要切开**。切开之后，NVIDIA 调优手册和 vLLM `optimization.md` 里的 `max_num_batched_tokens`，说的是同一件事。
