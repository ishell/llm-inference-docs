---
source: https://vllm.ai/blog/2025-02-17-distributed-inference
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 分布式推理：一张卡装不下以后

英文对照：`en/vllm/blog/serving/distributed-inference.md`  
原文：https://vllm.ai/blog/2025-02-17-distributed-inference  
2025-02-17。图在原网页。和 TensorRT-LLM 手册第 4 章同一张地图：通信才是约束。OOM 的两条路——降精度，或切开。量化单独不够，模型过千亿还是要切。

推理不是训练：形状在变，要低延迟，还要管 KV、投机解码、prefill→decode。vLLM 给出的刀：**节点内 TP**，**跨节点 PP**，再加通信 kernel 和控制面，少让 CPU 挡 GPU。

## Tensor parallelism

沿 Megatron-LM（Shoeybi et al., 2019）的路，为推理改过。列并行切权重的列、算完拼接；行并行切行、算完相加。Llama MLP：up-projection 列并行 → SILU 在分片上做 → down-projection 行并行 + **all-reduce**。权重切开等于多卡一起啃显存带宽，延迟能降。走廊必须快：NVLink / InfiniBand。

## Pipeline parallelism

一层楼装不下多卡节点时（DeepSeek R1、Llama 3.1 405B），按**连续的层**切开，激活 send/recv 一次过一站。通信比 TP 的 All-Reduce 轻，但**不天生降延迟**。vLLM 用 pipeline scheduling / micro-batch 让卡别闲着。

口诀（与 TRT-LLM 几乎同句）：节点间互联慢 → **节点内 TP、节点间 PP**；NVLink/IB 够快，TP 可以跨节点。两者一起用，是为了少付不该付的通信税。

## 超线性：KV 的房子

吞吐有时不是「2 张卡 = 2×」。切开以后每卡给 KV 的空位涨得比线性快，batch 才能长大。文中图：TP=1 → TP=2，KV block 大约 **13.9×**，token 吞吐大约 **3.9×**——不是 2×。

延伸阅读：Megatron-LM、Orca（iteration-level scheduling）、DeepSpeed、FasterTransformer。当时还点名往后看 MoE 的 expert parallelism 和更多量化——大规模 serving 那篇把 EP 写成了主菜。
