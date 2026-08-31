---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 大规模 serving：DeepSeek @ 2.2k tok/s/H200

英文对照：`en/vllm/blog/serving/large-scale.md`  
原文：https://vllm.ai/blog/2025-12-17-large-scale-serving  
2025-12-17。v0.11.0 拆掉 V0 最后一行。社区当时 1969 位贡献者、一个月 950+ commit。Coreweave H200 + IB ConnectX-7，生产形态多节点，持续约 **2.2k tokens/s/H200**（早先约 1.5k）。图在原网页。

优化清单：async scheduling、**dual-batch overlap (DBO)**、P/D 分离、CUDA graph `FULL_AND_PIECEWISE`、DeepGEMM 默认、DeepEP、**EPLB**、DeepSeek-R1 的 SiLU kernel。

## Wide-EP

DeepSeek-R1：671B 里每步只活 **37B**。MLA 不适合纯 TP——latent 投影会在每个 shard 上复制。`--enable-expert-parallel`：专家在 rank 间共享，token 被路由到该去的专家。Wide-EP = EP + **数据并行**（`mp` 或 `ray`）。DP 下 attention 各管各的 latent，有效 batch 才能涨。通信用 DeepEP all-to-all（还有 Perplexity MoE、NCCL AllGather-ReduceScatter）。文档：vLLM MoE kernel。

## Dual-batch overlap

`--enable-dbo`。先 `all_reduce` 决定值不值得切微批（`--dbo-decode-token-threshold`），两个 worker 线程交错：一个在等 MoE dispatch 时把 GPU 借给另一个。EP 度越高，通信越胖，这刀越有用。

## EPLB

训练时专家负载是匀的；线上不是。`--enable-eplb`：滑动窗口统计每 token 负载，到点算新的 logical→physical 映射，**热替换权重、不重启**。DeepSeek 的 hierarchical / global 策略。

## 为什么 MoE 更需要 P/D 分离

专家散在各 rank，一条 prefill 可能拖住**整组** EP 的 combine。compute-bound 的 prefill 和 decode 拆开，还能让 DeepEP 分别走高吞吐 / 低延迟 kernel。DistServe（Hao AI Lab, 2024）是这条路的名字。

## 三条部署走廊

- **llm-d**：K8s-native，「Wide EP well-lit path」可复现文中数字。
- **Dynamo**：KV-aware 路由、KV Block Manager、Planner；vLLM + wide-EP 是一等公民。
- **Ray Serve LLM**：P/D、DP attention、prefix 亲和路由；NIXL / LMCache；和 Ray 的数据 / RL 栈接在一起。

当时路线图：弹性 EP、长上下文、CPU 传 KV、确定性 / batch invariance、更大 MoE 融合、FlashInfer SwapAB、P/D 两侧独立 TP、GB200。活页：roadmap.vllm.ai。

必读 serving 线在这里收束：先会切卡（distributed），再有集群盘子（stack / AIBrix），再有认得 KV 和 P/D 的路由器，多模态再拆编码器，最后 Wide-EP 把 DeepSeek 那样的稀疏 MoE 铺到多机。CATALOG 里还有几十篇 day-0 模型文，不是这条主线。
