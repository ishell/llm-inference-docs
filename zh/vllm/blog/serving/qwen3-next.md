---
source: https://vllm.ai/blog/2025-09-11-qwen3-next
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Qwen3-Next：Gated DeltaNet + 满 attention 交错，hybrid KV 按物理页对齐

英文对照：[en/vllm/blog/serving/qwen3-next.md](../../../../en/vllm/blog/serving/qwen3-next.md)  
原文：https://vllm.ai/blog/2025-09-11-qwen3-next  
2025-09-11。署名 **The vLLM Team**。当时 nightly。80B-A3B，1:50 MoE。后继 3.5/3.8 见 [qwen35-25k-tps.md](qwen35-25k-tps.md) / [qwen38.md](qwen38.md)。hybrid 线性 serving：[hybrid-ssm.md](hybrid-ssm.md)。Qwen3.5 的 GDN+P/D 是这条线的后续，不是同一篇。

本地图（原文版权仍归原站；学习对照用）：

![qwen](../../../../assets/vllm/blog/serving/qwen3-next/01-qwen.png)

**原文 TL;DR：**

- Hybrid attention：Gated DeltaNet（线性）和满 attention 交错；目标 **65K+**。
- Hybrid KV manager 把满 attention 的逻辑 block 调到和线性层状态一样大的物理页。
- Triton launch 在 decode-only 上 CPU 贵，所以默认 full CUDA graph。
- MTP 引擎侧原生。当时 roadmap：GDN kernel、hybrid 上的 prefix cache 和 P/D。

## Quickstart

当时 nightly：

```
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly --torch-backend=auto
```

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4
```

菜谱：[Qwen3-Next](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)。

## Hybrid Attention：把长上下文的账算清

标准 attention 换成：

- **Gated DeltaNet** — 线性 attention，长上下文省
- **Full Attention** — 高保真推理

两层交错；声称能高效扩到 **65K** 以外。

vLLM：Triton kernel 来自 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)；[hybrid KV cache manager](https://arxiv.org/abs/2503.18292) 同时管线性层和满 attention——少碎片，显存吃满时吞吐才上得去。

满 attention 的逻辑 block size 被调到和线性层状态占同一块**物理** GPU 页。Hybrid 的分页内存；卡塞满时才有吞吐。

![hybrid](../../../../assets/vllm/blog/serving/qwen3-next/02-hybrid.png)

**Figure。** Hybrid KV：满 attention 的逻辑 block 和线性状态对齐到同一物理页。

Flash Linear Attention 是 Triton。Triton launch 在 decode-only batch 上 CPU 贵 → 默认打开 full CUDA graph，低延迟才站得住。

## High-sparsity MoE：极端效率

MoE 激活比 **1:50**。旗舰 **80B-A3B**：每 token 只亮 **3B**。走 vLLM 自带的 MoE 路径。

## Multi-Token Prediction (MTP)

MTP：预训练效率和推理速度。vLLM 原生——一步多个 token，应用代码不用改。怎么开：看 recipe。

## Looking ahead

当时的 roadmap（这篇，不是 3.5 那篇）：

- 继续拧 GatedDeltaNet kernel
- 更好的内存管理；hybrid 上的 automatic prefix caching 和 P/D 分离
- 吞吐和 CPU 开销

Qwen3.5 的 GDN + P/D 是后续：[qwen35-25k-tps.md](qwen35-25k-tps.md)。

## Acknowledgements

Qwen Team：Tao He, Jianwei Zhang。Flash Linear Attention：Yu Zhang 等（gated deltanet 审 kernel + 数值）。NVIDIA：Vadim Gimpelson（测试）。IBM Research：Thomas Parnell（hybrid 内存 + CUDA graph）。Red Hat：Tyler Michael Smith, Doug Smith, Tarun Kumar, Elvir Crncevic（测试 + MoE kernel）。社区：Meta、Roblox。vLLM：Jie Li, Kaichao You, Chen Zhang, Simon Mo。
