---
source: https://vllm.ai/blog/2025-12-17-large-scale-serving
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 大规模 serving：DeepSeek @ 2.2k tok/s/H200

英文对照：[en/vllm/blog/serving/large-scale.md](../../../../en/vllm/blog/serving/large-scale.md)  
原文：https://vllm.ai/blog/2025-12-17-large-scale-serving  
2025-12-17。署名 **vLLM Team**。学习译文，不是官方译本。数字是这一天、这一套配方上的。后来的弹性宽度：[elastic-ep.md](elastic-ep.md)；跨实例前缀池：[mooncake.md](mooncake.md)；认得 P/D 的网关：[router.md](router.md)。

## Introduction

v0.11.0 拆掉 V0 engine 最后一行代码——完整迁到改进后的 [V1](../architecture/v1-alpha.md)。这件事离不开社区：当时（2025-12-18）**1,969** 位贡献者，过去一个月 **950+** commit。

这些努力也写进了 SemiAnalysis 开源 [InferenceMax](https://inferencemax.semianalysis.com/) 性能基准。生产信任他们点名：Meta、LinkedIn、Red Hat、Mistral、Hugging Face。

DeepSeek 风格的 Prefill/Decode 分离，加上稀疏 Mixture-of-Experts（MoE），仍是他们眼里高性能 LLM 推理的前沿形态。这篇把当时把吞吐再往前推的优化列出来：

- Async scheduling
- Dual-batch overlap（DBO）
- Disaggregated serving
- CUDA graph mode `FULL_AND_PIECEWISE`
- DeepGEMM 默认打开
- DeepEP kernel 接入
- Expert parallel load balancing（EPLB）
- DeepSeek-R1 的 SiLU kernel

邻居文章他们点了四篇：llm-d 的 [large scale serving](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga)、PyTorch 的 [disaggregated serving](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/)、NVIDIA Dynamo 的 [distributed inference](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/#boosting_inference_performance_on_nvidia_gb200_nvl72_by_30x)、Anyscale 的 [wide-EP](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm)。

本地图（原文版权仍归原站；学习对照用）。

## Results

社区基准（llm-d 文里的 [Wide-EP 数字](https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga#wide-ep-performance)）：Coreweave **H200** 集群，InfiniBand **ConnectX-7** NIC。生产形态的多节点部署，持续吞吐约 **2.2k tokens/s per H200**。

早先大约 **1.5k tokens/s per GPU**。这一跳他们点名两块：kernel（silu-mul-quant 融合、Cutlass QKV、TP attention 修复）+ Decode 上的 **Dual Batch Overlap (DBO)**。

对运营的意思很具体：同样目标 QPS，可以少买几份 replica，token-per-dollar 跟着下来。

![prefill throughput](../../../../assets/vllm/blog/serving/large-scale/01-prefill_throughput.png)

**图注（原文）。** Prefill Results。

![decode throughput](../../../../assets/vllm/blog/serving/large-scale/02-decode_throughput.png)

**图注（原文）。** Decode Results。

## Key Components

### Wide-EP

把 DeepSeek-V3 家族铺成大规模 serving，要同时记住两件事：

- **Sparse expert activation。** DeepSeek-R1：671B 里每步只活 **37B**。
- **KV cache management。** 纯 TP 不适合 DeepSeek 的 multi-head latent attention（MLA）：latent 投影会在每个 shard 上复制。

Expert parallelism（EP）就是吃这两件事、把有效 KV 做大的部署形态。vLLM 旗标：`--enable-expert-parallel`。一套专家在 rank 之间共享；forward 时 token 被路由到拥有对应专家的 rank。

![wide ep](../../../../assets/vllm/blog/serving/large-scale/03-wide_ep.gif)

**图注（原文）。** Wide-EP token routing。

Wide-EP = EP + **数据并行（DP）**。DP 可以用 `mp` 或 `ray` backend 拉起；在 Ray 集群里后一种更省事。相对 TP 的好处见下图：DeepSeek-V3 在 TP 与 EP 两种切法下每张 GPU 的内存。

TP 策略下每张 H200 大约还剩 **34GB** 空闲，但 MLA 仍会在每个 rank 上复制 latent attention 投影。DP 部署里 attention 层在各 rank 上各有一份，latent 彼此独立，整个部署的有效 batch 才能涨。

![kv cache](../../../../assets/vllm/blog/serving/large-scale/04-kv_cache.png)

EP 度越高，rank 之间同步越胖。vLLM 接了 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的高吞吐 / 低延迟两套 all-to-all kernel；另外还有 Perplexity [MoE kernels](https://github.com/perplexityai/pplx-kernels)，以及基于 NCCL 的 AllGather-ReduceScatter all-to-all。当时后端清单见 vLLM MoE [kernel docs](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)（[fused MoE modular all2all backends](https://docs.vllm.ai/en/latest/design/moe_kernel_features/#fused-moe-modular-all2all-backends)）。

![a2a backends](../../../../assets/vllm/blog/serving/large-scale/05-a2a_backends.png)

**图注（原文）。** vLLM all-to-all backends。

口诀：密层（attention）走 DP Attention，稀疏层走 EP。组大小是 `DP × TP`。后来的 [Elastic EP](elastic-ep.md) 动的就是这个 DP 个数。

### Dual-batch Overlap (DBO)

vLLM 把 DeepSeek 的 [microbatching](https://github.com/deepseek-ai/profile-data) 做成 dual batch overlap（DBO），命令行 `--enable-dbo`。把 compute 和集合通信叠起来，抬 GPU 利用率。实现分三拍：

1. Rank 之间一次集体 `all_reduce`，同意「切微批划算」。门槛可调：`--dbo-decode-token-threshold`。
2. 主线程拉起微批 worker 线程，这些 worker 做完 CUDA graph capture。
3. 模块化 MoE all-to-all kernel 基类协调微批 worker 的 launch：GPU 活还没干完时 **yield**。

下面是 DeepSeek Decode、**没有** DBO 的 profiling。`MoE Dispatch/Combine` 那一段：compute 并不重，集合通信却特别长。

![dbo before](../../../../assets/vllm/blog/serving/large-scale/06-dbo_before.png)

**图注（原文）。** Before DBO。

同一负载、**打开** DBO。第一个微批 worker 发起并完成 MoE dispatch，立刻把控制权让给第二个；第二个做完自己的 dispatch，再让回第一个；第一个做完 combine，再让给第二个做 combine。

EP 度高、通信胖的部署，这刀最有用。Elastic EP 当时**还不支持 DBO**——弹性缩容和微批重叠还没焊在一起。

![dbo after](../../../../assets/vllm/blog/serving/large-scale/07-dbo_after.png)

**图注（原文）。** After DBO。

### Expert Parallel Load Balancing (EPLB)

训练时专家负载是匀的；线上不是。NVIDIA 有一组 [MoE expert routing 实验](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/#experimental_results) 专门量这件事。

Wide-EP 上，有的 EP rank 闲着，有的在扛大批 token。vLLM 实现了 DeepSeek [EPLB](https://github.com/deepseek-ai/EPLB) 的 hierarchical / global 策略。旗标 `--enable-eplb`；窗口大小、rebalance 间隔、冗余专家、日志都可以配。

![eplb](../../../../assets/vllm/blog/serving/large-scale/08-eplb.gif)

**图注（原文）。** EPLB in action。

每次 MoE forward 记 per-token 负载，滑动窗口把统计在 EP rank 之间聚起来。到 rebalance 间隔，负载均衡器算一套新的 logical→physical 专家映射，再编排一次 **weight shuffle**——新布局生效，**不必重启模型**。

Elastic EP 的 scale-up 在切换拓扑之后立刻做一次 reshuffle；scale-down 必须**先** reshuffle，免得要离开的 rank 还握着专家。

### Disaggregated Serving

Hao AI Lab 2024 的 DistServe [论文](https://hao-ai-lab.github.io/blogs/distserve-retro/) 写的 Prefill/Decode 分离，在 EP 部署上尤其值钱。

![disaggregated serving](../../../../assets/vllm/blog/serving/large-scale/09-disaggregated_serving.gif)

**图注（原文）。** P/D disaggregation in action。

专家散在各 rank：一条请求的 token 从某一个 rank 出发，可能要被另一个 rank 上的专家处理。MoE 层之间必须同步；某个 rank 这拍没用上，也要 dummy pass，好让 layer combine 的 collective 在该到的时候接到 token。

于是一条 compute-bound 的 Prefill 可以拖住**整组** EP 的 forward——分离的收益被放大。DeepSeek 部署还可以按池子专挑 DeepEP：高吞吐 kernel 给 Prefill，低延迟 kernel 给 Decode。

[Router](router.md) 是把请求送进对的那一组；这里解释**为什么 MoE 上不拆会痛**。文本 P/D 不要和 [EPD](epd.md) 的视觉编码器分离混成一件事。

## Deployment Paths

### llm-d

llm-d 是 Kubernetes-native 的分布式推理栈，给大规模生成式模型留了几条「点亮的路」。目标是：在多数加速器与基础设施上，尽快摸到关键开源模型的 SOTA 性能。复现这篇数字，走 llm-d 的 Wide EP [well-lit path](https://github.com/llm-d/llm-d/tree/main/guides/wide-ep-lws)。

![llm d](../../../../assets/vllm/blog/serving/large-scale/10-llm-d.png)

### Dynamo

Dynamo 面向高吞吐、低延迟的生产 LLM。KV-aware 路由、KV Block Manager（cache offload）、Planner（按负载动态匹配）用来咬更紧的 SLA，同时把 GPU 铺得更开。vLLM + wide-EP 在 Dynamo 里是一等公民。细节：[Dynamo 文档](https://docs.nvidia.com/dynamo/latest/index.html)；复现这篇性能的 [example recipe](https://github.com/ai-dynamo/dynamo/pull/4463/files#diff-363ddf6952864a610a1047f6b99c52461d6de9a4e198f89eb49d34f009a4d22b)。

![dynamo](../../../../assets/vllm/blog/serving/large-scale/11-dynamo.png)

### Ray Serve LLM

建在 Ray Serve 原语上。一等公民的 serving 形态：[Prefill/Decode 分离](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html)、[data parallel attention](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/data-parallel.html)、[prefix cache-affinity 路由](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html)。卖点是模块化、在 Ray 集群（含 Kubernetes 上的 KubeRay）上好部署。和更广的 Ray 生态接在一起——数据处理、强化学习（RL）——是它和别的走廊不一样的地方。

KV 传输接 NIXL 与 LMCache connector。各阶段可以按各自的负载曲线独立 autoscaling。整层是可编程的：不难扩展、组合出别的 serving 形态。Elastic EP 的 scale 操作也依赖 Ray DP backend。

![ray serve llm](../../../../assets/vllm/blog/serving/large-scale/12-ray_serve_llm.png)

## Roadmap

当时还在做：

- Elastic expert parallelism
- Long context serving
- KV cache transfer via CPU
- Full determinism and batch invariance
- Large MoE optimizations（例如 DeepSeek-R1 与 gpt-oss 的 op fusion）
- 更好接 FlashInfer 新 kernel（例如 SwapAB）
- 分离部署里 Prefill / Decode **独立的 TP 大小**
- GB200 上的大规模 serving 优化

活页：[roadmap.vllm.ai](http://roadmap.vllm.ai)。弹性 EP 和远端 KV 几个月后各自成篇（[elastic-ep.md](elastic-ep.md)、[mooncake.md](mooncake.md)）。长上下文的序列切法见 [DCP](../performance/dcp.md)。

## Summary

- vLLM 已完整迁到 V1；DeepSeek 风格 MoE 上，wide-EP 做到 **2.2k tok/s/H200**。
- Wide-EP 给 MLA 把 KV 效率做大；DBO 与 EPLB 分别对付通信瓶颈和专家负载不均。
- Prefill/Decode 分离再把 MoE 的阅读和说话拆开。部署走廊：llm-d、Dynamo、Ray Serve LLM。

必读 serving 线在这里收成一张地图：先会切卡（[distributed-inference](distributed-inference.md)），再有集群盘子（[production-stack](production-stack.md) / [AIBrix](aibrix.md)），再有认得 KV 和 P/D 的路由器，多模态再拆编码器，Wide-EP 把 DeepSeek 那样的稀疏 MoE 铺到多机——然后 Mooncake 让跨实例的前缀不必重读，Elastic EP 让铺开的宽度不必为了加减卡而重启。
