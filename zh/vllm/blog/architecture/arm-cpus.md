---
source: https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Arm CPU 上的 vLLM：瓶颈不在 GEMM

英文对照：[en/vllm/blog/architecture/arm-cpus.md](../../../../en/vllm/blog/architecture/arm-cpus.md)  
原文：https://vllm.ai/blog/2026-07-29-optimizing-vllm-on-arm-cpus  
2026-07-29。署名 **Arm Team**。学习译文，不是官方译本。数字是 Neoverse 上相对 2025-10 BF16 基线的演示，不是 SLA。另一条「硬件进主干」的路见 [hardware-plugin](hardware-plugin.md)；PagedAttention 本身见 [paged-attention](paged-attention.md)；INT8 / W4A16 亲戚：[autoround-llmc](autoround-llmc.md)；CPU 对 Arc XPU：[intel-arc](intel-arc.md)。

适用：Arm Neoverse 服务器，要 wheel、chunked prefill / prefix cache、INT8 W8A8 / W4A8。不适合：把页上的 **2.7–6.2×** 当承诺——热力图把 allocator 贡献拿掉了。

## Introduction

CPU 上侍候大模型，是一条正经的部署路：成本低、基建简单、云和机房里到处都是。Arm® Neoverse™ 服务器铺开之后，开源 serving（比如 vLLM）在 Arm CPU 上就得能用、功能齐、还要快。

几个月里，他们和 vLLM、PyTorch、oneDNN、KleidiAI 社区一起往上游送。结果是更好用、模型和功能更齐，以及任何跑 vLLM 的 Neoverse 服务器都能吃到的性能增益。

下文先讲能用和覆盖，再挖主要优化，以及端到端 serving 数字。

## Enablement

性能之外，他们把 Arm® CPU 上的 vLLM 用起来、功能补齐，好在 Arm® 服务器上部署。

点名的 enablement：

- 预编译 [wheel](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_2:~:text=venv/bin/activate-,Pre%2Dbuilt%20wheels,%C2%B6,-When%20specifying%20the) 与 [Docker image](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/#arm-aarch64_4:~:text=%C2%B6-,Pre%2Dbuilt%20images,%C2%B6,-Intel/AMD%20x86)
- 崩溃、精度、线程、CPU 利用率的修复
- chunked prefill、prefix caching
- INT8 W8A8 与 INT8 W4A8
- GPT-OSS、Whisper、Qwen 3.5 / 3.6
- 跟 [PyTorch](https://github.com/pytorch/pytorch) 和 [UXL](https://github.com/uxlfoundation) 生态接得更紧

能用之后，才去找、去拆性能瓶颈。

## Performance Improvements

2025 年 10 月第一次在 Arm CPU 上打 vLLM，成绩远低于预期——尽管大约 **80%** 的模型运行时已经花在派给高度优化 BF16 GEMM 的密集层上。那些层背后的独立 GEMM kernel 已经贴着该有的硬件效率，所以最大的增益不太可能只从 GEMM kernel 来。

profile 指向更宽的问题：allocator 行为、运行时同步、框架开销、attention kernel、量化执行。

### Memory Allocation

LLM serving 把 CPU 内存分配器压得很紧。Prefill 和 Decode 里，vLLM 反复为调度、KV-cache 管理、中间算子输出申请 / 释放张量。最初的 bench 里，分配就是瓶颈：大块复用差，缺页很多。

根因：PyTorch 用 glibc `malloc`。大分配跨反复的推理步复用不好；线程一多，alloc/free 变成互抢。早期 workaround：预加载 caching allocator——多一步手动配置，成绩绑在运行时选项上。

为了开箱就快，他们让 [mimalloc](https://github.com/microsoft/mimalloc) 成为 PyTorch 在 Arm CPU 上的 **默认** 分配器。mimalloc 是 caching allocator，按多线程分配压力设计。选它：TorchBench 上一大片负载成绩好；非 Arm Linux 构建里它已经是 PyTorch 依赖。

Llama 3.1 8B 开箱：离线吞吐 **2.3×**；低并发 serving 大约 **7×**。

> 文中所有性能图都 **去掉** allocator 贡献——否则刻度会被它占满，别的优化看不见。图上因此只剩栈里其余部分。

### Synchronization at High Core Counts

分配改善之后，下一只瓶颈出现在往更高核数缩放时。过了某个点，再加核不涨吞吐，甚至倒退。

高线程数下对单层做 profile。一份结果：paged attention 时间的 **74%** 耗在 OpenMP 动态调度：

```text
97.94% gomp_thread_start
  90.08% paged_attention_v1_impl
    74.07% gomp_iter_dynamic_next
     7.00% reduceValueBlock::lambda(int)
```

`gomp_iter_dynamic_next` 是 libgomp 动态循环调度路径。这条路上，运行时用 atomic fetch-add 把 loop chunk 分给 worker。PyTorch wheel 里那份 libgomp，把这次原子更新做成 load-linked / store-conditional 重试环：

```c
for (;;) {
    long old = LDXR(p);
    long newv = old + delta;
    int fail = STLXR(p, newv);
    if (fail == 0) {
        DMB_ISH();
        return old;
    }
}
```

高核数：许多 worker 抢同一处原子更新 → 反复失败的 store，加上重试流量。

追到汇编，看见一次错过的硬件机会。bench 机是 Neoverse™ V2，支持 [Arm Large System Extensions（LSE）](https://learn.arm.com/learning-paths/servers-and-cloud-computing/lse/example/)。LSE 提供硬件原子指令，例如 `LDADDAL`，可以换掉上面那个低效环。PyTorch 用的 OpenMP 运行时没有用 LSE atomics。

改法：在 PyTorch 里编一份会在能用的 CPU 上走 LSE atomics 的 libgomp。

Llama 3.1 8B：离线吞吐 **+9%**；低并发 serving 的 TPOT **−15%**。

### Dense-Layer Layout Overhead

allocator 和运行时改完，密集层仍把性能留在桌上。高性能 GEMM 对权重 layout 敏感：要跑得有效率，权重得是 blocked 格式，贴着 kernel 的向量化和 cache 访问。不预打包，每次调用都可能从框架张量 layout 转成 kernel 格式，付一次税。

低并发更贵：packing 成本摊不到大 batch 上。改法：给密集层开一条快的 oneDNN 路径，由 Compute Library for Arm Architecture 加速。这条路让 vLLM 在模型 warmup 时把 BF16 权重打成 kernel 要的格式，推理时复用这份打包表示。

Llama 3.1 8B：离线吞吐 **+16%**；低并发 TPOT **−60%**。

### Paged Attention

CPU paged attention kernel 没有为 Arm CPU 调过。QK / PV 矩阵乘，以及 softmax 里的指数，掉进参考实现。于是 prefill 靠 PyTorch 的 Scaled Dot-Product Attention——Arm CPU 路径上 chunked prefill 和 prefix caching 开不了。

QK / PV 用定制 GEMM，走 Arm [BFMMLA](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00-a) Advanced SIMD。softmax 指数：快的向量化三次多项式近似。

paged attention 最多约 **4×**；Llama 3.1 8B 离线吞吐 **+12%**。而且 Arm CPU 上 prefill 也能走 paged attention，chunked prefill 和 prefix caching 才开得了。

### BF16 Performance Improvements

同步、权重预打包、paged attention 合在一起，BF16 serving 基线比 2025 年 10 月出发时更硬。

![heatmap bf16 optimized vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/01-heatmap_bf16_optimized_vs_bf16_baseline.png)

**Figure。** 优化后的 BF16 serving 相对 2025-10 BF16 基线（学习对照）。

### INT8 W8A8 (8-bit weights and activations)

LLM 推理在 Prefill / Decode 里反复读大权重矩阵。权重存 INT8 而不是 BF16，带宽压力小，同样内存预算能塞更大模型。

带 I8MM 的 Arm CPU 上，W8A8 还映射到 [`SMMLA`](https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/smmla)——Arm 的有符号 INT8 矩阵乘加指令，相对 BF16 理论 matmul 吞吐 **2×**。

加速这条量化路径：[oneDNN](https://github.com/uxlfoundation/oneDNN) JIT kernel，在 SVE128 / SVE256 上用 `SMMLA`。

于是多份 Hugging Face INT8 W8A8 checkpoint 开箱就好，包括 `RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8` 和 `RedHatAI/whisper-large-v3-quantized.w8a8`。

相对优化后的 BF16 基线，per-token 激活量化、channelwise 权重量化的 W8A8：吞吐最多 **+88%**，TPOT **−45%**，TTFT **−54%**，看并发。

![heatmap int8 vs bf16 optimized](../../../../assets/vllm/blog/architecture/arm-cpus/02-heatmap_int8_vs_bf16_optimized.png)

**Figure。** INT8 W8A8 serving 相对优化后的 BF16 路径（学习对照）。

> Arm 上的 INT8 W8A8，可跟这条 [Arm Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/vllm-benchmark-quantisation/)。

### INT8 W4A8 (4-bit weights, 8-bit activations)

W4A8 把同一件事再推一档：权重量化到 INT4，推理时带宽更松。低并发尤其有用——batch 小，读权重的成本摊不掉。

加速走 [KleidiAI](https://github.com/ARM-software/kleidiai) 的 INT4 micro-kernel。

相对上面的 W8A8 基线，同样是 per-token 激活量化、channelwise 权重量化：吞吐最多 **+29%**，TPOT **−26%**，TTFT **−18%**，看并发。

最大的 W4A8 加速出现在低并发、推理大体 memory-bound 的时候——和预期一致。

![heatmap int4 vs int8](../../../../assets/vllm/blog/architecture/arm-cpus/03-heatmap_int4_vs_int8.png)

**Figure。** INT8 W4A8 serving 相对 INT8 W8A8 路径（学习对照）。

> 怎么用 llm-compressor 量化到 INT8 W4A8，见 [这些文档](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w4a8/)。

## Summary

Arm CPU 上的 vLLM，在能用、稳健、模型和功能覆盖、性能上都动过很大一截。

相对 2025-10 BF16 基线：

- 优化后的 BF16：serving 吞吐最多 **2.7×**
- INT8 W8A8：吞吐最多 **4.8×**，TPOT **5.7×**
- INT8 W4A8：最好——吞吐最多 **6.2×**，TPOT **7.8×**，TTFT **2.6×**

增益来自整条 CPU 推理栈：内存分配、OpenMP 同步、密集层预打包、paged attention、量化——不是只调 GEMM kernel。

![bars all vs bf16 baseline](../../../../assets/vllm/blog/architecture/arm-cpus/04-bars_all_vs_bf16_baseline.png)

**Figure。** 优化 BF16、INT8 W8A8、INT8 W4A8 相对 2025-10 BF16 的 serving 加速（学习对照）。

测到的性能之外，功能更齐、开箱更好用、上游接得更紧、模型更多——更像能上生产的 Neoverse 推理栈。

## Acknowledgements

感谢 vLLM 社区持续的支持和协作。

特别感谢 **[Li Jiang](https://github.com/bigPYJ1151)**（Intel®）维护 vLLM CPU backend，并实现了这项工作所依赖的大量基建。也感谢 **[Sanket Kale](https://github.com/sanketkaleoss)**（Fujitsu）最初在 vLLM 里做的 Arm CPU enablement，以及 **[Shreyas](https://github.com/Shreyas-fuj)**（Fujitsu）给 oneDNN 贡献的 SVE256 INT8 kernel。

Arm 是 Arm Limited（或其子公司 / 关联公司）的注册商标。PyTorch 是 The Linux Foundation 的商标。Intel 和 oneDNN 是 Intel Corporation 或其子公司的商标。原文版权 2026 Arm Limited and/or its affiliates（`open-source-office@arm.com`）。
