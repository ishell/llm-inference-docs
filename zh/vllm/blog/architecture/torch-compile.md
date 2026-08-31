---
source: https://vllm.ai/blog/2025-08-20-torch-compile
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# torch.compile：把优化从模型作者手里拿走

英文对照：`en/vllm/blog/architecture/torch-compile.md`  
原文：https://vllm.ai/blog/2025-08-20-torch-compile  
2025-08-20。从 Red Hat + Berkeley 的 office hours 整理。图在原网页。`optimization.md` 里的 `-O0`～`-O3`、`--enforce-eager`，底下就是这篇。

手写 kernel 能摸到天花板，但要为每一种模型、每一家硬件付一次税。`torch.compile` 是 JIT：给函数或 `nn.Module` 套一层，前端把张量运算收成图，后端再吐出融合过的 kernel。TorchBench 上他们引用的是 **1.8–2×** geomean——那是「先有一条不丢人的基线」，不是替你写完 FlashAttention。

vLLM 不只把它当加速器。原则是：**模型定义保持干净，优化发生在编译时。** 几百个模型不用每个都去拆抽象。

## 两段管道

**TorchDynamo（前端）。** 自定义字节码解释器，抽出只含 Tensor op 的 `torch.fx` 直线图。看见它不会的东西（磁盘 I/O 一类）就 **graph break**：当前图结束，跑那条不支持的语句，再开一张新图。不会为了一句 `torch.save` 把整座房子拆掉。

**TorchInductor（后端）。** 图上来了：融合 pointwise / reduction，autotune block size，matmul 在 cuBLAS / Triton / CUTLASS 之间选，还能套 CUDA graph。CUDA graph 要求静态地址、纯 CUDA；编译器会在不支持的 op 处切开，自己管静态输入缓冲。

## vLLM 怎么接

V1 在线 / 离线默认开。关掉：`-O0` 或 `--enforce-eager`。冷启动把 FX 图和 Triton kernel 写进 `~/.cache/vllm/torch_compile_cache`；热启动再读。`VLLM_DISABLE_COMPILE_CACHE=1` 可关。同一环境的机器之间可以拷这份 cache——autoscaling 时先烤一次、再分给新实例。

默认一张 **dynamic batch** 的图伺候所有 batch size。知道自己只会跑 1/2/4 时，`compile_sizes: [1, 2, 4]` 可以特化、多做一点 autotune。

不是所有 op 都能进 CUDA graph（cascade attention 就不行）。vLLM 把图画成 CUDA-graph-safe 与 unsafe 两段，分开执行——这就是 `-O1` 的 PIECEWISE 和 `-O2` 的 `FULL_AND_PIECEWISE`。

## 自定义 pass：为什么不改模型文件

模型作者按层、按模块写，正确优先。峰值性能常常要跨模块融合。vLLM 在 `torch.fx` 上改图，不改那几百份模型定义。

当时的例子：量化 MLP 里 SiLU 后面紧跟量化 down-proj。两个 op 单独都是 memory-bound。`ActivationFusionPass` 收成一个 fused kernel，吞吐最多大约 **+8%**（Llama 3.1 405B FP8，8×MI300）。后来 Inductor 自己能把 torch 版的 quant 和 SiLU 熔掉，这条 pass 在部分路径上过时了；凡是涉及 **自定义 op**（attention、通信、sub-byte quant）的融合，仍然要自己写 pass。

更狠的一刀：**Sequence Parallelism + Async TP**。TP 的 GEMM 和 all-reduce 分开跑，GPU 会在网上闲着。把 all-reduce 拆成 reduce_scatter + all_gather，把 all_gather 推到 layernorm 之后，才能和下一层 GEMM 熔成 GEMM+collective。这件事如果写进模型定义，要碰 vLLM 支持的每一个架构。社区贡献者做成两条 compile pass，CLI 打开，所有模型一起受益。文中引用最多大约 **+10%**。

当时已经有的融合：RMSNorm+FP8 quant、SiLU-Mul+FP8 quant、Attention+FP8 quant（约 +7%）、AllReduce+RMSNorm（约 +15%）、再加 FP4 变体。还有消掉多余 reshape 的 no-op pass。Pass 可以从 `PostGradPassManager`、`--compilation-config` 或离线 config 加——不必改 vLLM 源码。

## 当时还在修的

大量私有（下划线开头的）compile API，为的是 serving 时不要重编译。PyTorch 2.8 前后开始把推理需要的能力收进稳定面。启动时间是 autoscaling 的痛：冷、热都要收。`-O0`…`-O3` 那套 revamp 就是把「门开得快」和「跑得稳」做成一档礼貌。实验方向还有 MPK/Mirage 那种整网一个 megakernel。

读完 [V1](v1-alpha.md) 和 [MRV2](mrv2.md) 再读这一篇：V1 决定默认编译；MRV2 把 runner 的记账搬到 GPU 上；compile 则是「不要为了融合去改每一份模型」。
