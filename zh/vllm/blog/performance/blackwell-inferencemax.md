---
source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# InferenceMAX：Blackwell 相对 Hopper 是整条 Pareto，不是一个点

英文对照：[en/vllm/blog/performance/blackwell-inferencemax.md](../../../../en/vllm/blog/performance/blackwell-inferencemax.md)  
原文：https://vllm.ai/blog/2025-10-09-blackwell-inferencemax  
2025-10-09。**vLLM Team**。同一家人后来的 Pareto：[gpt-oss-optimizations.md](gpt-oss-optimizations.md)。Day-0 gpt-oss：[../serving/gpt-oss.md](../serving/gpt-oss.md)。榜：[inferencemax.ai](http://inferencemax.ai)；代码 [InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX)；说明 [SemiAnalysis newsletter](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)。数字跟 **当天** 曲线走——不是永久铭牌。

和 NVIDIA 一起在 Blackwell **B200/GB200** 上磨了几个月：更多 HBM 带宽，原生 **FP4** Tensor Core。Kernel 和调度一起改。他们报：在相近延迟下，吞吐相对 Hopper 最高约 **4×**（gpt-oss 120B、Llama 3.3 70B）。「一百多个 PR」。

## InferenceMAX 是什么

自动、**每天** 重跑的 serving 基准，好让软件改动出现在公开榜上。当时两只模型：

- MoE：**gpt-oss 120B**
- 稠密：**Llama 3.3 70B**

三档 ISL/OSL：

| ISL / OSL | 他们讲的故事 |
|---|---|
| 1K / 1K | 聊天，中等 |
| 1K / 8K | 推理，长输出 |
| 8K / 1K | 摘要，长输入 |

单点 TPS 会骗人：最高吞吐的配置很少是最低每用户延迟。他们画的是 **Pareto frontier**（响应性对吞吐）。

点名的 Blackwell 硬件：每张 B200 **192 GB HBM3e、8 TB/s**，NVLink 每卡 **1.8 TB/s**，第五代 Tensor Core + FP4。

本地图（原文版权仍归原站；学习对照用）：

![gpt oss 120b 1k 1k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/01-gpt-oss-120b-1k-1k.png)

![llama 70b 1k 8k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/02-llama-70b-1k-8k.png)

**Fig 1：** gpt-oss-120b 1k/1k Pareto，Blackwell 对 Hopper——相近 interactivity 下吞吐最高约 **4.3×**。  
**Fig 2：** Llama 3.3 70B 1k/8k——最高约 **3.7×**。用 SemiAnalysis 配置可复现。

## 他们列的优化

**性能**

- **FlashInfer：** GQA / MLA 的 FP8 attention；快的 FP8/FP4 GEMM；MoE；融合算子。例子：AllReduce + RMSNorm + quant 一次 launch。点名的栈：CUTLASS、CuTeDSL、cuBLAS、cuDNN、TRTLLM。
- **torch.compile 融合：** Attention + Output Quant；AllReduce + RMSNorm + Quant——跨架构复用，不必按模型手熔。
- **`--async-scheduling`：** 模型执行和 host 准备重叠，GPU 不必为同步空转。

**可用性**

- 自动认出量化、选 attention backend（Blackwell → FlashInfer / TRTLLM kernel，否则 FlashAttention）——不必手调一锅环境变量。
- FlashInfer GEMM/MoE **启动时 autotune**（按 batch / ISL / OSL 选 tactic）。
- [Quick-start recipes](https://github.com/vllm-project/recipes)：起服、对精度、打基准。

## 当时还在做的

集群规模上的 speculative decoding + **Data+Expert Parallel（DEP）**，给 DeepSeek、Qwen、gpt-oss。NVIDIA `gpt-oss-120b-Eagle3-v2`：他们预期吞吐大约 **2–3×**。DEP 吃 1800 GB/s 的 NVLink，并发想比 InferenceMAX 当时展示的更高。

## 致谢

和后来 gpt-oss Pareto 那篇同一串名字（Red Hat / NVIDIA / vLLM / Meta）。SemiAnalysis：Kimbo Chen、Dylan Patel 等。
