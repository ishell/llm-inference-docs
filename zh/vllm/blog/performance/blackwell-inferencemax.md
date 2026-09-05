---
source: https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# InferenceMAX：Blackwell 相对 Hopper 是整条 Pareto，不是一个点

英文对照：[en/vllm/blog/performance/blackwell-inferencemax.md](../../../../en/vllm/blog/performance/blackwell-inferencemax.md)  
原文：https://vllm.ai/blog/2025-10-09-blackwell-inferencemax  
2025-10-09。署名 **vLLM Team**。学习译文，不是官方译本。数字跟 **当天** 曲线走——不是永久铭牌。同一家人后来的 Pareto：[gpt-oss-optimizations.md](gpt-oss-optimizations.md)。Day-0 gpt-oss：[../serving/gpt-oss.md](../serving/gpt-oss.md)。榜：[inferencemax.ai](http://inferencemax.ai)；代码 [InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX)；说明 [SemiAnalysis newsletter](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)。

适用：在 Blackwell **B200/GB200** 上读 InferenceMAX 的 Pareto（gpt-oss 120B、Llama 3.3 70B），以及他们点名的 FlashInfer / `torch.compile` / `--async-scheduling`。不适合：把 **4×**、**4.3×**、**3.7×** 当成你机房的承诺。

## Introduction

几个月里，他们和 NVIDIA 贴着最新的 Blackwell（**B200/GB200**）磨 LLM 推理。Blackwell 自己已经带了一档新性能：更多 HBM 带宽，原生 **FP4** Tensor Core。开箱就不慢。要从硅里再抠一层，就要把旧 kernel 拆开重写，再为更底层的硬件利用率长出新 kernel。

[SemiAnalysis InferenceMAX](https://github.com/InferenceMAX/InferenceMAX) 把这些改动映到公开榜上。他们报：在相近延迟下，吞吐相对上一代 Hopper 最高约 **4×**——模型点名 gpt-oss 120B 和 Llama 3.3 70B。

这是一场持续数月的工程合作，vLLM 代码库里 **一百多个 PR**。和 NVIDIA 一起，几乎整条推理管线都动过：自定义 kernel（attention、GEMM、MoE），再到高层调度、把开销拆掉。下文按优化拆开，看 Blackwell 的架构特征怎么变成生产数字。

## Overview of InferenceMax

SemiAnalysis InferenceMAX 是一套 **自动、反复** 跑的 LLM serving 基准，结果 **每天** 更新，好让软件改动出现在公开数据里。同一套方法，比较才公平、才可复现。软件发了，榜上的点也该跟着动——中间不要隔一条新闻稿那么长。

当时评两只代表性开源模型：

- Mixture-of-Experts（MoE）：**gpt-oss 120B**
- Dense：**Llama 3.3 70B**

为了贴近真实用法，每只模型扫多种 prompt / response 长度（ISL = input sequence length，OSL = output sequence length）。三档：

| ISL / OSL | 他们讲的故事 |
|---|---|
| 1K / 1K | chat，中等输入 / 输出 |
| 1K / 8K | reasoning，长输出 |
| 8K / 1K | summarization，长输入 |

## Delivering Performance Across the Pareto Frontier

Blackwell 的计算架构把推理效率往上抬了一档：最新 **HBM3e**（每张 B200 **192 GB**、**8 TB/s**），NVLink 每卡 **1.8 TB/s**，第五代 Tensor Core 加上内建 **FP4**。

Kernel 跟着这些能力改。相对 Hopper 上的同一份 vLLM，他们看见吞吐（每 GPU）和响应性（每请求延迟）都明显抬头。

现代推理的序列长度、batch、并发差得很开。给出最高吞吐的配置，很少同时给出最低每用户延迟。单点 TPS 会骗人。InferenceMAX 用 **Pareto frontier**：把响应性对吞吐的权衡画成一条包络，覆盖真实会碰到的工作点。

和 NVIDIA 合作的主目标：让 vLLM 吃到 Blackwell 的特性，**整条** Pareto 都好看——不是只刷一个工作点。

SemiAnalysis 的结果：gpt-oss 120B 和 Llama 3.3 70B，在所有 interactivity 档上，Blackwell 相对 Hopper 都一致变好。

本地图（原文版权仍归原站；学习对照用）：

![gpt oss 120b 1k 1k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/01-gpt-oss-120b-1k-1k.png)

**Figure 1。** SemiAnalysis InferenceMax，gpt-oss-120b Pareto，vLLM Blackwell 对 Hopper，1k/1k ISL/OSL，扫很宽的 interactivity。相近 interactivity 下吞吐最高约 **4.3×**。

![llama 70b 1k 8k](../../../../assets/vllm/blog/performance/blackwell-inferencemax/02-llama-70b-1k-8k.png)

**Figure 2。** 同上，Llama 3.3 70B，1k/8k ISL/OSL。吞吐最高约 **3.7×**。

**这些数字当时就能用 SemiAnalysis 给的 InferenceMAX 配置复现。** 优化软件盯着硬件能挤出什么，会走到这里。走到这些点，靠的是 vLLM 里一整套优化，和 NVIDIA 工程师贴着做。下面列当时最要紧的几条。

## vLLM Blackwell Optimizations

Blackwell 上的成绩来自整条软件栈。有的加快 GPU 上的 kernel；有的减 CPU 开销；有的把硬件特性用满。当时点名的增强：

**Performance Improvements**

- **更快的 kernel，走 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。** 接入 NVIDIA 的 FlashInfer，收进一批高性能 kernel：GQA / MLA 的 FP8 attention；快的 FP8 / FP4 GEMM；MoE；融合算子。例子：AllReduce + RMSNorm + quantization **一次** kernel launch，延迟明显下来。底层栈点名：CUTLASS、CuTeDSL、cuBLAS、cuDNN、TRTLLM。
- **更聪明的图融合。** 把 vLLM 的 `torch.compile` 图融合扩到 Attention + Output Quant、AllReduce + RMSNorm + Quant 这类算子模式。融合 kernel 的成绩不必按模型手改；更要紧的是 **跨架构复用**。
- **`--async-scheduling` 减 host 开销。** 模型执行和 host 准备完全重叠，GPU 不必为同步空转。**整段 workload 流水起来**：这一拍还在 GPU 上跑，下一拍的数据已经在旁边摆。

**Usability Improvements**

- **自动认出量化和 backend。** 模型是否量化，vLLM 自己侦测、自己选 backend；attention backend 也按 GPU 选。Blackwell 上有则走 FlashInfer attention（里头带着 NVIDIA TensorRT-LLM kernel），否则退回 FlashAttention——不必手调一锅旗标或环境变量。
- **FlashInfer GEMM / MoE 启动时 autotune。** 理想 kernel  сильно依赖 batch 和序列长度。GPU runner 里加了 autotuning：启动时 FlashInfer 做 tactic selection——打一轮、选 kernel——ISL / OSL 变了仍能贴着峰值。
- **[Quick Start Recipes](https://github.com/vllm-project/recipes)。** 代码之外，和社区一起写常见场景的起手配置。按模型、按硬件：起服、拧参数、对精度、打基准。少走弯路，更快见到数字。

## Ongoing Work

上面每一条本身都是一个不小的项目，技术上贴着 NVIDIA 做——而且还没写全。合作还在继续，前面的改进多到几乎看不过来。

往前看：集群规模上的 speculative decoding 和 **Data+Expert Parallel（DEP）**，给 DeepSeek、Qwen、gpt-oss 以及更多模型解锁吞吐。NVIDIA 的 `gpt-oss-120b-Eagle3-v2` 带 Eagle speculative decoding，他们预期吞吐大约 **2–3×**。DEP 吃 Blackwell 上 **1,800 GB/s** 的低延迟 NVLink GPU-to-GPU 互连，预期还能再往上推，并发会比 InferenceMAX 当时展示的更高。

Blackwell 上的性能每天都在动：vLLM 和 NVIDIA 继续挖效率、挖规模。他们说还在不断发现把平台往外推的新机会。

## Acknowledgements

vLLM 社区里一起做这件事的人：

- **Red Hat：** Michael Goin、Alexander Matveev、Lucas Wilkinson、Luka Govedič、Wentao Ye、Ilia Markov、Matt Bonanni、Varun Sundar Rabindranath、Bill Nell、Tyler Michael Smith、Robert Shaw
- **NVIDIA：** Po-Han Huang、Pavani Majety、Shu Wang、Elvis Chen、Zihao Ye、Duncan Moss、Kaixi Hou、Siyuan Fu、Benjamin Chislett、Xin Li、Vadim Gimpelson、Minseok Lee、Amir Samani、Elfie Guo、Lee Nau、Kushan Ahmadian、Grace Ho、Pen Chun Li
- **vLLM：** Chen Zhang、Yongye Zhu、Bowen Wang、Kaichao You、Simon Mo、Woosuk Kwon、Zhuohan Li
- **Meta：** Yang Chen、Xiaozhu Meng、Boyuan Feng、Lu Fang

InferenceMAX 全部结果：[http://inferencemax.ai](http://inferencemax.ai)。跑榜的代码开源：[https://github.com/InferenceMAX/InferenceMAX](https://github.com/InferenceMAX/InferenceMAX)。他们对结果的说明：[https://newsletter.semianalysis.com/p/inferencemax-open-source-inference](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)。

感谢 SemiAnalysis 把硬件和开源软件共设计往前推，本意是给社区一套公平的尺子。点名 Kimbo Chen、Dylan Patel，以及其他人。

他们说：后面几周、几个月还会继续拧，把能力再往外推。
