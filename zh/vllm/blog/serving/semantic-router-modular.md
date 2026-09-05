---
source: https://vllm.ai/blog/2025-10-27-semantic-router-modular
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 模块化 LoRA：别为每个分类器跑一整只 BERT

英文对照：[en/vllm/blog/serving/semantic-router-modular.md](../../../../en/vllm/blog/serving/semantic-router-modular.md)  
原文：https://vllm.ai/blog/2025-10-27-semantic-router-modular  
2025-10-27。署名 **Ivar Flakstad (Hugging Face), OneZero-Y, Huamin Chen (Red Hat), Xunzhuo Liu (Tencent)**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。共享基座 LoRA 落进 [Iris](semantic-router-iris.md)。脊柱：[signal-decision](semantic-router-signal.md)。HaluGate 走同一扇 Candle 门：[halugate](halugate.md)。后来换 mmBERT：[athena](semantic-router-athena.md)。不要和引擎里的 [Router](router.md) 混。Flash Attention 的 2× / tok/s 是**文献引用**，不是 vLLM-SR 集群实测。

同目录还有：[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[vision](semantic-router-vision.md)、[themis](semantic-router-themis.md)、[session](semantic-router-session.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)、[mom](semantic-router-mom.md)。

每条分类请求独立跑好几只微调模型，成本随模型数 **线性** 涨。这篇是 Rust 分类层的重构：架构模块化、LoRA、并发。

本地图（原文版权仍归原站；学习对照用）：

![modular](../../../../assets/vllm/blog/serving/semantic-router-modular/01-modular.png)

**Figure 1.** 分层的 candle-binding：核不绑死某一架构；`DualPathUnifiedClassifier` 在全微调和 LoRA 之间选。

## 背景：从 BERT 走到模块系统

以前：BERT / ModernBERT 做意图和 jailbreak。ModernBERT 英语分类强，他们点名的上限：

- **语言覆盖：** 原版 ModernBERT 多语言比训在更杂数据上的模型薄。页上注：[mmBERT](https://huggingface.co/blog/mmbert)（1800+ 语言）是这次重构 **开始之后** 才发的——多语言问题的另一条路，后来被 [Athena](semantic-router-athena.md) 收成中心。
- **上下文：** ModernBERT 经 RoPE 到 **8,192**；Qwen3-Embedding 引用 **32,768**。
- **模型耦合：** 分类逻辑绑死具体架构，加新模型难。

模块化：更新的模型（含 mmBERT）可以和 Qwen3-Embedding、EmbeddingGemma 并列；router 按任务挑。

## 架构重组

**candle-binding** crate 里分层。核不依赖某一只模型；加新架构不必改旧代码。`DualPathUnifiedClassifier` 按任务在传统全微调和 LoRA 适配之间选。

## 长上下文 embedding

### Qwen3-Embedding

最长 **32,768**（[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)）。RoPE 撑更长上下文。模型卡写训在 **100+** 语言的文本上——ModernBERT-only 吃力的多语言路由。

### EmbeddingGemma-300M

更小、盯质量。上下文 **2,048**。Matryoshka：embedding 截到 **768 / 512 / 256 / 128** 不必重训（[embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)）。Multi-Query Attention：**3** 个 query 头、**1** 个 KV 头。Transformer 块之后 dense bottleneck：**768 → 3072 → 768**。

## 多任务分类用 LoRA

朴素：意图 + PII + jailbreak = 三次完整微调前向。

![full params](../../../../assets/vllm/blog/serving/semantic-router-modular/02-full-params.png)

**Figure 2.** 每个任务付一整只基座 Transformer。任务数 O(n)。

LoRA 共享基座一遍：

![lora](../../../../assets/vllm/blog/serving/semantic-router-modular/03-lora.png)

**Figure 3.** 基座一次前向；适配器通常 **<1%** 参数。

`parallel_engine.rs` 用 [Rayon](https://github.com/rayon-rs/rayon) 让适配器并发。三次分类：一次完整前向 + 三只轻适配器，不是三只完整模型。LoRA 赚在 **多任务**，单任务未必（没有可共享时，全微调可能更快）。

## 并发走 `OnceLock`

全局分类器状态以前用 `lazy_static`，并发负载下锁争用。换成 [`OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)：初始化之后无锁读（指针读，无同步）。`oncelock_concurrent_test.rs`：**10** 线程、**30** 次分类；吞吐宣称随线程数线性。并发请求不再在 mutex 后面排队。

### GPU 上的 Flash Attention

Flash Attention 2 作为 CUDA 构建的可选功能；Ampere+（compute capability ≥ 8.0）。注意力在片上 SRAM 里分块，少反复读 DRAM。

页上的引用（不是 vLLM-SR 集群实测）：

- ModernBERT：自注意力最多约 **3×**，内存更少；隔两层全局、其余局部滑动窗
- Qwen3：FlashAttention-2 注意力最多约 **4×**；14B 变体 **70–110** 对 **30–35** tok/s，长上下文更明显

Cargo feature：没有兼容 GPU 也能部署；硬件支持再打开。

## 跨语言接入

Rust 分类引擎 + **Go FFI**。

**为什么 Rust：** 近 C 的性能；内存安全；所有权挡数据竞争（配 Rayon）；没有 GC 停顿。Candle 踩这套。

**为什么 Go FFI：** Envoy `ext_proc` 过滤器是 Go——FFI 让过滤器调 Rust 分类，不必重写 Envoy 层。Kubernetes operator（controller-runtime）可以把分类嵌进去，不必再跳一次网络。服务网格（Istio、Linkerd、Consul）和带 Go 组件的 API 网关，可以留下 ML 分类，不加微服务。

**部署弹性：**

- **嵌入：** Go 经 CGO 链上 Rust 库
- **进程隔离：** 单独进程，gRPC 或 Unix socket
- **混合：** Go 管网络 / 编排，Rust 推理

主路由、配置、cache 在 Go；吃算力的分类在 Rust。

## 性能特征（他们列的）

- **单任务对多任务：** 没有可共享时 LoRA 帮不上忙；同一输入上多任务，适配器才划算。加速比 = 基座计算对适配器计算。
- **长上下文：** Qwen3-Embedding 到 32K 不必截断，对 ModernBERT 的 8K。兼容 GPU 上 FA2 随长度更赚。
- **多语言：** ModernBERT 训练薄的语言也能路由。
- **高并发：** `OnceLock` 去掉锁争用；分类吞吐跟 CPU 核走。
- **GPU：** FA2 注意力 **3–4×**（引用），长序列更明显。

## 往后

实现 `CoreModel` trait 就能加 embedding；Candle 有 Flash Attention 3 再接；4-bit / 8-bit 量化；领域路由的自定义 LoRA；Python / Java / C++ FFI。模块化底座让研究能落进来，不必再改架构；FFI 稳住，Rust 在 Go 部署下面自己长。

## 资源

- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [Candle](https://github.com/huggingface/candle)
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
