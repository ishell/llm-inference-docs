---
source: https://vllm.ai/blog/2025-10-27-semantic-router-modular
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 模块化 LoRA：别为每个分类器跑一整只 BERT

英文对照：[en/vllm/blog/serving/semantic-router-modular.md](../../../../en/vllm/blog/serving/semantic-router-modular.md)  
原文：https://vllm.ai/blog/2025-10-27-semantic-router-modular  
2025-10-27。署名 **Ivar Flakstad (Hugging Face), OneZero-Y, Huamin Chen (Red Hat), Xunzhuo Liu (Tencent)**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。共享基座 LoRA 落地见 [Iris](semantic-router-iris.md)。信号脊柱：[semantic-router-signal](semantic-router-signal.md)。后来：[athena](semantic-router-athena.md)、[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[vision](semantic-router-vision.md)、[themis](semantic-router-themis.md)。不要和引擎里的 [Router](router.md) 混。Flash Attention 2 加速（ModernBERT 约 **3×**，Qwen3 约 **4×**，14B 70–110 对 30–35 tok/s）是 **文献引用**，不是 vLLM-SR 集群跑出来的。LoRA「**<1%** 参数」和 10 线程 / 30 次分类测试是他们的。

同目录还有：[session](semantic-router-session.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)、[mom](semantic-router-mom.md)。

每个分类请求各自跑几只微调模型，成本随模型数 **线性** 涨。这篇是 Rust 分类层的重构：架构模块化、LoRA、并发。

本地图（原文版权仍归原站；学习对照用）：

![modular](../../../../assets/vllm/blog/serving/semantic-router-modular/01-modular.png)

**Figure 1.** 分层的 candle-binding：核心不绑死某一架构。

## 背景：从 BERT 到模块化系统

上一版意图和 jailbreak 主要靠 BERT / ModernBERT。ModernBERT 英语分类强。他们点的限制：

- **语言覆盖**：原版 ModernBERT 的多语，薄过在更多样数据上训的模型。页上注明：[mmBERT](https://huggingface.co/blog/mmbert)（1800+ 语言）是这次重构 **开始之后** 才发的——另一条多语路，不是这补丁训出来的。
- **上下文长度**：ModernBERT 用 RoPE 到 **8,192** token（[Transformers 文档](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/modernbert)）。他们引的 Qwen3-Embedding 是 **32,768**。
- **模型耦合**：分类逻辑绑死具体架构，加新模型难。

模块化架构让后来的模型（mmBERT、Qwen3-Embedding、EmbeddingGemma）可以并排坐；router 按任务挑。

## 架构重组

**candle-binding** crate 里的分层。核心不依赖某一只模型；新架构可以加，不用改旧代码。`DualPathUnifiedClassifier` 按任务在传统全微调和 LoRA 适配之间选。

## 长上下文 embedding 模型

### Qwen3-Embedding

上下文到 **32,768** token（[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)）。RoPE 撑更长距离的频率分辨率。训练文本覆盖 **100+** 语言（同一张 model card）——ModernBERT-only 路由撞上的多语缺口。

### EmbeddingGemma-300M

Google 更小的 embedding。上下文 **2,048** token。**Matryoshka**：embedding 可截到 **768 / 512 / 256 / 128** 维，不用重训（[google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)）。

**MQA**：3 个 query head，1 个 key-value head——少占内存带宽。Transformer 块后面有一层 dense bottleneck：**768 → 3072 → 768**，挂在 Matryoshka 训练故事上。

## 多任务分类的 LoRA

朴素路径：intent + PII + jailbreak = **三次完整 BERT 前向**。

![full params](../../../../assets/vllm/blog/serving/semantic-router-modular/02-full-params.png)

**Figure 2.** 三只独立微调：n 个任务就是 O(n) 次完整前向。

每只模型都付基座 transformer 的贵账。复杂度随分类任务数 **O(n)**。

LoRA 共享基座那一趟：

![lora](../../../../assets/vllm/blog/serving/semantic-router-modular/03-lora.png)

**Figure 3.** 基座一次，然后便宜的 adapter。LoRA 通常 **<1%** 参数。

基座一次 → 中间表示。每只 LoRA adapter 做任务相关的低秩更新。Adapter 通常改 **<1%** 参数；最后一步比整只模型便宜得多。

`parallel_engine.rs` 用 [Rayon](https://github.com/rayon-rs/rayon) 在 adapter 之间做数据并行。三次分类：一次完整前向 + 三次轻量 adapter，不是三次完整前向。

**LoRA 赚在多任务，不在单任务。** 单任务没有基座共享；传统全微调可能更快。加速比取决于基座计算对 adapter 计算的比例。

## 用 `OnceLock` 做并发

以前的全局分类器状态：`lazy_static`——并发负载下锁竞争。重构：标准库 [`OnceLock`](https://doc.rust-lang.org/std/sync/struct.OnceLock.html)。

第一次初始化之后，读是无锁指针读。他们点名的测试：`oncelock_concurrent_test.rs`——**10** 个并发线程，一共 **30** 次分类；他们报吞吐随线程数线性涨。`lazy_static` 下并发请求会在 mutex 后面排队。`OnceLock` 没有那道争用。

### 可选的 Flash Attention 2

CUDA build 的可选 Cargo feature。需要 **Ampere+**（compute capability **≥ 8.0**）。Attention 在片上 SRAM 里分块算，少跑 DRAM。

引用（不是 vLLM-SR 集群测量）：

- **ModernBERT**：self-attention 最高约 **3×**，显存更省（[他们链的源](https://medium.com/@alpernebikanli/some-berts-and-modernbert-39b261b1ce83)）。交替 attention：每三层一次 global，其余 local sliding-window（[Answer.AI](https://www.answer.ai/posts/2024-12-19-modernbert.html)）。
- **Qwen3**：FlashAttention-2 在 attention 上最高约 **4×**。14B：**70–110** tok/s，对着没开时的 **30–35**，长上下文更明显（[他们链的源](https://qwen3lm.com/qwen3-flashattention2-inference-guide/)）。

Rust 把 Flash Attention 做成可选，没有兼容 GPU 的机器仍能跑；硬件支持才吃到加速。

## 跨语言集成

Rust 分类引擎 + **Go FFI**。云原生部署是 Go 形的；热路径不是。

### 为什么推理用 Rust

- 近 C 的性能、零成本抽象、低延迟
- 编译期内存安全
- 所有权系统 + Rayon：无 data race 的并行
- 没有 GC 停顿

Candle 踩在这些 Rust 性质上，API 仍是 ML 形。

### 为什么要 Go FFI

Go 占着云原生控制面。FFI 是桥：

- **Envoy**：semantic router 当 Go 里的 [Envoy `ext_proc` filter](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)；FFI 让 filter 调 Rust 分类，不用重写 Envoy 层
- **Kubernetes operators**：通常是 Go / controller-runtime；把分类嵌进去，少一跳网络
- **Service meshes**：Istio、Linkerd、Consul——Go；ML 分类还不拆 mesh 控制面
- **API gateways**：Kong、Tyk 和其他 Go 组件；在网关做语义路由，不必再加一只微服务

### 部署弹性

- **Embedded**：Go 经 CGO 链上 Rust 库——延迟更低、部署更简单
- **Process isolation**：分类单独进程（gRPC 或 Unix sockets）
- **Mixed**：网络和编排用 Go，ML 推理用 Rust

主路由逻辑、配置、缓存：**Go**。吃算力的分类：**Rust**。FFI 边界干净。

## 性能特征（他们的定性表）

- **单任务对多任务**：单任务 LoRA 几乎不赚。同一输入上好几次分类，才明显赢。
- **长上下文**：Qwen3-Embedding 能对 **32K** 文档做路由而不截断（超过 ModernBERT 的 **8K**）。兼容 GPU 上开 Flash Attention 2：优势随上下文长。
- **多语**：ModernBERT 训练数据薄的语言，现在能路由。
- **高并发**：`OnceLock` 去掉锁竞争；分类吞吐可以随 CPU 核涨（上面那次测试里他们的主张）。
- **GPU**：Flash Attention 2 在 attention 上 **3–4×** 是 **引用带**，长序列更明显。

## 未来方向（点名，不是这篇交付）

- 更多 embedding 模型，走 `CoreModel` trait
- Candle 有了之后的 Flash Attention 3
- 量化（4-bit、8-bit）
- 领域路由的自定义 LoRA adapter
- Python、Java、C++ 的 FFI

地基：新研究进来，不必改架构。FFI 当稳定接口，Rust 可以在现有 Go 部署下面自己演化。

## 资源

- [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- [Candle](https://github.com/huggingface/candle)
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
