---
source: https://vllm.ai/blog/2025-01-27-v1-alpha-release
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# vLLM V1：把核心拆开重装（alpha）

英文对照：[en/vllm/blog/architecture/v1-alpha.md](../../../../en/vllm/blog/architecture/v1-alpha.md)  
原文：https://vllm.ai/blog/2025-01-27-v1-alpha-release  
2025-01-27。署名 **vLLM Team**。学习译文，不是官方译本。这是 **alpha 发布稿**。文中「还不支持 LoRA / spec decode / PP / Prometheus / logprobs / structured decoding、没有 encoder-decoder、没有 Mamba/Jamba、没有 embedding、只要 Ampere+ NVIDIA」是**当时的缺口**，不是今天的功能表。V1 后来成了默认引擎。结构细节以 [Anatomy](anatomy.md) 为准。把 CPU 从 GPU 路上挪开的上一轮成绩单是 [v0.6.0](../../performance/v0.6-throughput.md)。

他们宣布 **vLLM V1 的 alpha**：按过去一年半的教训，回头改关键设计、把功能收拢、把代码写简单，好让系统更灵活、更能长。当时已经自称 **SOTA**，后面还要再叠优化。开启：`VLLM_USE_V1=1`，**现有 API 不用改**。他们说测几周、收反馈，再把 V1 切成默认。

营销 Logo 不收录。本地图（原文版权仍归原站；学习对照用）：

![v1 server architecture](../../../../assets/vllm/blog/architecture/v1-alpha/01-v1_server_architecture.png)
![v1 scheduling](../../../../assets/vllm/blog/architecture/v1-alpha/02-v1_scheduling.png)
![v1 prefix caching](../../../../assets/vllm/blog/architecture/v1-alpha/03-v1_prefix_caching.png)
![v1 tp architecture](../../../../assets/vllm/blog/architecture/v1-alpha/04-v1_tp_architecture.png)
![persistent batch](../../../../assets/vllm/blog/architecture/v1-alpha/05-persistent_batch.png)
![torch compile cuda graph](../../../../assets/vllm/blog/architecture/v1-alpha/06-torch_compile_cuda_graph.png)
![v1 llama](../../../../assets/vllm/blog/architecture/v1-alpha/07-v1_llama.png)
![v1 qwen2vl](../../../../assets/vllm/blog/architecture/v1-alpha/08-v1_qwen2vl.png)

## Why vLLM V1?

### Learning from vLLM V0

一年半里，vLLM 横向很成功：模型、特性、硬件铺得很开。纵向却很难把优化叠在同一条栈上——功能各自生长，合在一起既不干净、也不容易。技术债堆在地基里，于是他们回头拆地基。

### Goals of V1

- 代码简单、模块化、好改。
- 近乎零的 CPU 开销。
- 把关键优化收进**同一套**架构，而不是互斥的插件。
- 零配置：好东西默认开。

### Scope of V1

重做：调度器、KV cache manager、worker、sampler、API server。

仍与 V0 共用：模型实现、GPU kernel、分布式控制面、一堆工具函数。赌注是：覆盖面和稳定性留给 V0，烧掉 CPU 的那条环自己重写。

## What’s New in vLLM V1?

### 1. Optimized Execution Loop & API Server

vLLM 既是 continuous batching 引擎，也是 OpenAI 兼容的 API server。两次 GPU 前向之间，CPU 管着请求的命：跑 API、调度、准备输入、detokenize、流式回包。GPU 越快，模型前向越短，这些 CPU 活就越丢脸。**Llama-8B on NVIDIA H100**，一步 GPU 可以大约 **5 ms**。这个量级上，tokenization、调度、detokenize、流式都会变成瓶颈。

[v0.6.0](https://vllm.ai/blog/2024-09-05-perf-update) 已经把 API server 用 **ZeroMQ** 拆到另一进程，让 HTTP 路径和 AsyncLLM 重叠。V1 再往里拆：独立的 `EngineCore` 执行环只跑**调度器 + model executor**；tokenize、多模态预处理、detokenize、流式，与这条核心环重叠，吞吐才能顶上去。

### 2. Simple & Flexible Scheduler

不再把 Prefill 和 Decode 当成两种物种。用户给的 prompt token 和模型吐出的 token 同一套账。每一步是一份字典：`{request_id: num_tokens}`——这一步每个请求处理多少 token。这张表够通用：chunked prefill、prefix cache、投机解码都能落上去。chunked prefill 无非是：固定 token 预算，动态决定分给谁多少（上图）。

### 3. Zero-Overhead Prefix Caching

仍是 **hash** + **LRU**，和 V0 同一套想法。

V0 的病：开了 prefix cache，命中率低时 CPU 开销会让吞吐掉下去，所以**默认关**。V1 把驱逐做成近似常数时间，少造 Python 对象。即便命中率是 0%，prefix cache 也几乎不伤吞吐。

实验里：命中率 **0%** 时吞吐掉不到 **1%**；命中率高时可以翻几倍。税够小，于是 **默认开**。

### 4. Clean Architecture for Tensor-Parallel Inference

V0 为了少广播输入，把调度器和 **Worker 0** 塞进同一进程。IPC 便宜了，架构却**不对称**，复杂度上去。V1 在 worker 侧缓存请求状态，每步只传 **diff**。IPC 小到调度器与 Worker 0 可以分进程，对称回来。单卡多卡同一套 worker 逻辑；分布式的大部分细节从 worker 眼里被抽象掉。

### 5. Efficient Input Preparation

V0 每步重建输入张量和 metadata，又是一笔 CPU 账。V1 用 **Persistent Batch**（他们指向 [LMDeploy](https://github.com/InternLM/lmdeploy)）：缓存张量，每步只打补丁；更新尽量走 **Numpy**，不走纯 Python。

### 6. torch.compile and Piecewise CUDA Graphs

少写手搓 kernel，也能覆盖很多模型：V1 靠 vLLM 的 `torch.compile` 集成自动优化。整图抓不住的缝，用 **piecewise CUDA graphs** 补。当时说会另开博客；后来的笔记是 [torch-compile.md](torch-compile.md)。

### 7. Enhanced Support for Multimodal LLMs

多模态 LLM 当一等公民。三件事：

1. **非阻塞预处理 + 缓存。** JPG/PNG 变成像素张量、裁切、变换：这活若坐在 worker 上，GPU 会闲着等。V1 搬到单独进程，并缓存处理结果——同一张图不必再解码一遍。
2. **图像进 prefix cache。** 除了 token ID 的 hash，再用 **image hash** 标识图像对应的 KV。多轮带图的对话可以复用。
3. **encoder cache**，好让文本 Prefill 切块。V0 里图像和文本必须同一步：decoder 的 `<img>` token 依赖 vision embedding，而 embedding 用完就丢。V1 把 vision embedding 暂存，文本 Prefill 可以跨步切块，不必每步重算视觉。

### 8. FlashAttention 3

最后一块拼图：V1 的 batch 里 Prefill 和 Decode 坐在一起，需要一颗既灵活又快的 attention。[FlashAttention 3](https://arxiv.org/abs/2407.08608) 是当时点名的那颗 kernel：功能宽，用例也快。

## Performance

相对 V0（**未开** multi-step scheduling），吞吐最高大约 **1.7×**。两边 kernel 几乎相同，差的是整条栈上的 CPU。VLM（例如 Qwen2-VL）上跳得更大——V1 把视觉当一等公民。

### Text Models: Llama 3.1 8B & Llama 3.3 70B

ShareGPT。V1 延迟更低，尤其在高 QPS——吞吐先上去，排队才松。kernel 几乎一样，所以差的是架构（少交 CPU 税）。

### Vision-language Models: Qwen2-VL

[VisionArena](https://arxiv.org/abs/2412.08687)。加速比文本那组更明显：预处理搬走了，多模态调度更灵活。多模态的 prefix cache 在 V1 里已经是原生能力；他们**故意没画**那组命中率曲线。

### Looking Forward

这些数字写成起点。新地基是为了让后面的功能便宜地长出来。他们说未来几周还会再发增强。

## Limitations & Future Work

下面全部是 **alpha、2025 年 1 月** 的话。不要当成 2026 年的支持矩阵。

### Model Support

解码器 Transformer（Llama 一类）、Mixtral 一类 MoE、若干 VLM（Qwen2-VL）。量化方法都可用。**当时没有：** encoder-decoder（例如当时的 multimodal Llama 3.2）、Mamba 系（Jamba）、embedding 模型。更细的名单当时指向文档 [supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)。

### Feature Limitations

缺：logprobs、prompt logprobs 采样参数、pipeline parallelism、structured decoding、投机解码、Prometheus metrics、LoRA。文末已经点名谁在补这几块，并说还会往 V1 引擎里加新优化。

### Hardware Support

只要 Ampere 及以后的 **NVIDIA**。TPU 等其他后端当时在做。

不设 `VLLM_USE_V1=1` 就继续走 V0，向后兼容。

## How to Get Started

1. `pip install vllm --upgrade`
2. `export VLLM_USE_V1=1`
3. [Python API](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/basic.py)（离线 `basic.py`）或 OpenAI 兼容服务：`vllm serve <model-name>`。现有 API 不用改。

当时请人试用、给反馈。

## Acknowledgment

设计受惠于 [LightLLM](https://github.com/ModelTC/lightllm)、[LMDeploy](https://github.com/InternLM/lmdeploy)、[SGLang](https://github.com/sgl-project/sglang)、[TGI](https://github.com/huggingface/text-generation-inference)、[TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。

不完全名单（角色以 alpha 稿为准）：

- 主力：UC Berkeley、Neural Magic（后并入 Red Hat）、Anyscale、Roblox。
- [Woosuk Kwon](https://github.com/WoosukKwon)：立项；调度器与 model runner。
- [Robert Shaw](https://github.com/robertgshaw2-redhat)：执行环与 API server。
- [Cody Yu](https://github.com/comaniac)：文本与图像的 prefix cache。
- [Roger Wang](https://github.com/ywang96)：V1 的 MLLM 支持。
- [Kaichao You](https://github.com/youkaichao)：`torch.compile` 与 piecewise CUDA graphs。
- [Tyler Michael Smith](https://github.com/tlrmchlsmth)：Python multiprocessing 上的 TP。
- [Rui Qiao](https://github.com/ruisearch42)：Ray 上的 TP；当时正在做 PP。
- [Lucas Wilkinson](https://github.com/LucasWilkinson)：FlashAttention 3。
- [Alexander Matveev](https://github.com/alexm-redhat)：多模态预处理器；当时正在做 TPU。
- [Sourashis Roy](https://github.com/sroy745)：sampler 里的 logit penalties。
- [Cyrus Leung](https://github.com/DarkLight1337)：MLLM 输入处理重构，并接到 V1。
- [Russell Bryant](https://github.com/russellb)：多进程相关问题。
- [Nick Hill](https://github.com/njhill)：engine loop 与 API server。
- [Ricky Xu](https://github.com/rickyyx)、[Chen Zhang](https://github.com/heheda12345)：KV cache manager 重构。
- [Jie Li](https://github.com/jeejeelee)、[Michael Goin](https://github.com/mgoin)：MLLM 支持与优化。
- [Aaron Pham](https://github.com/aarnphm)：当时正在做 structured decoding。
- [Varun Sundar Rabindranath](https://github.com/varun-sundar-rabindranath)：当时正在做 multi-LoRA。
- [Andrew Feldman](https://github.com/afeldman-nm)：当时正在做 logprobs / prompt logprobs。
- [Lily Liu](https://github.com/LiuXiaoxuanPKU)：当时正在做投机解码。
- [Kuntai Du](https://github.com/KuntaiDu)：当时正在做 Prefill 分离与 KV 传输。
- [Simon Mo](https://github.com/simon-mo)、[Zhuohan Li](https://github.com/zhuohan123)：V1 系统设计。

读这一篇，是为了看见 V1 想修的病：CPU 抢 GPU 的时间、功能互斥、prefix cache 不敢默认开。Anatomy 是这座城后来的地图。
