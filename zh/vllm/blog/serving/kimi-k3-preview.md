---
source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Kimi K3 开源日前的 preview：KDA prefix cache 才是硬骨头

英文对照：[en/vllm/blog/serving/kimi-k3-preview.md](../../../../en/vllm/blog/serving/kimi-k3-preview.md)  
原文：https://vllm.ai/blog/2026-07-22-kimi-k3-preview  
2026-07-22。vLLM Team。权重按当时计划 **2026-07-27** 放；落地见 [kimi-k3.md](kimi-k3.md)。公告 [Moonshot](https://www.kimi.com/blog/kimi-k3)；[推文](https://x.com/Kimi_Moonshot/status/2077830229968683203)。选中的 trusted partners，要 Moonshot **和** vLLM/Inferact **双方批准**，用的是准备开源的同一份代码。KDA prefix cache：Moonshot 贡献实现，随权重一起放；设计另文。页上原话：**vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.** 跳过社交预览图。本地图版权仍归原站。

上周 Moonshot 推出 Kimi K3：2.8T、原生视觉、1M 上下文、Kimi Delta Attention (KDA)、Attention Residuals (AttnRes)、高度稀疏 MoE。开源社区兴奋的是开权重在追专有模型。权重当时定在 2026-07-27 放。这期间 vLLM、Moonshot、NVIDIA、AMD 和更宽的社区在收最后的集成和校验，好让社区 **day 0** 就能 serve。

这篇是 preview，性能优化还在走。核心模型路径、KDA-aware prefix caching、多模态集成、tool-calling parsers、硬件专项优化，已经成形。

## TL;DR

- **Day-0 开源 serving：** 模型实现、Docker、deployment recipes、production validation，对着权重发布日准备。
- **新的 hybrid 架构：** KDA 主导的线性注意力 + 周期性 full-attention、跨深度 AttnRes、Stable LatentMoE、原生视觉。
- **Prefix caching 要改核心：** 物理 KDA state-block size 和 prefix-match 粒度拆开，才能在不大块存 recurrent state 的前提下拿到有用的 partial prefix-cache hit。
- **整栈 kernel：** FlashKDA、fused KDA decode、fused KDA projections 和 convolution、fused AttnRes、重写 MLA、SiTU 的 MXFP4 MoE、优化过的 expert routing。
- **NVIDIA 和 AMD：** NVIDIA kernel 最后调参；AMD 已有 FlyDSL MoE，更宽的校验还在走。

## Kimi K3 at a Glance

K3 不是更大的 K2。serving 问题同时在几维上变了。

| 属性 | Kimi K3 | serving 含义 |
| --- | --- | --- |
| **Model scale** | **2.8T** | 大规模 expert parallelism，高带宽加速器域 |
| **Context** | **1M tokens** | cache 容量、prefix reuse、chunked prefill、P/D 拆分都是一等公民 |
| **Attention** | **Hybrid KDA + full attention** | recurrent state cache 和 paged KV 必须停在同一逻辑前缀 |
| **Depth** | **AttnRes** | 跨层读写，要专门 kernel |
| **MoE** | **896 routed，每 token 16 active，加 shared** | routing、dispatch、load balance、MoE kernel 决定端到端 |
| **Quant** | 发布配置 **MXFP4** | 高效 FP4 MoE，还要吃 K3 的 **SiTU** |
| **Multimodality** | 原生视觉 + vision tower | 多模态预处理（当时 image-only）和稳的 vision parallelism |

对推理系统，每一项都把成本挪到新地方。KDA 不必为每个过去 token 留常规 KV，但引入大块 recurrent state。AttnRes 不再只靠一条均匀累加的 residual stream，却多了跨层内存流量。极端稀疏避免每 token 激活全部 2.8T，却把 routing 和通信的赌注抬高。vLLM 的活是让这些在同一套熟悉的 serving API 后面一起转。

## 跨几代 Kimi 的合作

K3 接着 Moonshot 和 vLLM 社区的长合作。

- [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html)：Moonshot 讲内部大规模用 vLLM，以及 vLLM + Mooncake 的 P/D 拆分。
- [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/)：K2 训练和推理，严 SLO 下的在线流量和 RL 负载。
- vLLM 做过 K2、K2-Thinking、K2.5、Kimi Linear 等的 day-0。
- 深合作：[Kimi K2 tool-calling](https://vllm.ai/blog/Kimi-K2-Accuracy)（正确性）、[CUDA debugging](https://vllm.ai/blog/improved-cuda-debugging)、[decode context parallelism](https://github.com/vllm-project/vllm/pull/23734)、Mooncake P/D、大规模性能校验。K2.5 也进过公开 [InferenceX](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07)。学习笔记：[kimi-k2-accuracy.md](kimi-k2-accuracy.md)、[cuda-debugging-source.md](../dev/cuda-debugging-source.md)、[dcp.md](../features/dcp.md)、[mooncake.md](../features/mooncake.md)。

这段历史要紧。Day-0 很少是公告之后写一个 PR。模型组和推理组早早共享架构、在真实 checkpoint 和现实并行下测、找出 serving 引擎的缺口、把 launch 之后仍有用的改进上游。页上原话：**vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.**

下面钻进当时最有意思的技术硬骨头。

## 最难的一块：KDA 的 Prefix Caching

常规 full attention 和 KDA，记前缀的方式完全不同。

Full attention：前缀是 per-token K/V。vLLM 把它们放进 paged blocks，对完整 token block 做 hash，另一条请求可以复用匹配的 block 序列。

KDA 是 recurrent。每层推进一个矩阵状 recurrent state，外加短 convolution state。要从缓存前缀恢复，引擎需要 **恰好在前缀边界** 的 KDA state。从更早的 state replay 到边界，会把 prefix caching 的好处抹掉大半。

![conventional attention vs KDA cached prefixes](../../../../assets/vllm/blog/serving/kimi-k3-preview/01-kda-prefix-state.png)

直白做法——在每个小 attention-cache 边界都存一份 KDA state——太贵。一份 KDA state 比一个普通 token 的 KV 大得多，所以实现用较大的物理 state block 摊成本。在这次改动之前，物理 block size 也约束 prefix-cache hit 能落在哪。几千 token 一块时，两条请求几乎共享整段 prompt，仍可能 miss：共同边界没填满同一物理块。

新设计把原先绑在一起走的三件事拆开：

- **Physical block size：** GPU 上怎么分配 KDA state 和 full-attention KV。
- **Scheduler alignment：** 执行必须停在哪，好让所有 cache group 一致。
- **Prefix-match unit：** 共享前缀被 hash、被匹配的更细 token 间隔。

![fine-grained prefix matching inside a larger physical KDA state block](../../../../assets/vllm/blog/serving/kimi-k3-preview/02-fine-grained-prefix-cache.png)

这样可以在较大的物理 state block **里面** 的细粒度边界登记一份有效 KDA state。后来的请求打中这块 partial block，先把缓存 state **拷** 到私有目的地，再往前走。Copy-on-write 保住共享前缀，新请求也能安全续生成。

实现还处理了容易漏的细节：

- Scheduler 停在对的 block 和 hash 边界，登记的 recurrent state 真对应宣称的 token 前缀。
- Full-attention 和 KDA cache group 对同一份 `num_computed_tokens` 达成一致，尽管物理 block size 不同。
- Partial cache 用链式细粒度 hash，边界标识 **整段** 前缀，不只是尾巴。
- 同一步 reuse 推迟到 state copy 安全，避免登记和扩展之间的 race。
- Cache transfer 和拆分的 P/D 路径，能把同一逻辑前缀带过 worker。

工作由 K3 和许多 hybrid attention 模型推动，但是 **vLLM 核心基础设施**，不是模型专用捷径。vLLM 和 Moonshot 在设计上合作很深。两队另发文讲设计、不变量、benchmark。落地见 [kimi-k3.md](kimi-k3.md)。

## 性能工作：拆掉新瓶颈

当时进度可以收成这张表：

| 区域 | 当时状态 |
| --- | --- |
| **Model and configuration** | 语言和视觉定义已接入；硬件路径不同处 **NVIDIA / AMD 分开实现** |
| **Optimized MLA for native PD** | 手工 kernel fusion，prefill/decode 分路径。Gate projection 和 attention 并行；decode 多流，prefill fused epilogue——对着 PD 拆分优化 |
| **Serving semantics** | chat 渲染、tokenizer、streaming parse、tool calls、reasoning、structured-output 已实现，**最终端到端校验中** |
| **KDA prefill** | FlashKDA 和 Triton 已接入；最终后端选择和数值校验 **进行中** |
| **KDA decode** | 覆盖 convolution、recurrent 更新、gating、normalization 的 fused **NVIDIA** decode kernel 已接入，保留可移植 fallback |
| **Prefix caching** | hybrid full-attention + recurrent-state 的细粒度 partial hit 已接入；拆分和 offload **校验中** |
| **AttnRes** | Triton 和 **NVIDIA** kernel 已接入，支持形状上融合 residual add 和 output RMSNorm |
| **MoE** | **SiTU** 接到 **MXFP4 TRTLLM-Gen** 和 **DeepGEMM**；优化过的 grouped top-k routing。**AMD** 用 FlyDSL **MLIR**，硬件调过的 **A16W4/A8W4** fused op 和 **SiTU** |
| **Production stack** | 非拆分 serving 能跑；Dynamo + vLLM + Mooncake 拆分、EP、vendor verification 在 **最终校验环** |

K3 改了热路径，优化不只 attention kernel。

### KDA prefill and decode

Prefill 接入 FlashKDA 和 Flash Linear Attention (FLA)。核心 recurrence 周围，vLLM 融合 input projections 和 causal convolution，一次 gather 初始 recurrent states。

Decode 在支持的架构和形状上走 fused NVIDIA kernel。短 convolution、KDA state 更新、output gate、normalization 不再每 token 各 launch 一次。K3 有很多 KDA 层；每层一点 launch 或内存惩罚，很快变成大的 TPOT 惩罚。

### Attention Residuals

AttnRes 从更早 layer block 写下的表示里取，不只靠一条均匀累加的 residual stream。朴素实现会在 **93 层** 网络里到处多读写、reduction、normalization launch。

Release branch 有 Triton 和 NVIDIA kernel：支持的情况下融合 residual update、AttnRes mixing、output RMSNorm。Sequence-parallel 也把 attention-residual 流量按 rank 切。早期 kernel 结果令人鼓舞；端到端收益还在按 prefill 长度和并行配置量。

### Optimized MLA module for native PD disaggregation

K3 每四层仍用 MLA。上一模型大量靠 `torch.compile` 自定义融合把小 kernel 合成 fused kernel，启动慢，仍有许多没融上。这次新 MLA 模块手工融合。Prefill 和 decode 的 launch 顺序不同，所以两条路径、不同融合图案，专门对着 PD 拆分。K3 还引入可与主 attention 并行的 gate projection。Decode 可选多流；prefill 里多流 overlap 不优，就把 elementwise multiply 和 sigmoid 融进 gate-projection epilogue。

### MXFP4 MoE

发布配置：MXFP4 权重 + SiTU。此前 MXFP4 TRTLLM-Gen 不支持 SiTU，会落到更慢的实现。现在把 K3 的 SiTU 参数映射进优化过的 FP4 expert 路径；大 token-by-top-k launch grid 安全 chunk。

已在 **16-GPU DP16+EP16** 上校验：所有 rank 选中优化 MXFP4 backend，过正确性检查。

AMD：K3 MoE 走 FlyDSL 的 MLIR Python kernel stack，含硬件调过的 A16W4/A8W4 quantized fused operators 和 SiTU，建在 FlyDSL 模块化抽象上。

## 开源日能期待什么

计划中的 day-0 包：

- vLLM 模型、parser、cache、kernel 集成；
- 初始开源 Docker；
- 校验过的 NVIDIA launch recipes；
- 初始 AMD 路径（FlyDSL MoE），更多 ROCm 调参随后；
- 多模态、tool-use、reasoning、structured-output 例子；
- 初始性能数字。

Trusted deployment partners 已在 Moonshot 和 vLLM/Inferact 的双重批准下跑 release candidate。真实生产反馈，又不把预发布权重广散。也给完整 serving 系统——前端语义、batching、cache transfer、EP、可观测、失败处理——而不只是孤立 kernel，一次试跑的机会。落地数字见 [kimi-k3.md](kimi-k3.md)。

## 致谢

K3 day-0 是模型厂商、推理引擎、硬件社区的合力。

- **Moonshot：** 做出 K3；权重前共享架构；初始模型集成和 KDA prefix-caching；正确性和生产校验上密切合作。
- **Inferact：** 接到 vLLM；扩展核心 cache manager 做 partial hybrid prefix hit；serving 语义和多模态；deployment recipes；端到端性能。
- **NVIDIA：** KDA decode 和 AttnRes kernel、MXFP4 MoE、整盘性能。
- **AMD：** 初始 day-0 ROCm，并继续把 K3 铺到更多 AMD GPU。
- 更宽的开源社区：期待、测试、反馈。页上说期待把权重和推理引擎支持交到手里。

## 还有一件事：为什么公告和开源要拆开

K3 还有一套发布流程，页上希望更多模型厂商考虑：**先公告模型，再放权重和推理引擎支持。**

vLLM 提出拆开，Moonshot 同意并执行。理由实际。前沿模型公告有躲不开的 last-mile 不确定性。模型组同时在稳自己的产品、API、评测、安全、文档、商业发布。如果开源权重和开源支持必须落在 **同一时刻**，像 vLLM 这样的社区项目会被移动的 deadline 拖垮。

拆开时间线，两边合同更好：

1. 模型厂商可以专心产品发布，冻结最终 checkpoint、配置、tokenizer、serving 语义。
2. 开源推理引擎组拿到稳定的集成窗口：正确性测试、性能调参、Docker、recipe 校验。
3. 社区拿到公开、有界的预期，而不是含糊的 “coming soon”。

拆开不是从 day-0 撤退。它是对着用户真正会下载的那份产物，更可持续地交付 day-0。页上鼓励更多模型厂商跟。
