---
source: https://vllm.ai/blog/2026-07-22-kimi-k3-preview
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Kimi K3 preview：权重还没到，cache 先改了

英文对照：[en/vllm/blog/serving/kimi-k3-preview.md](../../../../en/vllm/blog/serving/kimi-k3-preview.md)  
原文：https://vllm.ai/blog/2026-07-22-kimi-k3-preview  
2026-07-22。署名 **vLLM Team**。权重计划 **2026-07-27**。落地数字见 [kimi-k3.md](kimi-k3.md)。tool-calling 握手：[kimi-k2-accuracy.md](kimi-k2-accuracy.md)。合作名单里点名的 CUDA 调试：[cuda-debugging-source.md](../architecture/cuda-debugging-source.md)。DCP：[dcp.md](../performance/dcp.md)。缓存 / P/D：[mooncake.md](mooncake.md)。优化还在飞。**引擎骨架没换**。**不是发布指南**——那篇是 [kimi-k3.md](kimi-k3.md)。

上周 Moonshot AI [介绍 Kimi K3](https://www.kimi.com/blog/kimi-k3)：2.8T、原生视觉、1M 上下文、Kimi Delta Attention（KDA）、Attention Residuals（AttnRes）、极稀疏 MoE。开源权重计划 2026-07-27。vLLM、Moonshot、NVIDIA、AMD 和社区在收尾，好让开源社区 Day-0 就能伺候。

这篇是 **预告**。核心模型路径、KDA 感知的前缀缓存、多模态、tool-calling parser、硬件专用活，已经成形。获双方批准的合作伙伴（Moonshot + vLLM/Inferact）已经用准备开源的同一份代码做部署验证。

宣布博说过：KDA 给常规前缀缓存出了新题。Moonshot 把对应实现贡献进 vLLM，跟权重一起发。设计细写后来成了 [kimi-k3.md](kimi-k3.md) 加这篇 preview。

**Figure（social preview，未收录）：** 原文 `/assets/figures/2026-07-22-kimi-k3-preview/social-preview.png`。

本地图（原文版权仍归原站；学习对照用）：

![kda prefix state](../../../../assets/vllm/blog/serving/kimi-k3-preview/01-kda-prefix-state.png)

![fine grained prefix cache](../../../../assets/vllm/blog/serving/kimi-k3-preview/02-fine-grained-prefix-cache.png)

## TL;DR

- **Day-0 开源 serving：** 模型实现、Docker、部署配方、生产验证，等权重。
- **新 hybrid 架构：** KDA 为主的线性 attention 夹周期性满 attention，沿深度的 AttnRes，Stable LatentMoE，原生视觉。
- **前缀缓存要动 core：** 物理 KDA state 块大小和前缀匹配粒度拆开——不必在每个小 attention 块上都存一份 recurrent，也能吃到有用的部分前缀命中。
- **Kernel：** FlashKDA、fused KDA Decode、fused KDA 投影和卷积、fused AttnRes、重写的 MLA、接通 SiTU 的 MXFP4 MoE、优化过的专家路由。
- **NVIDIA 和 AMD：** NVIDIA kernel 在做最后调优；AMD 初版 FlyDSL MoE 已经在，更宽的验证还在走。

## Kimi K3 一眼

这不是更大的 Kimi K2。Serving 问题一次变了好几维。

| 属性 | Kimi K3 配置 | Serving 含义 |
| --- | --- | --- |
| **规模** | **2.8T** | 大规模 expert parallelism，高带宽加速域 |
| **上下文** | **1M tokens** | cache 容量、前缀复用、chunked Prefill、Prefill/Decode 分离变成一等事 |
| **Attention** | **Hybrid KDA + 满 attention** | Recurrent 状态 cache 和 paged KV 必须在 **同一** 逻辑前缀上往前走 |
| **深度** | **Attention Residual** | 跨层读写要专用 kernel |
| **MoE** | **896 routed，每 token 激活 16，外加 shared** | 路由、dispatch、均衡、MoE kernel 坐在端到端上 |
| **量化** | **发布配置里的 MXFP4 权重** | 高效 FP4 MoE，还要接 Kimi K3 的 **SiTU** |
| **多模态** | **原生视觉 + vision tower** | 多模态预处理（当时图）和靠谱的视觉并行策略 |

每一项都把代价挪到新地方。KDA 不必给每个过去 token 留一对常规 KV，却引进很大一块 recurrent。AttnRes 松开单一残差流，却多了跨层内存交通。极稀疏 MoE 不必每 token 激活全部 2.8T，路由和通信的赌注却变大。vLLM 的活，是让这些在一张熟的 serving API 后面一起转。

## 几代 Kimi 攒下来的合作

- [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html)：Moonshot 工程师讲内部大规模用 vLLM，以及 vLLM + Mooncake 的 Prefill/Decode 分离。
- 后来在 [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/) 讲 Kimi K2 训练和推理——在线流量的严格 SLO，还有 RL。
- Kimi K2、Kimi K2-Thinking、Kimi K2.5、Kimi Linear 等的 Day-0 伙伴。
- 深的技术合作：[Kimi K2 tool-calling](kimi-k2-accuracy.md)、[CUDA 调试](../architecture/cuda-debugging-source.md)、[decode context parallelism](https://github.com/vllm-project/vllm/pull/23734)（[dcp.md](../performance/dcp.md)）、Mooncake 的 P/D、大规模性能验证。Kimi K2.5 也出现在公开 [InferenceX](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07)。

Day-0 很少是宣布之后写一个 PR。架构细节要早给，真实 checkpoint 要在真实并行下测，引擎缺口要找出来，上游改进要在这一发之后还值钱。

## 最难的一块：KDA 的前缀缓存

满 attention 和 KDA 记住前缀的方式完全不同。

满 attention：前缀是逐 token 的 key / value。vLLM 把它们放进 paged 块，给完整 token 块做哈希，匹配上的块序列可以复用。

KDA 是 recurrent。不是每个 token 一对常规 KV，每一层 KDA 推进一块矩阵式 recurrent，外加短卷积状态。要从缓存前缀接着走，引擎需要 **恰好落在前缀边界** 的 KDA 状态。把更早的状态重放到那个边界，前缀缓存的好处就没了。

**Figure。** 常规 attention 和 KDA 各自怎么表示缓存前缀。

直白解法——在每个小 attention-cache 边界都存一份 KDA 状态——太贵。一份 KDA 状态比一个普通 token 的 KV 大得多，实现上用相对大的物理 state 块来摊存储。在这次工作之前，物理块大小也卡住了前缀命中能落在哪。几千 token 一块时，两份请求几乎共用整段 prompt，仍可能 miss——共同边界没填满同一块物理块。

新设计把三件曾经绑在一起的事拆开：

- **物理块大小：** GPU 上 KDA 状态和满 attention KV 怎么分配。
- **调度对齐：** 执行必须停在哪，好让所有 cache group 一致。
- **前缀匹配单位：** 共享前缀被哈希、可能命中的更细 token 间隔。

**Figure。** 大物理 KDA state 块里的细粒度前缀匹配。

vLLM 可以在大物理块 **内部** 的细粒度边界登记一份有效的 KDA 状态。后来的请求命中这块残尾：先拷到私有目的地，再往前延伸。Copy-on-write 保住共享前缀；新请求可以安全接着生成。

容易漏的细节：

- 调度停在对的块和哈希边界，登记的 recurrent 才真对应宣称的 token 前缀。
- 满 attention 和 KDA 的 cache group 同意同一个 `num_computed_tokens`，哪怕物理块大小不同。
- 部分 cache 条目用链式细粒度哈希，边界标识的是 **整段** 前缀，不只是尾巴。
- 同一步的复用等到状态拷贝安全才发生——登记和延伸之间不抢。
- Cache 搬运和分离 Prefill/Decode 可以把同一逻辑前缀带到别的 worker。

动机来自 Kimi K3 和其他 hybrid attention 模型，但这是 **core 基础设施**，不是模型私房。设计、不变量、基准的细写，现在在 [kimi-k3.md](kimi-k3.md)。

## 性能工作：把新瓶颈拆掉

| 区域 | 当时进度 |
| --- | --- |
| **模型和配置** | 语言和视觉定义已接上；硬件路径不同处，**NVIDIA** / **AMD** 分开实现 |
| **给原生 P/D 优化的 MLA** | 手写融合，Prefill/Decode 分路径。Gate 投影和 attention 并行；Decode 可多流；Prefill fused epilogue |
| **Serving 语义** | Chat 渲染、tokenizer、流式解析、tool call、reasoning、structured-output——**最后一轮端到端验证** |
| **KDA Prefill** | FlashKDA 和 Triton 已接；最终 backend 选择和数值验证 **进行中** |
| **KDA Decode** | Fused **NVIDIA** Decode：卷积、recurrent KDA 更新、gate、归一化；可移植回退还在 |
| **前缀缓存** | Hybrid 满 attention + recurrent 的细粒度部分命中已接；分离和 offload **在验** |
| **Attention Residuals** | Triton 和 **NVIDIA** kernel；支持的形状上融合残差加和输出 RMSNorm |
| **MoE** | **SiTU** 接到 **MXFP4 TRTLLM-Gen** 和 **DeepGEMM**；优化过的 grouped top-k。**AMD** FlyDSL **MLIR**：硬件调过的 **A16W4/A8W4** fused op + **SiTU** |
| **生产栈** | 非分离 serving 能跑；Dynamo + vLLM + Mooncake 分离、expert parallelism、厂商验证在 **最后验证环** |

Kimi K3 改了热路径，所以优化不只是 attention kernel。

### KDA Prefill 和 Decode

Prefill 接 FlashKDA 和 Flash Linear Attention（FLA）。核心 recurrence 周围：输入投影和因果卷积融合；一次操作收集初始 recurrent。

Decode：支持的架构和形状上走 fused NVIDIA kernel。卷积、KDA 状态更新、输出 gate、归一化一次做完，而不是每个生成 token 拆成好几次 launch。Kimi K3 的 KDA 层很多；每层一点点 launch 或内存惩罚，会变成很大的 **TPOT** 惩罚。

### Attention Residuals

AttnRes 从更早层块写下的表示里取，不只依赖一条均匀累加的残差流。朴素实现：整张 **93 层** 网上多出读写、归约、归一化 launch。

发布分支：Triton，加上在支持形状上融合残差更新、AttnRes 混合、输出 RMSNorm 的 NVIDIA kernel。Sequence-parallel 把 attention-residual 交通按 rank 切开。Kernel 级早期结果看好；端到端还在不同 Prefill 长度和并行配置上量。

### 给原生 P/D 优化的 MLA

Kimi K3 每四层仍用 MLA。以前 vLLM 很依赖 `torch.compile` 的自定义融合：启动慢，许多 kernel 仍没焊上。这发：新 MLA 模块，**手写** 融合。Prefill 和 Decode 的 kernel launch 顺序不同，于是两条代码、两套融合，专门对着 P/D。Kimi K3 还加了可以和主 attention 并行的 gate 投影：Decode 可选多流；Prefill 上多流重叠不划算，就把逐元素乘和 sigmoid 焊进 gate 投影的 epilogue。

### MXFP4 MoE

发布配置：MXFP4 权重 + SiTU 激活。此前 MXFP4 TRTLLM-Gen 路径不认 SiTU，会掉到更慢的实现。现在把 Kimi K3 的 SiTU 参数映进优化过的 FP4 expert 路径，大 token-by-top-k launch grid 也安全切开。

在 **16 GPU DP16+EP16** 上验过：所有 rank 都选了优化过的 MXFP4 backend，正确性过。

AMD：Kimi K3 MoE 走 FlyDSL 的 MLIR Python kernel 栈——硬件调过的 A16W4/A8W4 量化 fused op 和 SiTU，建在 FlyDSL 的模块抽象上。

## 开源当天可以预期什么

计划中的 Day-0 包裹：

- vLLM 模型、parser、cache、kernel 集成
- 初版开源 Docker
- 验过的 NVIDIA 启动配方
- 初版 AMD 路径（FlyDSL MoE）；更多 ROCm 调优随后
- 多模态、工具、reasoning、structured-output 例子
- 初版性能数字

受信任的部署伙伴已经在 Moonshot 和 vLLM/Inferact 双重批准下跑发布候选。不必把预发布权重铺得很开，也能拿到生产反馈。测的是整套 serving——前端语义、batch、cache 搬运、expert parallelism、可观测、故障处理——不只是孤立 kernel。

## 致谢

模型方、推理引擎、硬件社区一起。

**Moonshot AI** — Kimi K3，权重前分享架构，初版模型集成和 KDA 前缀缓存，正确性和生产验证。

**Inferact** — 接到 vLLM，扩展 core cache manager 做 hybrid 部分前缀命中，serving 语义和多模态，部署配方，端到端性能。

**NVIDIA** — KDA Decode 和 Attention Residual kernel，MXFP4 MoE，整条性能。

**AMD** — 初版 Day-0 ROCm；继续把 Kimi K3 铺到更多 AMD GPU。

更广的开源社区：期待、测试、反馈。

## 还有一件事：为什么宣布和开源拆开

先宣布模型，再发权重和推理引擎支持。

vLLM 提的拆法，Moonshot 同意。实际：前沿模型宣布总有最后一公里的不确定。模型团队同时在稳产品、API、评测、安全、文档、商业发布。如果开源权重和开源支持必须落在 **同一时刻**，像 vLLM 这样的社区项目会被那根移动的 deadline 拖着走。

拆开时间线：

1. 厂商可以冻住最终 checkpoint、配置、tokenizer、serving 语义。
2. 开源推理团队拿到稳定窗口：正确性、性能、Docker、配方。
3. 社区拿到公开、有界的预期，而不是含糊的「即将到来」。

不是从 Day-0 后退。是用用户真正会下载的那份产物，更可持续地把 Day-0 做完。
