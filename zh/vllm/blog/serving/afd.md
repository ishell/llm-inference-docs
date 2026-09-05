---
source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# AFD Plugin：Attention 和 FFN 也可以不住在同一栋楼

英文对照：[en/vllm/blog/serving/afd.md](../../../../en/vllm/blog/serving/afd.md)  
原文：https://vllm.ai/blog/2026-07-23-vllm-afd-plugin  
2026-07-23。署名 **AFD Plugin Contributors**。实验性外部插件：https://github.com/vllm-project/afd-plugin。走 `vllm.general_plugins` 和 `--additional-config`，**不改 vLLM 源码**。当时钉在 vLLM **0.19.1**、Python **3.10–3.13**、仅 model runner **v1**。两边都加载**完整权重**。数字是受控实验，不是 SLA。原文自己也说：还需要在更多后端上做大规模测试。

EPD 拆的是视觉编码器；Router 拆的是文本 Prefill/Decode；AFD 拆的是层内 Attention 与专家。三把刀切的不是同一块肉。插件系统：[plugin-system](../architecture/plugin-system.md)；硬件门：[hardware-plugin](../architecture/hardware-plugin.md)；当时还没接的 runner：[mrv2](../architecture/mrv2.md)。

**原文 TL;DR：**

- Attention–FFN Disaggregation（AFD）：Attention 和 FFN 独立部署，请求生命周期和 OpenAI 口仍留给 vLLM。
- 后端：NVIDIA GPU 与昇腾 NPU。Connector：`P2pNcclAFDConnector`、`CAMP2pAFDConnector`、`CAMAsyncAFDConnector`。
- 同步 Decode 走 `FULL_DECODE_ONLY` graph；异步 Prefill **当时还没有 graph**。
- 包装：DeepSeek V2/V3 家族（含 V3.2）、GLM MoE DSA。DBO **恰好两个** ubatch。
- 910C 上 DeepSeek-V3.2 W8A8：64A16F 相对 EP64，16K **+11.3%**、32K **+9.0%** tokens/s/die。异步 Prefill 10 层实验：12 rps 中位 TTFT **15.1 s → 8.0 s**。

原文分节：Why Attention-FFN Disaggregation? → Inside the Architecture（Connector and backend support / Supported features）→ A Performance Snapshot（Synchronous AFD Decode Throughput with `CAMP2pAFDConnector`：16K / 32K；Asynchronous AFD Prefill Performance with `CAMAsyncAFDConnector`）→ Getting Started（Install / Deployment Recipes）→ Current Scope and Roadmap → Join the Community。

[vLLM AFD Plugin](https://github.com/vllm-project/afd-plugin) 把 Attention-FFN Disaggregation 接到 MoE：Attention 和 FFN 分成独立服务。请求生命周期和 OpenAI 兼容口不动，两条路径可以各自伸缩。

当时已支持 NVIDIA GPU 和昇腾 NPU、同步与异步 connector、DeepSeek V2/V3 家族包装，以及在已验证边界内的 eager / graph / dual-batch 路径。

> 项目仍是实验性的，还需要在不同硬件后端上做更大规模的测试。

## Why Attention-FFN Disaggregation?

MoE 推理在每一层 Transformer 里叠了两种脾气相反的活。Attention **有状态**，跟调度和 KV cache 绑在一起；FFN / 专家路径主要是 routed 专家计算和 all-to-all。两条路共用同一套 worker 拓扑，serving 就只能给两种完全不同的需求选**一个**伸缩数字。

要把拆开做成能跑的系统，得先回答几道设计题：

1. **Attention 和 FFN 的伸缩需求不同。** Attention 容量跟着请求状态、序列长度、KV 压力走。专家容量跟着 token routing 和专家负载走。serving 应该允许两边用不同的 rank 拓扑，而不是强迫共享一份布局。
2. **运行时职责不同。** Attention 要调度、KV 协调、采样。FFN 只要 activation、routing 元数据、和一条把专家输出送回家的路。拆开之后，FFN 可以是一只 connector 驱动的轻量 **daemon**。
3. **通信绑在后端上。** CUDA 和昇腾：集合通信库、graph 运行时、优化过的 MoE 算子都不一样。一份中立的 connector 合同让模型侧的流程稳住，各后端自己管数据路径。
4. **通信和计算都该重叠。** 异步 dispatch 和 MoE ubatch 可以把彼此独立的阶段叠起来，不必把全部专家活串在 Attention 后面。

合在一起，AFD 的核心目标就清楚了：请求面前的 Attention 路径仍是 vLLM 的，FFN 执行退到一条窄的 connector 接口后面——可以自己伸缩、自己通信、自己执行。

## Inside the Architecture

![vllm afd plugin architecture](../../../../assets/vllm/blog/serving/afd/01-vllm-afd-plugin-architecture.svg)

**Figure。** vLLM AFD Plugin 运行时（学习对照；版权仍归原站）。

插件从 `vllm.general_plugins` 入口和标准 `--additional-config` 通道接入，**不要求改 vLLM 源码树**。

运行时三件零件：

- **Attention service。** Attention worker 留下 vLLM 的调度器、KV cache、batching、模型生命周期、采样。插件自己的 model runner 把 AFD 元数据装进 forward context，并把 data-parallel、ubatch、层、graph 状态发布给 FFN 一侧。
- **FFN service。** FFN worker 没有请求流量、没有调度器、没有 KV。后台循环：收到元数据和 activation → 在插件包装上调用 `compute_ffn_output()` → 把结果送回 Attention。请求**永远**打到 Attention 的 API server。
- **Connector layer。** 每个切开的层上，connector 把 Attention hidden state 连同 FFN 需要的执行元数据送过去，再把算好的 FFN 输出送回来。后端中立的接口定义这次交换；各后端自己实现通信和运行时优化。

接入面故意收得很窄。vLLM 继续管 serving 控制面——现有抽象本来就适合待在那儿。插件提供 AFD worker、model runner、connector、元数据、模型切开点，以及一小撮跟版本绑定的兼容补丁。

### Connector and backend support

| Connector | Backend | Execution | Recommended stage | Graph support |
| --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | GPU | Synchronous P2P | Decode | `FULL_DECODE_ONLY` CUDA graph |
| `CAMP2pAFDConnector` | NPU | Synchronous CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY` ACL graph |
| `CAMAsyncAFDConnector` | NPU | Asynchronous CAM | Prefill | **当时还不支持** |

高层交换一样：Attention 输出去 FFN，FFN 输出回 Attention。后端包装分开，免得 CUDA graph、ACL graph、NCCL、昇腾自定义算子互相渗。

### Supported features

- **Native vLLM serving surface。** 现有用户仍用 `vllm serve` 启动，请求打 OpenAI 兼容口，运行时用 `--additional-config` 配。
- **GPU 和 NPU 实现。** GPU worker 扩 vLLM v1 类；NPU worker 直接扩 **vLLM-Ascend** 类。共享行为住在配置、拓扑、元数据、connector 合同里，不是跨设备继承。
- **同步 AFD 伺候 Decode 吞吐。** `P2pNcclAFDConnector` 和 `CAMP2pAFDConnector` 同步交换 Attention activation 和 FFN 输出，让两个角色在吞吐型 Decode 部署里独立伸缩。当时的 graph 路径分别是 CUDA / ACL 上的 `FULL_DECODE_ONLY`。
- **异步 AFD 伺候 Prefill。** `CAMAsyncAFDConnector` 用 CAM 异步 dispatch / combine，把 Prefill 的 Attention rank 和专家 worker 解开。配上 AFD 管的 MoE ubatch，独立的 Attention / FFN 阶段可以重叠，少卡在流水线里。这条路当时瞄准 **P/D 分离里的 Prefill**，**还不支持 graph**。
- **MoE 模型接入。** 包装注册 DeepSeek V2/V3 家族（含 DeepSeek V3.2）和 GLM MoE DSA。包装把 Attention 和 FFN 计算分开，层实现仍复用上游。
- **Graph 和 ubatch。** 同步 GPU / NPU connector 支持 Decode-only graph capture。Dual Batch Overlap **恰好两个** ubatch；CAM async 的 Prefill 另有 AFD 管的 MoE ubatch。

## A Performance Snapshot

### Synchronous AFD Decode Throughput with `CAMP2pAFDConnector`

同步 Decode 食谱：[vllm-project/afd-plugin#67](https://github.com/vllm-project/afd-plugin/pull/67)。对照常规 EP64 和基于 `CAMP2pAFDConnector` 的 AFD。模型：DeepSeek-V3.2 **W8A8**，昇腾 **910C**。测的是饱和 Decode 吞吐，不是在线 serving 延迟。

| Deployment | Physical topology | Total dies |
| --- | --- | ---: |
| EP64 | DP64, EP64, TP1 | 64 |
| 48A16F | 48 Attention ranks, 16 FFN ranks | 64 |
| 64A16F | 64 Attention ranks, 16 FFN ranks | 80 |

> 受控性能结果，不是精度实验，也不是生产 serving。机器不够：物理 48A16F / 64A16F 用来**模拟**逻辑上的 **192A64F / 256A64F**。自然 routed 的专家 ID 换成**确定性的强制均衡环**——**模型输出会变**。`AFDDecodeBenchConnector` 提供 decode-only KV；AFD 开了 **DBO**。

吞吐按部署的总 die 数归一：

```text
tokens/s/die = aggregate output token throughput / total deployed dies
```

两边负载都是固定长度输入；输出在 **512–1536** token 上均匀。

#### 16K fixed input

![throughput dsv3 2 16k](../../../../assets/vllm/blog/serving/afd/02-throughput_dsv3-2_16k.png)

**Figure。** DeepSeek-V3.2 16K Decode，每 die 吞吐。

EP64 **232.6** tokens/s/die；48A16F **220.3**；64A16F **258.9**。相对 EP64：48A16F **−5.3%**，64A16F **+11.3%**。

#### 32K fixed input

![throughput dsv3 2 32k](../../../../assets/vllm/blog/serving/afd/03-throughput_dsv3-2_32k.png)

**Figure。** DeepSeek-V3.2 32K Decode，每 die 吞吐。

EP64 **168.2** tokens/s/die；48A16F **151.4**；64A16F **183.3**。相对 EP64：48A16F **−10.0%**，64A16F **+9.0%**。

两种输入长度上，48A16F 都低于 EP64 基线；64A16F 归一吞吐最高：16K **+11.3%**，32K **+9.0%**。这句话是：Attention 对 FFN 的**配比**才要紧；拆开本身不保证吞吐上涨。

机器不够，他们没测更高的 Attention:FFN 比。当时看到的趋势：测过的配比上，FFN rank 仍有**算力余量**，还没算力打满。再提高 Attention rank 的比例，吞吐或许还有空间。

### Asynchronous AFD Prefill Performance with `CAMAsyncAFDConnector`

仓库里有一次早期 CAM async 实验：两台昇腾 910C，DeepSeek V3.2 W8A8 **砍到 10 层**。强制专家均衡。基线 `DP4PCP8 TP1`，对照 AFD：Attention `DP3PCP8 TP1` + FFN `EP8`。

![text matched dp afd median ttft](../../../../assets/vllm/blog/serving/afd/04-text_matched_dp_afd_median_ttft.png)

**Figure。** CAM async 实验的中位 TTFT。

测过的请求率上，AFD 降低中位 / P50 TTFT。**12 rps**：**15.1 s → 8.0 s**，大约 **47%**。**10 和 12 rps** 的差距大约都是 **7.2 s**。

这是 CAM async 执行路径的定点验证，**不是**完整 DeepSeek V3.2、也不是每一种 AFD 拓扑的通用声明。收益随负载变。

## Getting Started

当时实现要求 Python **3.10–3.13**，目标 vLLM **`0.19.1`**。

### Install

安装步骤看插件 [README](https://github.com/vllm-project/afd-plugin#install)，博客不另抄一份。

### Deployment Recipes

部署命令取决于后端、connector、模型、rank 拓扑。不要在博客里复制配置，用仓库里维护的 [AFD Plugin recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe)：

- **GPU 同步 AFD：** [DeepSeek V2 Lite P2P NCCL recipes](https://github.com/vllm-project/afd-plugin/tree/main/recipe/gpu/p2p_nccl/deepseek_v2_lite)——Decode 向的混跑与 Prefill/Decode 分离、eager 与 CUDA graph、多种 DP/TP。
- **NPU 异步 Prefill AFD：** [DeepSeek V3.2 CAM async recipe](https://github.com/vllm-project/afd-plugin/blob/main/recipe/npu/cam_async/DeepSeek-V3.2.md)——环境、拓扑、AFD 配置、bench、当时的限制。

最新 connector 矩阵、配置字段、完整启动命令，以仓库 README 和 recipe 目录为准。

## Current Scope and Roadmap

他们故意把当时的边界摊开：精确的 vLLM 版本钉死、仅 model runner v1、**两边完整权重**、只 Decode 的 graph 模式、DBO **恰好两个** ubatch、端到端测试被硬件门卡住。

下一阶段当时写的是：

- **更宽的 vLLM 兼容和上游对齐：** 跟上更新的 vLLM，评估 **model runner v2**，补丁尽量小，能通用的抽象成熟了就往上游送。
- **更灵活的执行：** 扩 graph 模式、ubatch 个数、异步阶段、已验证的 rank 拓扑。
- **生产规模验证：** 在全模型和真实负载上公布可复现的精度、延迟、吞吐、稳定性、多机结果。
- **更多模型和 connector：** 经现有包装和 connector 接口加 MoE 架构和后端传输，每个新模型和 connector 配部署 recipe。
- **多模态和 vLLM-Omni：** 探索 AFD 怎么进 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 和异构多模态流水线——自回归（AR）、Diffusion Transformer（**DiT**）、以及其他想让 Attention 和 FFN 独立伸缩的阶段。
- **异构硬件和低延迟 serving：** 把 Attention 和 FFN 角色放到不同加速器和互联上；connector、调度、放置、以及为 TTFT 和 ITL 做的计算–通信重叠。

## Join the Community

插件还早，模型和 serving、硬件社区的反馈会定方向。

- **代码和文档：** [github.com/vllm-project/afd-plugin](https://github.com/vllm-project/afd-plugin)
- **运行时设计文档：** [GPU Attention/FFN 与昇腾 Attention/FFN](https://github.com/vllm-project/afd-plugin/tree/main/docs)
- **Issues：** [GitHub Issues](https://github.com/vllm-project/afd-plugin/issues)
