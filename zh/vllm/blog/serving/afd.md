---
source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# AFD Plugin：Attention 和 FFN 也可以不住在同一栋楼

英文对照：[en/vllm/blog/serving/afd.md](../../../../en/vllm/blog/serving/afd.md)  
原文：https://vllm.ai/blog/2026-07-23-vllm-afd-plugin  
2026-07-23。实验性外部插件：https://github.com/vllm-project/afd-plugin。走 `vllm.general_plugins` 和 `--additional-config`，**不改 vLLM 源码**。当时钉在 vLLM **0.19.1**、Python **3.10–3.13**、仅 model runner **v1**。两边都加载**完整权重**。数字是受控实验，不是 SLA。原文自己也说：还需要在更多后端上做大规模测试。

本地图（原文版权仍归原站；学习对照用）：

![vllm afd plugin architecture](../../../../assets/vllm/blog/serving/afd/01-vllm-afd-plugin-architecture.svg)

![throughput dsv3 2 16k](../../../../assets/vllm/blog/serving/afd/02-throughput_dsv3-2_16k.png)

![throughput dsv3 2 32k](../../../../assets/vllm/blog/serving/afd/03-throughput_dsv3-2_32k.png)

![text matched dp afd median ttft](../../../../assets/vllm/blog/serving/afd/04-text_matched_dp_afd_median_ttft.png)

## 为什么要把 Attention 和 FFN 拆开

MoE 每一层里两件脾气相反的活。Attention **有状态**，跟调度和 KV 绑在一起；FFN / 专家是 routed compute + all-to-all。绑在同一套 rank 上，伸缩只能选一个数字。

插件在回答的系统题：

1. **伸缩需求不同。** Attention 跟着请求状态、序列长度、KV 压力走；专家跟着 routing 和负载走。拓扑应该允许不一样。
2. **运行时职责不同。** Attention 留下调度、KV、采样；FFN 只要 activation、routing 元数据、和一条回家的路。FFN 可以是一只 connector 驱动的 **daemon**。
3. **通信绑在后端上。** CUDA 和昇腾：集合通信、graph 运行时、MoE 算子都不一样。一份**中立的 connector 合同**让模型侧的流程稳住。
4. **重叠。** 异步 dispatch 和 MoE ubatch 可以把专家活从 Attention 后面拉开，不必全串行。

请求仍然打到 **Attention** 的 OpenAI 口。vLLM 继续管 serving 控制面；plugin 管 AFD worker、runner、connector、元数据、切开点，以及一小撮跟版本绑定的兼容补丁。

## 架构

三件零件：

- **Attention worker。** 调度、KV、batch、生命周期、采样都留在这边。Plugin 的 model runner 把 AFD 元数据装进 forward context，把 DP / ubatch / 层 / graph 状态告诉 FFN。
- **FFN worker。** 没有请求、没有 KV。后台循环：元数据 + activation → 包装上的 `compute_ffn_output()` → 送回。
- **Connector。** 每个切开的层上：Attention 的 hidden state 和执行元数据过去，FFN 输出回来。

GPU worker 扩的是 vLLM v1 类；NPU worker 直接扩 **vLLM-Ascend** 类。共享的是配置、拓扑、元数据、connector 合同——不是跨设备继承。

### Connector

| Connector | 后端 | 执行 | 适合 | Graph |
| --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | GPU | 同步 P2P | Decode | `FULL_DECODE_ONLY` CUDA graph |
| `CAMP2pAFDConnector` | NPU | 同步 CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY` ACL graph |
| `CAMAsyncAFDConnector` | NPU | 异步 CAM | Prefill | **当时还没有 graph** |

高层交换一样；后端包装分开，免得 CUDA graph、ACL graph、NCCL、昇腾算子互相渗。

### 当时已支持的

- 原有的 `vllm serve` + OpenAI 口 + `--additional-config`
- GPU 与 NPU 实现
- 同步 AFD 伺候 Decode 吞吐（`P2pNcclAFDConnector`、`CAMP2pAFDConnector`），graph 语义是 `FULL_DECODE_ONLY`
- 异步 AFD 伺候 Prefill（`CAMAsyncAFDConnector`）：CAM 异步 dispatch/combine，AFD 管的 MoE ubatch，目标是 **P/D 分离里的 Prefill**；**当时还没有 graph**
- 模型包装：DeepSeek **V2/V3 家族**（含 **V3.2**）、**GLM MoE DSA**——切开 Attention / FFN，层实现仍复用上游
- Dual Batch Overlap：**恰好两个** ubatch；CAM async 的 Prefill 另有自己的 ubatch

## 成绩快照（受控）

### 同步 Decode，`CAMP2pAFDConnector`

食谱：[afd-plugin#67](https://github.com/vllm-project/afd-plugin/pull/67)。DeepSeek-V3.2 **W8A8**，昇腾 **910C**。饱和 Decode 吞吐，不是在线延迟。

| 部署 | 物理拓扑 | 总 die |
| --- | --- | --- |
| EP64 | DP64, EP64, TP1 | 64 |
| 48A16F | 48 Attention + 16 FFN | 64 |
| 64A16F | 64 Attention + 16 FFN | 80 |

**当时的边界：** 不是精度实验，也不是生产 serving。机器不够：物理 48A16F / 64A16F 用来**模拟**逻辑上的 **192A64F / 256A64F**。专家 ID 换成**确定性的强制均衡环**——**输出会变**。`AFDDecodeBenchConnector` 提供 decode-only KV；AFD 开了 **DBO**。

按 die 归一：`tokens/s/die = 合计 output token 吞吐 / 部署的总 die 数`。固定长度输入；输出在 **512–1536** token 上均匀。

**16K**（Figure 2）：EP64 **232.6**；48A16F **220.3（−5.3%）**；64A16F **258.9（+11.3%）**。

**32K**（Figure 3）：EP64 **168.2**；48A16F **151.4（−10.0%）**；64A16F **183.3（+9.0%）**。

拆开本身不保证更快——Attention 和 FFN 的**配比**才是那句话。更高的 Attention 比他们当时没测到；趋势像是 FFN 侧还有**算力余量**，再加 Attention 可能还有空间。

### 异步 Prefill，`CAMAsyncAFDConnector`

早期实验：**两台** 910C，DeepSeek V3.2 W8A8 **砍到 10 层**，强制均衡专家。基线 `DP4 PCP8 TP1` 对 Attention `DP3 PCP8 TP1` + FFN `EP8`。Figure 4。

测过的请求率上，AFD 降低中位 / P50 TTFT。**12 rps**：**15.1 s → 8.0 s**（约 **47%**）。**10 和 12 rps** 的差距大约都是 **7.2 s**。这是路径验证，**不是**全模型声明；收益随负载变。

## 怎么开始

安装见插件 [README](https://github.com/vllm-project/afd-plugin#install)。部署命令在仓库 recipes 里，博客不重复抄：

- GPU 同步：[DeepSeek V2 Lite P2P NCCL](https://github.com/vllm-project/afd-plugin/tree/main/recipe/gpu/p2p_nccl/deepseek_v2_lite)——混跑与 P/D 分离的 Decode、eager 与 CUDA graph、多种 DP/TP
- NPU 异步 Prefill：[DeepSeek V3.2 CAM async](https://github.com/vllm-project/afd-plugin/blob/main/recipe/npu/cam_async/DeepSeek-V3.2.md)——环境、拓扑、AFD 配置、bench、当时的限制

## 当时的范围和下一步

他们自己列出的边界：钉死的 vLLM 版本、只 runner v1、**两边完整权重**、只 Decode 的 graph 模式、DBO **恰好两个** ubatch、端到端测试被硬件门卡住。

下一步：跟上更新的 vLLM、评估 **model runner v2**、补丁尽量小、能上游的就上游；更多 graph / ubatch / 异步阶段 / 拓扑；全模型、真实负载上的精度 / 延迟 / 吞吐 / 稳定性 / 多机；更多 MoE 包装和传输；**vLLM-Omni** / 多模态（AR、**DiT**、以及任何想让 Attention 和 FFN 独立伸缩的阶段）；异构加速器和互联，为 TTFT 和 ITL 做重叠。

链接：[代码](https://github.com/vllm-project/afd-plugin)，[GPU / 昇腾设计文档](https://github.com/vllm-project/afd-plugin/tree/main/docs)，[Issues](https://github.com/vllm-project/afd-plugin/issues)。

EPD 拆的是视觉编码器；Router 拆的是文本 P/D；AFD 拆的是层内的 Attention 与专家。三把刀切的不是同一块肉。
