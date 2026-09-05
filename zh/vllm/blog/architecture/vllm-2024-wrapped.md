---
source: https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 2024 年报 / 2025 愿景：星标 2.3×，V1 重写，遥测可关

英文对照：[en/vllm/blog/architecture/vllm-2024-wrapped.md](../../../../en/vllm/blog/architecture/vllm-2024-wrapped.md)  
原文：https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision  
2025-01-10。署名 **vLLM Team**。学习译文，不是官方译本。当时的愿景文档；后来的 V1 / MRV2 / Wide-EP 才是落地。基于第 16 期双周 [Office Hours](https://hubs.li/Q02TFDTT0)；[录像](https://www.youtube.com/watch?v=xmz8lHsrbGM)。用法站：https://2024.vllm.ai。V1 重写见 [v1-alpha](v1-alpha.md)；更早的治理 / 性能路线见 [lfai-roadmap](lfai-roadmap.md)；后来的 runner 见 [mrv2](mrv2.md)。可插拔的门：[plugin-system](plugin-system.md)、[hardware-plugin](hardware-plugin.md)。口头里要变成默认的投机 / 结构化输出：[spec-decode](../performance/spec-decode.md)、[struct-decode](../performance/struct-decode.md)。

适用：读 2024 增长和 2025 口头路线（单节点 GPT-4o 级、开箱生产、V1）。不适合：把这页当当前架构——它是年报，不是后来的落地。

2024 年，vLLM 社区从一只专门的推理引擎，长成开源 AI 生态里事实上的 serving 方案。增长写在这些数上：

- GitHub star：**14,000 → 32,600**（**2.3×**）
- 贡献者：**190 → 740**（**3.8×**）
- 月下载：**6,000 → 27,000**（**4.5×**）
- GPU hours：近六个月大约 **10×**
- 更多用法数据：[https://2024.vllm.ai](https://2024.vllm.ai)

他们写：vLLM 已经是领先的开源 LLM serving 与推理引擎，生产里用得很开（例如 Amazon Rufus、LinkedIn AI）。双月 meetup 变成和 IBM、AWS、NVIDIA 谈伙伴关系的场合，口头目标是开源 AI 生态的万能 serving。下文是 2024 做成了什么、2025 打算往哪走。

*这篇基于第 16 期双周 [vLLM Office Hours](https://hubs.li/Q02TFDTT0)。录像在 [这里](https://www.youtube.com/watch?v=xmz8lHsrbGM)。*

## 2024 Achievements: Scaling Models, Hardware, and Features

### Community Contributions and Growth

![vllm contributor groups](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/01-vllm-contributor-groups.png)

**Figure。** vLLM 主要贡献者群体（按 commit；学习对照）。

2024 对 vLLM 是特别的一年。贡献社区扩得很开：

- **6+** 家组织里 **15+** 全职贡献者
- **20+** 家活跃组织当关键 stakeholder / sponsor
- UC Berkeley、Neural Magic、Anyscale、Roblox、IBM、AMD、Intel、NVIDIA，以及世界各地的个人
- 模型作者、硬件厂、优化人连成的生态
- 出席很好的双周 office hours：透明、社区、战略伙伴

这些数不只是变大。它们说明 vLLM 已经是 AI 生态里的关键基建：从研究原型到给几百万人说话的生产系统，都靠它撑着。

### Expanding Model Support

![model architecture serving usage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/02-model-architecture-serving-usage.png)

**Figure。** serving 里按模型架构的用量（学习对照）。

2024 年初：只伺候寥寥几种模型。年末：近 [**100 种架构**](https://docs.vllm.ai/en/latest/models/supported_models.html) 能跑得动——几乎每只显眼的开源 LLM、多模态（图 / 音频 / 视频）、encoder-decoder、投机解码、分类、embedding、reward。特别写了一笔：状态空间语言模型进了 **生产支持**，在探非 Transformer 语言模型的下一步。

### Broadening Hardware Compatibility

![gpu hours by vendor](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/03-gpu-hours-by-vendor.png)

**Figure。** 按硬件厂商拆的 GPU hours（学习对照）。

最初瞄准 NVIDIA A100。之后铺开：

- **NVIDIA GPU：** H100 一等公民优化；从 V100 起每一代都支持
- **AMD GPU：** MI200、MI300、Radeon RX 7900；MI300X 用量在很快变长
- **Google TPU：** v4、v5p、v5e，以及当时最新的 v6e
- **AWS Inferentia 和 Trainium：** trn1 / inf2
- **Intel Gaudi（HPU）和 GPU（XPU）：** 把 Intel GPU 和 Gaudi 用到 AI 负载上
- **CPU：** 越来越长的 ISA 名单——x86、ARM、PowerPC

硬件面变宽，是为了不同人的要求，同时也把性能改进带进去。口头路径：**所有模型在所有硬件上都能跑，优化都开着。**

### Delivering Key Features

![quantization deployment percentage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/04-quantization-deployment-percentage.png)

**Figure。** 用了量化的 vLLM 部署占比在涨（学习对照）。

2024 的开发路线强调性能、可扩展、好用：

- **权重与激活量化。** 多种量化方法和 kernel，好在不同硬件上高效推理。点名：FP8+INT8 的激活量化；GPTQ/AWQ/wNa16 的 Marlin+Machete；FP8 KV cache；AQLM、QQQ、HQQ、bitsandbytes、GGUF。**超过 20%** 的部署用了量化。
- **Automatic Prefix Caching。** 给上下文很重的应用减成本、压延迟。
- **Chunked Prefill。** 交互场景里 ITL（inter-token latency）更稳。
- **投机解码。** 同时预测和验证 token，加速生成。支持 draft 模型、prompt 里 n-gram matching、MLP speculator（Medusa / EAGLE）。
- **结构化输出。** 要 JSON、pydantic 这类格式时，走高性能路径。
- **Tool calling。** 有支持的 chat template 的模型可以自己生成 tool call，方便数据处理和 agent 流。
- **分布式推理。** pipeline parallelism 和拆开的 prefill，把负载铺到多卡、多节点。

## Our 2025 Vision

他们预期 2025 会同时推预训练规模和推理时 scaling 的边界。开源模型在迅速追上闭源；蒸馏让巨大的模型变小、变聪明、更适合上生产。

### Emerging Model Capabilities: GPT-4o Class Models served on single node

愿景写得很具体：单卡达到 GPT-4o 级，单节点跑 GPT-4o，下一代规模用不大的集群。三条优化前线：

- KV cache 和 attention：sliding window、跨层 attention、原生量化
- MoE：共享专家、大量细粒度专家
- 长上下文：状态空间一类替代架构

性能之外，还要按垂直场景裁剪。每种用法要自己的优化：推理应用要自定义 token 和灵活的推理步数；写代码要 fill-in-the-middle 和 prompt lookup decoding；agent 框架吃树状缓存；创作要多样的采样，包括 beam 变体和 contrastive decode。

vLLM 在模型训练流程里的位子也在扩。John Schulman 这类研究者开始用，被当成后训练变重要的信号。他们打算和数据策展、后训练接得更紧，让 vLLM 成为整条 AI 开发生命周期里的工具，而不只是上线那一截。

### Practical Scale: Powering Thousands of Production Clusters

LLM 变成现代应用的脊梁时，他们看见 vLLM 给 **上千个** 生产集群 24/7 值班。不是实验部署——是产品功能上不断来的流量，由专门的平台团队养着。

为了这个规模，vLLM 要真正 **battery-included**。量化、prefix caching、投机解码变成 **默认**，不是可选项。结构化输出当标配，而不是特例。他们在写 routing、caching、auto-scaling 的完整菜谱，覆盖生产部署的整段生命周期。

部署越过单副本之后，要给集群级方案留稳定接口。按流行模型和硬件给稳健默认；再给多样场景留灵活的优化路径。他们要养一个专门把 vLLM 效率往上推的社区，让平台跟着新挑战长。

### Open Architecture: The Foundation of Our Future

继续成功的关键，他们说是开放架构。当时要发的从零重写的 **V1**，就是这句话的例子。每一个部件——模型架构、调度策略、内存管理、采样——都该能在研究 fork 和私有 fork 里改、能伸。

开放不只是代码。他们要引入：

- 可插拔架构：新模型、硬件 backend、自定义扩展接得上
- 一等公民 `torch.compile`：自定义算子融合 pass，实验可以很快
- 灵活的组件系统：私有扩展进得去，核心仍稳

社区开发要加倍：跨组织协调工程，同时给生态项目留位子。核心团队靠清楚的招聘和组织长起来。目标不只是技术上最好用——而是每一个往 vLLM 里投时间的人，都该觉得投对了。

架构不只是技术选择；它是一种承诺：靠可扩展和可修改连成生态，而不是锁死。vLLM 又强又能改，才坐得住推理生态的中心。

## A Bit of Reflection

回头看这段路，几条主题一直在塑造成长，也还在指路。

### Building Bridges in the AI Ecosystem

从一只推理引擎，长成一座桥：把 AI 地景里原先不相往来的世界接上。模型作者、硬件厂、优化专家在 vLLM 里找到一台放大器。硬件团队做出新加速器，立刻有一整片应用生态可进；研究者发明新优化，立刻有生产平台可演示。**贡献 ↔ 放大** 变成身份的一部分，逼着平台更好进、更好伸。

### Managing Growth While Maintaining Excellence

2024 的指数增长带来机会，也带来麻烦。代码库和贡献者扩得太快，速度前所未有：能啃更大的技术题，能对社区需要很快反应。同一股速度也把代码库变复杂。他们没有让技术债堆着，而是决定把地基重做。2024 下半年对核心架构做了一次大胆的重设计，就是后来的 **V1**。不只是技术翻新——是故意让平台在 AI 生态继续膨胀时，仍然养得动、拆得开。

### Pioneering a New Model of Open Source Development

也许最特别的挑战是：用一群 **受赞助的志愿者**，搭出世界级的工程组织。不像传统开源项目靠一家机构出钱，vLLM 在走另一条路。多家组织不只出代码，还出资源、出战略方向。协调、规划、执行都是新难题；创新和韧性也因此少了一家独大的单点。他们在学——有时是在发明——分布式决策、跨组织远程协作这一类的最佳实践。

### Our Unwavering Commitment

变来变去，根本使命仍清楚：做 **世界上最快、最好用的开源 LLM 推理与 serving 引擎**。把高效推理的门槛压低，先进的 AI 应用才更实际、更可及。不只是技术卓越——是给整个 AI 社区一块一起往前走的底板。

## Usage Data Collection

文中的指标和洞察，来自 vLLM 的 [usage system](https://github.com/vllm-project/vllm/blob/main/vllm/usage/usage_lib.py)，收集匿名部署数据。每个实例生成一个 UUID，上报技术字段：

- 硬件规格（GPU 数量 / 类型、CPU 架构、可用内存）
- 模型配置（架构、dtype、tensor parallelism 度）
- 运行时设置（量化类型、是否开 prefix caching）
- 部署上下文（云厂商、平台、vLLM 版本）

这份遥测用来给常见硬件配置排优化优先级，并看出哪些功能需要性能改进。数据落在本机 `~/.config/vllm/usage_stats.json`。关掉：设 `VLLM_NO_USAGE_STATS=1`、`DO_NOT_TRACK=1`，或建 `~/.config/vllm/do_not_track`。实现和完整 schema 在 [usage stats 文档](https://docs.vllm.ai/en/latest/serving/usage_stats.html)。

## Join the Journey

2024 的路说明开源协作能把格局改掉。2025 的愿景写清楚了：让 AI 推理更可及、更能铺、更高效。贡献代码、来 [Office Hours](https://hubs.li/Q02TFDTT0)、把 vLLM 用进生产——每一个人都在塑这条很快的项目。

进入 2025，他们继续请人参加：

- **Contributing Code：** 帮着拧核心，或把能力伸出去——许多 RFC 和功能还缺人手
- **Providing Feedback：** 功能、用例，经 GitHub / Slack / Discord / 活动，去塑路线图
- **Building with vLLM：** 用进自己的项目，把经验养出来，再把经验交回去

[Developer Slack](https://slack.vllm.ai/)：项目负责人带，站在 AI 推理创新的前头。

**Together, we'll advance open-source AI innovation in 2025!**
