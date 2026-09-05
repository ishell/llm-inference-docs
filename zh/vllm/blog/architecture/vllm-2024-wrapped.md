---
source: https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 2024 年报 / 2025 愿景：星标 2.3×，V1 重写，遥测可关

英文对照：[en/vllm/blog/architecture/vllm-2024-wrapped.md](../../../../en/vllm/blog/architecture/vllm-2024-wrapped.md)  
原文：https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision  
2025-01-10。署名 **vLLM Team**。当时的愿景文档；后来的 V1 / MRV2 / Wide-EP 才是落地。第 16 期双周 [Office Hours](https://hubs.li/Q02TFDTT0)；[录像](https://www.youtube.com/watch?v=xmz8lHsrbGM)。用法站：https://2024.vllm.ai。V1 重写见 [v1-alpha](v1-alpha.md)；更早的治理 / 性能路线见 [lfai-roadmap](lfai-roadmap.md)；后来的 runner 见 [mrv2](mrv2.md)。可插拔的门：[plugin-system](plugin-system.md)、[hardware-plugin](hardware-plugin.md)。口头里要变成默认的投机 / 结构化输出：[spec-decode](../performance/spec-decode.md)、[struct-decode](../performance/struct-decode.md)。

适用：读 2024 增长和 2025 口头路线（单节点 GPT-4o 级、开箱生产、V1）。不适合：把这页当当前架构——它是年报，不是后来的落地。

## 增长（页上的数）

- GitHub star **14,000 → 32,600**（**2.3×**）
- 贡献者 **190 → 740**（**3.8×**）
- 月下载 **6,000 → 27,000**（**4.5×**）
- 近六个月 GPU hours 约 **10×**（发文时）

生产例子：Amazon Rufus、LinkedIn AI。双月 meetup 里有 IBM、AWS、NVIDIA。口头目标：开源 AI 生态的万能 serving。

## 2024：模型、硬件、功能

### 社区

![vllm contributor groups](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/01-vllm-contributor-groups.png)

**Figure。** 按 commit 计的主要贡献者群体（学习对照）。

- **6+** 家组织里 **15+** 全职贡献者
- **20+** 家活跃组织当 stakeholder / sponsor
- UC Berkeley、Neural Magic、Anyscale、Roblox、IBM、AMD、Intel、NVIDIA，加上个人
- 模型作者、硬件厂、优化人连成的生态
- 双周 office hours

### 模型

![model architecture serving usage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/02-model-architecture-serving-usage.png)

**Figure。** serving 里按模型架构的用量（学习对照）。

2024 年初：寥寥几种。年末：近 [**100 种架构**](https://docs.vllm.ai/en/latest/models/supported_models.html)——主流开源 LLM、多模态（图 / 音频 / 视频）、encoder-decoder、投机解码、分类、embedding、reward。**状态空间** 语言模型也进了生产支持。

### 硬件

![gpu hours by vendor](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/03-gpu-hours-by-vendor.png)

**Figure。** 按厂商拆的 GPU hours（学习对照）。

最初瞄准 NVIDIA A100：

- **NVIDIA：** H100 一等公民；V100 及更新
- **AMD：** MI200、MI300、Radeon RX 7900；MI300X 用量在长
- **Google TPU：** v4、v5p、v5e、v6e
- **AWS Inferentia / Trainium：** trn1 / inf2
- **Intel Gaudi（HPU）和 GPU（XPU）**
- **CPU：** x86、ARM、PowerPC

口头路径：所有模型、所有硬件、优化都开。

### 功能

![quantization deployment percentage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/04-quantization-deployment-percentage.png)

**Figure。** 用了量化的部署占比（学习对照）。

- **权重与激活量化：** FP8+INT8 激活量化；GPTQ/AWQ/wNa16 的 Marlin+Machete；FP8 KV cache；AQLM、QQQ、HQQ、bitsandbytes、GGUF。**超过 20%** 的部署用了量化。
- **Automatic prefix caching**
- **Chunked prefill**（交互场景 ITL 更稳）
- **投机解码：** draft 模型、prompt 里 n-gram、MLP speculator（Medusa / EAGLE）
- **结构化输出**（JSON、pydantic）
- **Tool calling**（靠 chat template）
- **分布式推理：** pipeline parallelism、prefill 拆开

## 2025 愿景（当时口头）

开源模型在追闭源；蒸馏让它们更小、更好上生产。预训练规模和推理时 scaling 两边都在推。

### 单卡 / 单节点 GPT-4o 级

口头：单卡 GPT-4o 级，单节点 GPT-4o，下一代规模用不大的集群。三条前线：

- KV / attention：sliding window、跨层 attention、原生量化
- MoE：共享专家、大量细粒度专家
- 长上下文：状态空间一类替代架构

垂直场景：推理（自定义 token、灵活步数）、写代码（FIM、prompt lookup）、agent（树状缓存）、创作（beam 变体、contrastive decode）。后训练：点名 John Schulman 当信号；跟数据策展 / 后训练接得更紧。

### 上千个生产集群

量化、prefix caching、投机解码变成 **默认**，不是选项。结构化输出当标配。routing / caching / auto-scaling 的菜谱。集群级稳定接口；按模型 / 硬件给稳健默认；有人专门把效率往上推。

### 开放架构 / V1

从零重写的 **V1**：模型架构、调度、内存、采样——研究 fork 和私有 fork 都该改得动。模型、硬件、扩展可插拔。一等公民 `torch.compile`（自定义融合）。私有扩展灵活、核心仍稳。招核心团队；给生态项目留位子。可扩展，而不是锁死。

## 反省（页上的主题）

**搭桥。** 模型作者、硬件厂、优化人把 vLLM 当放大器：新加速器立刻有应用生态；新技术立刻有生产演示。贡献 ↔ 放大。

**增长对卓越。** 2024 的速度也把代码库变复杂。下半年把核心重设计 → **V1**，好让平台还能养。

**赞助志愿者组织。** 不是一家公司出钱。多家组织出代码、资源、方向。跨组织协同还在发明。

**使命句：** 世界上最快、最好用的开源 LLM 推理与 serving 引擎。

## 用法统计

数字来自 vLLM 的 [usage system](https://github.com/vllm-project/vllm/blob/main/vllm/usage/usage_lib.py)。每个实例一个 UUID；技术字段：

- 硬件（GPU 数量 / 类型、CPU 架构、内存）
- 模型配置（架构、dtype、TP 度）
- 运行时（量化类型、prefix caching）
- 上下文（云、平台、vLLM 版本）

本地文件：`~/.config/vllm/usage_stats.json`。关掉：`VLLM_NO_USAGE_STATS=1`、`DO_NOT_TRACK=1`，或建 `~/.config/vllm/do_not_track`。schema：[usage stats 文档](https://docs.vllm.ai/en/latest/serving/usage_stats.html)。

## 加入（原文）

贡献（RFC 还开着）；GitHub / Slack / Discord / 活动给反馈；用 vLLM 做事。[Developer Slack](https://slack.vllm.ai/)。Office Hours 同上。
