---
source: https://vllm.ai/blog/2025-11-30-vllm-omni
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vLLM-Omni：文本之外的流水线

英文对照：[en/vllm/blog/serving/vllm-omni.md](../../../../en/vllm/blog/serving/vllm-omni.md)  
原文：https://vllm.ai/blog/2025-11-30-vllm-omni  
2025-11-30。署名 **vLLM-Omni Team**。仓库：[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)。文档：[vllm-omni.readthedocs.io](https://vllm-omni.readthedocs.io/en/latest/)。首发叠在 **vLLM v0.11.0** 上（包版本 **v0.11.0rc**）。这不是把一只 LLM 当万金油——**阶段异构** 才是这栋楼。页上对 Hugging Face Transformers 的吞吐对比是当时的图，不是你的 SLA。

后续把阶段再拆细的笔记：[扩散 cache](omni-diffusion-cache.md)、[TTS](omni-tts.md)、[layerwise offload](omni-layerwise-offload.md)。同目录里还有 [AutoRound](omni-autoround.md)、[Qwen3-Omni](qwen3-omni.md)。

## 为什么要 Omni

vLLM 从一开始盯的是高吞吐、省显存的 **LLM** serving。生成式 AI 的地形在变：不再只是 text-in、text-out。SOTA 要跨文本、图像、音频、视频推理，还用不同架构吐出异构输出。

**vLLM-Omni** 被写成开源里较早把 omni-modality serving 撑起来的框架之一：把 vLLM 的性能伸到多模态和非自回归推理。

本地图（原文版权仍归原站；学习对照用）：

![omni modality model architecture](../../../../assets/vllm/blog/serving/vllm-omni/01-omni-modality-model-architecture.png)

传统 serving 引擎为基于文本的自回归（AR）任务优化。模型变成会看、会听、会说的「omni」agent，基础设施得跟着变。原文点了架构上的三记转向：

1. **真 omni-modality。** 处理并生成文本、图像、视频、音频。
2. **走出自回归。** 把 vLLM 的内存管理伸到 **Diffusion Transformer（DiT）** 和其他并行生成。
3. **异构流水线。** 一次请求可以依次叫醒多只异构部件：多模态编码、AR 推理、基于扩散的多模态生成，等等。

## 架构

不只是套一层 wrapper。数据流在 vLLM 里外被重想过：完全可分离的流水线，生成的不同阶段可以动态分资源。图上收成三截：

- **Modality Encoders：** 多模态输入（ViT、Whisper 等）
- **LLM Core：** 用 vLLM 做自回归文本和 hidden states，一只或多只语言模型
- **Modality Generators：** 给 DiT 和其他 decoding head 做高性能 serving，吐出富媒体

阶段的名字，后文和后续笔记里常写成可分离的 **OmniStage**。

### 关键能力

![vllm omni user interface](../../../../assets/vllm/blog/serving/vllm-omni/02-vllm-omni-user-interface.png)

- **简单。** 会用 vLLM 就会用 Omni。接 Hugging Face 模型，提供 OpenAI 兼容 API。
- **灵活。** `OmniStage` 抽象把 Qwen-Omni、Qwen-Image 以及其他当时的 SOTA 收成同一套阶段故事。
- **性能。** 阶段流水、计算重叠：一只阶段在算，别的不必闲着。

![vllm omni pipeline async stage](../../../../assets/vllm/blog/serving/vllm-omni/03-vllm-omni-pipeline-async-stage.png)

他们拿 vLLM-Omni 对 Hugging Face Transformers 做了 omni-modal serving 的效率对照。数字在图上，正文没有另给一张表。

![vllm omni vs hf](../../../../assets/vllm/blog/serving/vllm-omni/04-vllm-omni-vs-hf.png)

## 路线图

当时写的是：扩模型、把高效推理再往前推，同时给 omni-modality 研究留一座稳的框架。

- **更多模型：** 开源 omni 模型和扩散 Transformer，随它们出现再接。
- **框架自适应：** 新的 omni 模型和执行形态进来时，框架跟着长——生产和研究共用地基。
- **更深地并进 vLLM：** 核心 omni 能力往上游合，让多模态在整个 vLLM 生态里成为一等公民。
- **扩散加速：** 并行（DP / TP / SP / USP…）、cache（TeaCache / DBCache…）、计算（量化 / sparse attention…）。后来的落地见 [omni-diffusion-cache](omni-diffusion-cache.md)。
- **完全分离：** 借 OmniStage，encoder / prefill / decode / generation 全拆开，吞吐上去、时延下来。
- **硬件：** 跟着 [hardware plugin](../architecture/hardware-plugin.md) 那套，把 backend 铺开，Omni 不绑死在一家卡上。

## 上手：安装和 serving

首发 **vllm-omni v0.11.0rc**，叠在 **vLLM v0.11.0** 上。原文**没有**把 `pip install` 写进正文，安装步骤指向文档：

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)

Serving 同样不在这篇里给一条万能 CLI，而是指向仓库的 examples：图像、音频、视频生成各有启动脚本。

- [examples](https://github.com/vllm-project/vllm-omni/tree/main/examples)

另有 Gradio，改善上手体验。文中的 demo 是 serving **Qwen-Image**：

![vllm omni gradio serving demo](../../../../assets/vllm/blog/serving/vllm-omni/05-vllm-omni-gradio-serving-demo.png)

CLI 旗标和 Python `Omni(...)` 的旋钮以当时文档和 examples 为准；后来的 TTS / cache / offload 笔记里会出现分阶段的命令，不要倒灌进这一篇的「第一条命令」。

## 社区

多模态 serving 的开头。原文请社区一起定下一截架构。

- 代码和文档：[GitHub](https://github.com/vllm-project/vllm-omni) · [Documentation](https://vllm-omni.readthedocs.io/en/latest/)
- Slack：[slack.vllm.ai](https://slack.vllm.ai) 的 `#sig-omni`
- 周会：每周二 **19:30 PDT**。[Join](https://tinyurl.com/vllm-omni-meeting)
