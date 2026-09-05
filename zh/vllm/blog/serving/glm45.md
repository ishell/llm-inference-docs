---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# GLM-4.5 / 4.5V：hybrid thinking，parser 叫 glm45，当时不要 V0

英文对照：[en/vllm/blog/serving/glm45.md](../../../../en/vllm/blog/serving/glm45.md)  
原文：https://vllm.ai/blog/2025-08-19-glm45-vllm  
2025-08-19。署名 **Yuxuan Zhang**。当时 nightly，外加 `transformers-v4.55.0-GLM-4.5V-preview`。355B/32B 与 Air 106B/12B。FP8/BF16 同一条 serve。后续 5.2 生产见 [glm52-b300.md](glm52-b300.md)。**63.2** / **59.8** 是页上的分，不是你的 SLA。

**原文 TL;DR：**

- `--tool-call-parser glm45` `--reasoning-parser glm45`。关思考：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。
- 4.5V：`--allowed-local-media-path`，视频 `num_frames: -1`。
- 8×H100 跑满血 4.5 当时可能 `--cpu-offload-gb 16`。FlashInfer 不顺就 `VLLM_ATTENTION_BACKEND=XFORMERS`。
- **V0 不支持。** Grounding：`<|begin_of_box|>` 框坐标，xy 按宽高归一再 ×1000。

## Introduction

[GLM](https://aclanthology.org/2022.acl-long.26/) 出自 Zhipu.ai（现 [Z.ai](https://z.ai/)）。和 vLLM 的合作早到 [ChatGLM](https://github.com/zai-org/ChatGLM-6B)。这篇：[GLM-4.5](https://arxiv.org/abs/2508.06471) 和 [GLM-4.5V](https://arxiv.org/abs/2507.01006)，给智能代理——当时 Hugging Face 热榜。

GLM-4.5：**355B** 总参 / **32B** 激活。GLM-4.5-Air：**106B** / **12B**。推理、编码、代理能力收成一套。

两只都是 hybrid reasoning：**thinking**（复杂推理 + 工具）和 **non-thinking**（立刻答）。

页上 12 项工业榜：GLM-4.5 **63.2**（专有+开源里第 3）；Air **59.8**，效率更高。

**Figure bench_45（未抓图；原文远程图）。** Caption：GLM-4.5 / Air 综合分对照。来源 `https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/bench.png`。

GLM-4.5V 叠在 Air 上；接着 GLM-4.1V-Thinking。声称同尺寸 42 项公开视觉语言榜 SOTA。

**Figure bench_45v（未抓图；原文远程图）。** Caption：GLM-4.5V 对照同尺寸 VLM。来源 `https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_45v.jpeg`。

仓库：[GLM-4.5](https://github.com/zai-org/GLM-4.5)、[GLM-V](https://github.com/zai-org/GLM-V)。这篇：vLLM 在 NVIDIA Blackwell 和 Hopper 上加速。

## Installation

当时 `main`。Nightly vLLM + 一份 preview Transformers：

```shell
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install transformers-v4.55.0-GLM-4.5V-preview
```

## Usage

FP8 和 BF16：**同一条** `vllm serve`。

GLM-4.5（Air 示例，TP 4）：

```shell
vllm serve zai-org/GLM-4.5-Air \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice
```

GLM-4.5V：

```shell
vllm serve zai-org/GLM-4.5V \
     --tensor-parallel-size 4   \
     --tool-call-parser glm45   \
     --reasoning-parser glm45   \
     --enable-auto-tool-choice  \
     --allowed-local-media-path / \
     --media-io-kwargs '{"video": {"num_frames": -1}}'
```

### Important notes

- 推理包在 `reasoning_content` 里；`content` 只留终答。关掉：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。
- 8× H100 跑满血 GLM-4.5 显存不够 → `--cpu-offload-gb 16`。
- `flash_infer` 出问题：临时换 `VLLM_ATTENTION_BACKEND=XFORMERS`。或设 `TORCH_CUDA_ARCH_LIST='9.0+PTX'` 继续用 flash_infer——arch 字符串看卡。
- **vLLM V0 不支持**（原文：“vLLM v0 is not support our model”）。

### Grounding in GLM-4.5V

问物体在哪；模型一步步想，再吐 bounding box。示例：

- Help me to locate `<expr>` in the image and give me its bounding boxes.
- Please pinpoint the bounding box `[[x1,y1,x2,y2], …]` in the image as per the given description. `<expr>`

框是 `[x1, y1, x2, y2]`（左上 / 右下）。每个值：x 按图宽、y 按图高归一，再 **×1000**。

特殊 token `<|begin_of_box|>` / `<|end_of_box|>` 标出框。括号写法可以变（`[]`、`[[]]`、`()`、`<>`）；意思一样。

## Cooperation with vLLM and GLM Team

发模之前：vLLM 和 GLM 一起把启动问题拧平，让 `main` 在开源当天就吃满 GLM-4.5 系列。

## Acknowledgement

vLLM：Kaichao You, Simon Mo, Zifeng Mo, Lucia Fang, Rui Qiao, Jie Li, Ce Gao, Roger Wang, Lu Fang, Wentao Ye, Zixi Qi。
