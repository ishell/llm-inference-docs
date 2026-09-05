---
source: https://vllm.ai/blog/2025-10-31-run-multimodal-reasoning-agents-nvidia-nemotron
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron Nano 2 VL：12B 视频/文档，EVS 砍冗余帧

英文对照：[en/vllm/blog/serving/nemotron-nano-vl.md](../../../../en/vllm/blog/serving/nemotron-nano-vl.md)  
原文：https://vllm.ai/blog/2025-10-31-run-multimodal-reasoning-agents-nvidia-nemotron  
2025-10-31。署名 **NVIDIA Nemotron Team**。这篇是 day-0 上手，当时走 **nightly** 轮子——没有钉死 `vllm==`。上下文 **128K**。Encoder：[CRADIOH-V2](https://huggingface.co/nvidia/C-RADIOv2-H)；砍 token：[Efficient Video Sampling (EVS)](https://arxiv.org/abs/2510.14624)；LLM 骨架：[Nano 2](nemotron-nano2.md)。后继加音频：[Nano Omni](nemotron-omni.md)。同一套 hybrid 家族：[Nano](nemotron-3-nano.md) / [Super](nemotron-3-super.md) / [Ultra](nemotron-3-ultra.md) / [Lightning](nemotron-35-lightning.md)。Mamba 拆分 serving：[hybrid-ssm.md](hybrid-ssm.md)。这是 vLLM 上的 **perception VLM**，不是 [vLLM-Omni](vllm-omni.md) 那条扩散/TTS 栈。VLM 套榜均分 **74** vs 当时顶 VL **64.2** 是页上的演示，不是你的 SLA。

**原文 TL;DR：**

- 开源 VLM，给视频理解和文档智能。Hybrid Transformer–Mamba；**12B**；上下文 **128k**；多图文档 / 视频 / 文本进，文本出。
- **EVS** 砍冗余帧，同样算力塞更长视频。`--video-pruning-rate 0` 是**不砍**。
- Day-0：BF16、FP8、FP4-QAD。量化起服：`--quantization modelopt` / `modelopt_fp4`。
- 客户端系统提示 `/no_think` 关思考。
- Hugging Face slug 是 `Nemotron-Nano-12B-v2-VL-*`。页上 FP8 / FP4 的 **serve** 路径是 `Nemotron-Nano-VL-12B-V2-*`。两边都按原文留。

## 为什么要这只

[NVIDIA Nemotron Nano 2 VL](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-BF16) 被写成开源 VLM，视频理解和文档智能搁同一只里。Hybrid Transformer–Mamba：Transformer 的推理，Mamba 的算力效率；声称吞吐更高，多模态推理精度还是「当时 SOTA」，多图输入也更快。

[EVS](https://arxiv.org/abs/2510.14624) 是点名的 token 压缩：丢掉冗余视频 token，同样效率多吃几段视频。

vLLM 是 serving 层：OpenAI-compatible API，BF16 / FP8 / FP4，并发请求。

## 视频和文档：一只高效多模态

NVIDIA 自己筛的多模态数据。点名的榜：MMMU、MathVista、AI2D、OCRBench、OCRBench-v2、OCR-Reasoning、ChartQA、DocVQA、Video-MME。点名的能力：多模态 reasoning、认字、图表推理、视觉问答。点名的企业活：从视频、文档、表格、图表里抽东西、看懂。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/nemotron-nano-vl/01-figure1.png)

**Figure 1。** 视频理解和文档智能各榜领先。分数在图里，不在表里。

### Improving Efficiency with EVS

声称：吞吐上去、响应更快，精度不丢。EVS 砍冗余帧，语义还在，同样预算能吃更长视频。营销句：会议、培训、客服通话那种论小时的片子，几分钟看完。

![figure2](../../../../assets/vllm/blog/serving/nemotron-nano-vl/02-figure2.png)

**Figure 2。** EVS 不同 token-drop 阈值下，Video-MME 和 LongVideo 的精度。轴上的数字在图里。正文 **没有** TPS / TTFT 表。

## About Nemotron Nano 2 VL

- **Architecture：**
  - Vision encoder：[CRADIOH-V2](https://huggingface.co/nvidia/C-RADIOv2-H)（Hugging Face 上是 `C-RADIOv2-H`）
  - Token compression：Efficient Video Sampling
  - LLM：hybrid Transformer–Mamba，骨架是带 reasoning 的 [Nemotron Nano 2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- **Accuracy：**
  - OCRBench v2 领先
  - 下面这串榜均分 **74**，当时「current top VL model」**64.2**：MMMU、MathVista、AI2D、OCRBench、OCRBench-v2、OCR-Reasoning、ChartQA、DocVQA、Video-MME——营销对照，复测用 cookbook
- **Model size：** 12B
- **Context length：** 128k
- **Model input：** 多图文档、视频、文本
- **Model output：** 文本
- **上手：**
  - 权重：[BF16](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-BF16)、[FP8](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-FP8)、[FP4-QAD](https://huggingface.co/nvidia/Nemotron-Nano-12B-v2-VL-FP4-QAD)
  - 用 vLLM 跑
  - 技术报告：[NVIDIA-Nemotron-Nano-V2-VL-report.pdf](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-V2-VL-report.pdf)

页上还链了数据说明：[nemotron-vlm-dataset-v2](https://huggingface.co/blog/nvidia/nemotron-vlm-dataset-v2)。

## 用 vLLM 跑优化过的推理

BF16、FP8、FP4。OpenAI-compatible API。

### Install vLLM

当时：`uv` 装 nightly，允许 prerelease：

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow
```

### Deploy and query the inference server

三条。`--video-pruning-rate 0` = 不走 EVS 砍帧。vLLM 默认端口 **8000**（下面客户端打 `localhost:8000`）。

页上 FP8 / FP4 的 **serve** 仓库名，和 Hugging Face 上手 slug（`Nemotron-Nano-12B-v2-VL-FP8` / `…-FP4-QAD`）**对不上**。两边都按原文留。

```bash
vllm serve nvidia/Nemotron-Nano-12B-v2-VL-BF16 --trust-remote-code --dtype bfloat16 --video-pruning-rate 0

# FP8
vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP8 --trust-remote-code --quantization modelopt --video-pruning-rate 0

# FP4
vllm serve nvidia/Nemotron-Nano-VL-12B-V2-FP4-QAD --trust-remote-code --quantization modelopt_fp4 --video-pruning-rate 0
```

客户端（`api_key="null"`；系统提示 `/no_think`；`temperature=0.0`，`max_tokens=1024`；图走 `image_url`）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="null")
# Simple chat completion
resp = client.chat.completions.create(
    model="nvidia/Nemotron-Nano-12B-v2-VL-BF16",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": [
            {"type": "text", "text": "Give me 3 interesting facts about this image."},
            {"type": "image_url", "image_url": {"url": "https://blogs.nvidia.com/wp-content/uploads/2025/08/gamescom-g-assist-nv-blog-1280x680-1.jpg"}
            }
            ]},
    ],
    temperature=0.0,
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

更多例子：[Nemotron-Nano2-VL/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Nano2-VL/vllm_cookbook.ipynb)，还有 vLLM 菜谱 [Nemotron-Nano-12B-v2-VL](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-Nano-12B-v2-VL.html)。

## Get started

上手三件套同规格表：BF16 / FP8 / FP4-QAD 权重、cookbook + recipe、技术报告。

想法墙：[nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335)。订阅：[NVIDIA Nemotron](https://developer.nvidia.com/nemotron)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)、[X](https://x.com/NVIDIAAIDev)、[YouTube](https://www.youtube.com/@NVIDIADeveloper)，Discord 上的 [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper)。
