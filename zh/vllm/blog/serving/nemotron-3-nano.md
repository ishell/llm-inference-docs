---
source: https://vllm.ai/blog/2025-12-15-run-nvidia-nemotron-3-nano
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron 3 Nano：30B/3B hybrid MoE，Thinking Budget，NVFP4 后补

英文对照：[en/vllm/blog/serving/nemotron-3-nano.md](../../../../en/vllm/blog/serving/nemotron-3-nano.md)  
原文：https://vllm.ai/blog/2025-12-15-run-nvidia-nemotron-3-nano  
2025-12-15。署名 **NVIDIA Nemotron Team**。这篇是 day-0 上手，不是 kernel 深挖。上下文 **1M**。当时安装：`git+https://github.com/vllm-project/vllm.git@main`，加 `VLLM_USE_PRECOMPILED=1`。更大号：[Super](nemotron-3-super.md) / [Ultra](nemotron-3-ultra.md)。后来蒸馏的：[Lightning](nemotron-35-lightning.md)。前身 9B：[Nano 2](nemotron-nano2.md)。多模态亲戚：[Nano 2 VL](nemotron-nano-vl.md)、[Nano Omni](nemotron-omni.md)。Spark：[dgx-spark.md](dgx-spark.md)。Mamba 拆分：[hybrid-ssm.md](hybrid-ssm.md)。**4×** 画在图上，不是你的 SLA。

**1 月 28 日补丁。** NVFP4 checkpoint，开箱就能跑。Quantization-Aware Distillation (QAD) 保 NVFP4 精度；B200 相对 FP8-H100 声称 **4×** 吞吐。权重：[NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)。Brev：[launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-386KFyCvmg3y22JIf0q8BUh6jia)。

**原文 TL;DR：**

- Hybrid Mamba-Transformer MoE：**30B** 总参，**3B** 激活；上下文 **1M**；文本进文本出。
- 相对 [Nano 2](nemotron-nano2.md)：FFN → 稀疏 MoE；多数 attention → Mamba-2。声称最高约 **4×** token 吞吐。
- Day-0：BF16 和 FP8。NVFP4 + QAD 是一月补的。
- 当时 `--reasoning-parser deepseek_r1`（后来 Nemotron 3 系多用 `nemotron_v3`）。`VLLM_ATTENTION_BACKEND=FLASHINFER`。
- `--tool-call-parser qwen3_coder`。Thinking Budget。

## 为什么要这只

Nemotron 3 Nano 被写成当时新 Nemotron 3 家里的小号高效开源模，给 agentic AI。Hybrid Mamba-Transformer MoE 加 1M 上下文：多文档、长跑任务里要靠得住、吞吐要高。

Fully open：权重、数据、菜谱——在自己的基础设施上改、部署。

本地图（原文版权仍归原站；学习对照用）：

![figure 1](../../../../assets/vllm/blog/serving/nemotron-3-nano/01-figure_1.png)

**Figure 1。** “NVIDIA Nemotron 3 Sets a New Standard for Open Source AI。” Artificial Analysis Openness vs Intelligence Index 上最好看的那个象限。

点名的强项：coding、reasoning、agentic。点名的榜：SWE Bench Verified、GPQA Diamond、AIME 2025、Arena Hard v2、IFBench。正文 **没有** 分榜分数表。

## About Nemotron 3 Nano

- **Architecture：**
  - MoE + Hybrid Transformer-Mamba
  - Thinking Budget：声称用最少 reasoning token 换到合适精度
- **Accuracy：** coding、科学推理、解题、数学、instruction following、chat 领先
- **Model size：** 30B，3B 激活
- **Context length：** 1M
- **Model input：** 文本
- **Model output：** 文本
- **Supported GPUs：** NVIDIA RTX Pro 6000、DGX Spark、H100、B200
- **上手：**
  - 权重：[BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)、[FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
  - [用 vLLM 跑](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs)
  - 技术报告：[NVIDIA-Nemotron-3-Nano-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)

## 用 vLLM 跑优化过的推理

BF16 和 FP8。OpenAI-compatible API。

### Install vLLM

当时：从 `main` 装，走预编译：

```shell
VLLM_USE_PRECOMPILED=1 pip install git+https://github.com/vllm-project/vllm.git@main
```

### Serve the model

`VLLM_ATTENTION_BACKEND=FLASHINFER`。BF16 两条等价（`vllm serve` 或 `python -m vllm.entrypoints.openai.api_server`）。端口 **5000**。

页上的 serve 路径是 `nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16`——**Nano-3**，不是 Hugging Face 上的 `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`。两边都按原文留。

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER

# BF16
vllm serve --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1
```

或者：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser deepseek_r1
```

FP8 换模型名（页上用下划线 `--reasoning_parser`）：

```bash
vllm serve --model "nvidia/NVIDIA-Nemotron-Nano-3-30B-A3B-FP8" \
    --dtype auto \
    --trust-remote-code \
    --served-model-name nemotron \
    --host 0.0.0.0 \
    --port 5000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning_parser deepseek_r1
```

客户端（`api_key="null"`；先打 `reasoning_content` 再打 `content`）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

# Simple chat completion
resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about GPUs."}
    ],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.reasoning_content, resp.choices[0].message.content)
```

省事：[Brev cookbook launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs)。

## 专门 agent 任务：效率高，精度也带头

叠在 [Nano 2](nemotron-nano2.md) 的 hybrid Mamba-Transformer 上：FFN 换成稀疏 MoE；多数 attention 换成 Mamba-2。MoE：激活参数少一截，精度还更好；算力下去，真实延迟才跟得上。

Hybrid：声称最高 **4×** token 吞吐——想得快，又准。Thinking Budget：别 overthink；推理成本更低、更可预期。

![figure 2](../../../../assets/vllm/blog/serving/nemotron-3-nano/02-figure_2.png)

**Figure 2。** 开源推理模里吞吐更高、精度领先。正文 **没有** TPS 表。

NVIDIA 自己筛的数据。榜同引言：SWE Bench Verified、GPQA Diamond、AIME 2025、Arena Hard v2、IFBench——coding、reasoning、数学、instruction following。点名的企业场景：金融、网络安全、软件开发、零售。

![figure 3](../../../../assets/vllm/blog/serving/nemotron-3-nano/03-figure_3.png)

**Figure 3。** 开源小推理模里，常见学术榜领先。分数在图里，不在表里。

## Get started

开源权重、训练数据、菜谱：本地或云上微调、部署。

- 权重：[BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)、[FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)
- Cookbook / Brev：[launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-36ikINrMffBCbrtTVLr6MFcllcs)

想法墙：[nemotron.ideas.nvidia.com](http://nemotron.ideas.nvidia.com/?ncid=so-othe-692335)。订阅：[NVIDIA Nemotron](https://developer.nvidia.com/nemotron)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)、[X](https://x.com/NVIDIAAIDev)、[YouTube](https://www.youtube.com/@NVIDIADeveloper)，Discord 上的 [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper)。
