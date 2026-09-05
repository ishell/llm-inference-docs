---
source: https://vllm.ai/blog/2026-03-11-nemotron-3-super
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron 3 Super：120B/12B，1M 上下文，Thinking Budget

英文对照：[en/vllm/blog/serving/nemotron-3-super.md](../../../../en/vllm/blog/serving/nemotron-3-super.md)  
原文：https://vllm.ai/blog/2026-03-11-nemotron-3-super  
2026-03-11。署名 **NVIDIA Nemotron Team**。镜像 `vllm==0.17.1`。**4 × H100** BF16 示例。cookbook 才是完整菜谱；这篇是 day-0 骨架。同一套 hybrid 家族：[Nano](nemotron-3-nano.md) / [Ultra](nemotron-3-ultra.md)。蒸馏下来的亲戚：[Lightning](nemotron-35-lightning.md)。更早的 9B：[Nano 2](nemotron-nano2.md)。多模态亲戚：[Nano 2 VL](nemotron-nano-vl.md)、[Nano Omni](nemotron-omni.md)。桌上 Spark 的坑：[dgx-spark.md](dgx-spark.md)。Mamba 拆分 serving：[hybrid-ssm.md](hybrid-ssm.md)。Artificial Analysis / **4×** / **5×** 画在图上，不是你的 SLA。

**原文 TL;DR：**

- Hybrid Transformer-Mamba MoE：**120B** 总参，**12B** 激活；上下文最高 **1M**；文本进文本出。
- BF16、FP8、NVFP4。Blackwell 上 NVFP4 相对 H100 FP8 声称 **4×** 吞吐、精度持平。
- 相对上一只 Nemotron Super：图上最高约 **5×** 吞吐、**2×** 精度。
- MTP；Latent MoE（4 expert 的推理成本当 1）；Thinking Budget。
- `--kv-cache-dtype fp8` `--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`。

## 为什么要这只

Nemotron 3 Super 被写成 Nemotron 3 家的中号开源模，给复杂 multi-agent：规划、推理、多步执行。既要深度，又要能连续跑。

两条硬需求：

- **The "Context Explosion" Problem。** 多代理把历史、工具输出、推理步骤一遍遍重发，窗口很快炸。Super 的答：最高 **1 million** token 上下文——长记忆，少 goal drift。
- **The "Thinking Tax"。** 推理型代理用常规巨模又贵又慢。Hybrid MoE 声称最高 **4×** 吞吐，子任务不必每次付满模延迟和钱。

vLLM 是 serving 层：OpenAI-compatible API，高效率、高精度的 multi-agent 推理。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/nemotron-3-super/01-figure1.png)

**Figure 1。** Artificial Analysis：intelligence vs. openness。开源模里领先。Fully open：权重、数据、菜谱——在自己的基础设施上改、部署。

## About Nemotron 3 Super

- **Architecture：** MoE + Hybrid Transformer-Mamba
- 同尺寸档吞吐效率最高；相对上一只 Super 最高 **5×** 吞吐
- **Multi-Token Prediction (MTP)：** 一次前向预测若干未来 token；长文生成
- **Thinking Budget：** 声称用最少 reasoning token 换到合适精度

**Key specs：**

- **Accuracy：** 同尺寸档 Artificial Analysis Intelligence Index 领先；相对上一只 Super 最高 **2×**
- **Latent MoE：** 4 expert 的推理成本当 1
- **Model size：** 120B 总参，12B 激活
- **Context length：** 最高 1M
- **Model I/O：** 文本进、文本出
- **Supported GPUs：** B200、H100、DGX Spark、RTX 6000

**上手：**

- 权重：[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) — BF16、FP8、NVFP4
- 用 vLLM 跑
- 技术报告：[NVIDIA-Nemotron-3-Super-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## 用 vLLM 跑优化过的推理

BF16、FP8、NVFP4。Blackwell 上 NVFP4：相对 H100 FP8 声称 **4×** 吞吐、精度持平。FP8 / NVFP4 细节看 cookbook。

### Install vLLM

```bash
pip install vllm==0.17.1
```

### Serve the model

OpenAI-compatible API。下面按 **4 × H100** 写。硬件不同就改并行。FP8 和 NVFP4 看 cookbook。

```bash
# BF16
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --served-model-name nemotron \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

页上的客户端（`base_url` 端口 **5000**，`api_key="null"`，读的是 `reasoning_content`）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="null")

# Simple chat completion
resp = client.chat.completions.create(
    model="nemotron",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 bullet points about vLLM"}
    ],
    temperature=0.7,
    max_tokens=256,
)
print("Reasoning:", resp.choices[0].message.reasoning_content,
      "\nContent:", resp.choices[0].message.content)
```

和后来 Nemotron 3 帖对一下：客户端打的是 **:5000**（vLLM 默认 8000，除非你传 `--port`）；推理字段这里是 `reasoning_content`，Lightning / Ultra 常读 `reasoning`。省事：[cookbook](https://github.com/anushapant/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) 或 [NVIDIA Brev](https://brev.dev)。

## 多代理：效率带头，精度也带头

![figure2](../../../../assets/vllm/blog/serving/nemotron-3-super/02-figure2.png)

**Figure 2。** Artificial Analysis：intelligence vs. efficiency，同尺寸开源模。声称精度领先、效率更高。正文 **没有** TPS / TTFT 表。

## Get started

开源权重、数据、菜谱：从工作站到云都能微调、部署。

- 权重：[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) — BF16、FP8、NVFP4
- Cookbook：[Nemotron-3-Super/vllm_cookbook.ipynb](https://github.com/anushapant/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) 和 [Brev](https://brev.dev)
- 技术报告：[NVIDIA-Nemotron-3-Super-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

页上的订阅入口：[NVIDIA news](https://www.nvidia.com/en-us/preferences/email-signup/)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/company/nvidia/)、[X](https://x.com/NVIDIAAI)、[YouTube](https://www.youtube.com/nvidia)，Discord 上的 Nemotron：[invite](https://discord.gg/nvidia)。

## Acknowledgement

感谢把 Nemotron 3 Super 接到 vLLM 的所有人。

NVIDIA：Nirmal Kumar Juluru, Anusha Pant。

vLLM team and community：Roger Wang, Michael Goin, Thomas Parnell, Kevin Luu, Robert Shaw, Tyler Michael Smith。
