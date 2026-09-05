---
source: https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron 3.5 Lightning：常开 agent 的小炉子

英文对照：[en/vllm/blog/serving/nemotron-35-lightning.md](../../../../en/vllm/blog/serving/nemotron-35-lightning.md)  
原文：https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm  
2026-08-10。署名 **NVIDIA Nemotron Team and vLLM Team**。镜像 `vllm/vllm-openai:v0.27.1`。从 [Nemotron 3 Ultra](nemotron-3-ultra.md) **蒸馏**下来；和 [Nano](nemotron-3-nano.md) / [Super](nemotron-3-super.md) 同一套 hybrid 家族。更早的 9B：[Nano 2](nemotron-nano2.md)。多模态亲戚：[Nano 2 VL](nemotron-nano-vl.md)、[Nano Omni](nemotron-omni.md)。桌上 Spark 的坑：[dgx-spark.md](dgx-spark.md)。Pareto / PinchBench 是页上的演示，不是你的 SLA。

角色写死了：前沿模型管编排，这只跑碎步。架构与 Nemotron 3 几乎同构，差在权重和投机栈。**不是新引擎。**

**原文 TL;DR：**

- Hybrid MoE **30B 总参 / 3B 激活**，文本进文本出，上下文最高 **1 million** tokens。
- 从 Ultra 蒸馏；为流行 agent harness 训过；允许 post-training。
- Day-0：**BF16** 和 **NVFP4**。投机：**MTP**、**DFlash**、**DSpark**。
- 相对同尺寸开源模，声称最高 **4×** 吞吐；PinchBench：完成 1 万 agent 任务，相近准确率下最高快 **30%**。
- 低延迟：H100 / H200 / DGX Spark 上用 **DSpark**。当时冲吞吐：**关掉**投机。

## 两模型切分里的那只小的

常开 agent：本地私人助手，到机房、云上的高并发碎步。原文点名的强项：coding、tool use、instruction following、multi-turn intelligence。

现代 agent 平台越来越把活切开。难的规划和编排交给 frontier；频次高、范围清楚的步骤交给小模型。Lightning 就是第二个角色，又不丢掉真实 agent 工作流要的能力。

页上两条硬需求：

- **规模上要快。** Agent 系统大量时间耗在又小又多的步骤。Hybrid MoE（每 token 只激活 3B / 30B）再加 multi-token prediction；声称相对同尺寸开源模最高 **4×** 吞吐。
- **智能要能改。** 组织黑话、政策、工具、多轮上下文。为流行 harness 训过，可以 post-train。点名的领域：金融与风控自动化、网络安全调查、电信运维、零售、本地私人助手。

vLLM 给出 OpenAI-compatible API，现成的 agent 框架、本地应用、企业自动化可以直接接。

## TL;DR: About Nemotron 3.5 Lightning

- **Architecture：** Hybrid MoE
- **Model size：** 30B 总参，3B 激活
- **Context length：** 最高 1 million tokens
- **Modalities：** 文本进、文本出
- **Speculative decoding：** MTP、DFlash、DSpark
- **Reasoning：** 每个请求可开可关；reasoning-token budget 可配
- **Training：** 从 Nemotron 3 Ultra 蒸馏；为流行 agent harness 训过
- **Customization：** 开源模型、开源数据，可在专门工作流上 post-train
- **Availability at launch：** BF16、NVFP4
- **Deployment targets：** DGX Spark、DGX Station、RTX PRO、RTX、Jetson、H100、H200、A100、L40S、B200/GB200、B300/GB300
- **上手：**
  - 权重：[BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)、[NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
  - cookbook：[Nemotron-3.5-Lightning/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb)

## 用 vLLM 跑高吞吐推理

卡面很宽。vLLM 是 serving 层：continuous batching、prefix caching、投机解码、OpenAI-compatible API。

BF16 是老实的基线。NVFP4 发布时就有，给能吃低精度的环境。

### Install vLLM

```bash
docker pull vllm/vllm-openai:v0.27.1

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.27.1
```

### Serve the model

这条假定 **1 × H100**。

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --moe-backend humming \
  --linear-backend humming \
  --mamba-ssu-algorithm horizontal \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --reasoning-parser nemotron_v3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --host 0.0.0.0 \
  --port 8000
```

页上的客户端：`temperature=1.0`，`top_p=0.95`，`max_tokens=1024`；读的是 `choice.message.reasoning` 和 `choice.message.content`（不是后来有的 `reasoning_content`）。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is vLLM?"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

choice = response.choices[0]
print("Reasoning:", choice.message.reasoning)
print("Content:", choice.message.content)
```

## 长跑 agent：三条投机

三条：**MTP**、**DFlash**、**DSpark**。Draft-and-verify；声称保住目标模型的输出质量。

- **MTP：** 模型内嵌的轻量预测头，一次提若干未来 token。
- **DFlash：** 扩散 drafter 并行吐一整块候选。要单独的 draft checkpoint，和 MTP 分开配。
- **DSpark：** 带置信度的半自回归草稿；夹在 MTP（全自回归）和 DFlash（全扩散）中间。原文说三条里 **DGX Spark 上它最好**。

架构几乎同构，性能活大多落在 runtime。他们往上游送的：

- **DSpark integration：** 接到 vLLM 和 Nemotron 模型定义里，和 MTP、DFlash 并列三条。
- **Quantized DSpark draft head：** 草稿头收到 W4A16，内存和每步延迟下去，acceptance rate 不伤——内存在 DGX Spark 这种地方最要紧。
- **拿掉 sync、打开 async scheduling：** draft-and-verify 环路上的 host-device sync 去掉；下一 batch 在当前还在跑时就能准备。
- **MoE / linear 的 W4A16 backend：** 默认 Marlin 换成 Hopper 上的 **Humming**；Nemotron 非门控 ReLU² MoE 用 W4A16 GEMM，大约 **+20%** 吞吐；同一套方子伸到 dense linear。
- **ReplaySSM** 接 Mamba2：砍 hybrid 里 recurrent 路径的每步开销。

页上的用法：低延迟 → H100 / H200 / Spark 上 DSpark。当时要最大吞吐 → **不要**投机。

### Multi-token prediction

内建 MTP 头提未来 token，目标模型核实，长回答少走几步串行 Decode。

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.moe_backend flashinfer_cutlass \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

### DFlash

专用扩散 draft 提出线性 token 块，目标并行核实。

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DFlash 草稿：[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash)。

### DSpark

混合 speculator。原文：三条里 Spark 上它最好。

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

DSpark 草稿：[nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)。

## Local Deployment on NVIDIA DGX Spark

单用户本地开发的起点（NVFP4 + DSpark；`cudagraph_capture_sizes` 按原文整串抄）：

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --compilation_config.cudagraph_capture_sizes '[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 1024, 2048, 4096, 8192]' \
  --speculative_config.num_speculative_tokens 3 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

**Figure 1（未抓图；按页上 caption）。** DGX Spark 上各条投机的 Pareto。配置：Prefix **32K**，再 **10** 轮 **2k** input / **1k** output。轴上的 TPS 数字在图里，正文没表。

## Deploy on NVIDIA H100

单用户本地开发的起点。这条 **没有**投机（和「今天冲吞吐就关投机」对齐），MoE / linear 走 **Humming**：

```bash
vllm serve --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --moe-backend humming \
  --linear-backend humming \
  --max-num-seqs 256 \
  --trust-remote-code \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --mamba-ssu-algorithm horizontal \
  --reasoning-parser nemotron_v3
```

**Figure 2（未抓图；按页上 caption）。** 同一套 Pareto，换 H100。同样：Prefix **32K**，再 10 轮 2k / 1k。正文没有 TPS / TTFT 表。

## Local Deployment on NVIDIA Jetson

单用户本地开发的起点。这条也没有投机：

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align
```

## 专门 agent 任务：准确率和效率

每 token 只点亮 3B / 30B；MTP 少走串行步。合起来：相对同尺寸开源模最高 **4×** 吞吐。

准确率：从 Ultra 蒸馏，再在流行 harness 上训，agent 生产力、coding、tool use、instruction following、长上下文推理。页上 **没有** 分榜分数表。

**Figure 3（未抓图；按页上 caption）。** PinchBench 准确率对上完成 **10,000** 任务的时间。声称在相近准确率下，agent 任务最高快 **30%**，站在效率前沿。

## Summary

本地、边缘、机房、云都能放的可定制 agent 智能。30B hybrid MoE、3B 激活、最高 1M 上下文、可控 reasoning。收束句写的投机是 MTP **或** DFlash，**没点 DSpark**——正文三条都有。

Day-0 vLLM：OpenAI-compatible 栈，接本地助手、harness、专门的企业工作流。

上手链接同 TL;DR。页上的订阅入口：[NVIDIA Nemotron](https://developer.nvidia.com/nemotron)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)、[X](https://x.com/NVIDIAAIDev)、[YouTube](https://www.youtube.com/@NVIDIADeveloper)，Discord 上的 [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper)。

## Acknowledgement

NVIDIA：Nirmal Kumar Juluru, Anusha Pant, Amir Klein, Faradawn Yang, Nave Assaf, Ryan Stewart, Alex Steiner, Bita Rouhani。

## FAQs

### 相对 Nemotron 3 Nano 新在哪？

Nano 已经把 hybrid Mamba-Transformer MoE 立住：30B / 3B、1M 上下文、可控 reasoning。Lightning 叠在这上面。原文写 **four important ways**，下面却只列了 **三条**：

- **Frontier-model distillation：** 从 Ultra 把能力蒸进小得多的部署体积。
- **Agent-harness optimization：** 流行 harness 和多轮；coding、tool use、instruction following、专门任务。
- **Speculative decoding：** MTP、DFlash、DSpark；并行草稿再核实。

第四条抓下来的正文里没有。结果句：更多 agent 任务，更准，更少时间。
