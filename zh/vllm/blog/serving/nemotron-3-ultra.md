---
source: https://vllm.ai/blog/2026-06-04-nemotron-3-ultra-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Nemotron 3 Ultra：550B/55B hybrid MoE，训练 rollout 也走 vLLM

英文对照：[en/vllm/blog/serving/nemotron-3-ultra.md](../../../../en/vllm/blog/serving/nemotron-3-ultra.md)  
原文：https://vllm.ai/blog/2026-06-04-nemotron-3-ultra-vllm  
2026-06-04。署名 **NVIDIA Nemotron Team**。镜像 `vllm/vllm-openai:v0.22.0`。**8 × B200** 示例。cookbook 才是完整菜谱；这篇是 day-0 骨架。蒸馏下去的亲戚：[Lightning](nemotron-35-lightning.md)。同一家：[Super](nemotron-3-super.md)、[Nano](nemotron-3-nano.md)、[Nano 2](nemotron-nano2.md)、[Nano 2 VL](nemotron-nano-vl.md)、[Nano Omni](nemotron-omni.md)。训练侧 rollout 挨着 [native-rl.md](native-rl.md)。Mamba 拆分 serving：[hybrid-ssm.md](hybrid-ssm.md)。**30%** 成本、「领先吞吐」画在图上，不是可复现 SLA。

**Hero（封面；未抓图；按页上路径）。** 原文 `/assets/figures/2026-nemotron-3-ultra/hero.png`。caption 没有数字。

**原文 TL;DR：**

- Hybrid Transformer-Mamba MoE：**550B** 总参，**55B** 激活；上下文最高 **1M**；文本进文本出。
- BF16 和 NVFP4。NVFP4 在 Blackwell 能跑；同一份 NVFP4 也被写成靠专用量化 kernel 在 Hopper **和** Blackwell 上都能用。
- 训练也用 vLLM：多机 rollout 和评估；NeMo RL 里当 generation backend，接 NeMo Gym 做多步 / 多轮环境。

## 为什么要这只

[Nemotron 3 Ultra](https://blogs.nvidia.com/blog/nvidia-gtc-taipei-computex-2026-news/#nemotron-3-ultra) 被写成前沿级推理，给长跑自治 agent：复杂编排、coding、深研究、企业自动化——规划、调工具、从错误里爬起来、在很长的上下文里想。

持久 agent 不是答完一道题就走。它们搜、写代码、跑测试、看失败、协调工具、评估证据，沿很长的任务地平线接着干。既要推理深度，又要推理快到能部署。

两条：

**Fast Task Completion。** 同一段墙上时间里多走几步推理。Hybrid Transformer-Mamba MoE、multi-token prediction、NVIDIA 优化过的推理精度。

**Advanced Agentic Reasoning。** 架构规划、多步调试、源评估、合规审阅、设计核验。为 reasoning、tool use、instruction following 做过 post-train。

vLLM 在训练环里：高吞吐多机推理，伺候 rollout 和评估。在 [NeMo RL](https://github.com/nvidia-nemo/rl) 里当 generation backend——采样、可扩展推理、接 [NeMo Gym](https://github.com/NVIDIA-NeMo/gym)。Nemotron 团队还用 vLLM 盯每一段训练有没有把模型往对的方向推。

## TL;DR: About Nemotron 3 Ultra

- **Architecture：** MoE + Hybrid Transformer-Mamba
  - Model size：550B 总参，55B 激活
  - Context length：最高 1M tokens
  - Modalities：文本进、文本出
- **Efficiency：** NVFP4 和 BF16 高吞吐。NVFP4 checkpoint 能在 Blackwell 上跑。
- **Reasoning：** 长跑自治 agent、tool calling、coding、深研究、编排
- **Training：** 多环境强化学习 post-train
- **Deployment：** 开源权重、数据、菜谱
- **Supported GPUs：**
  - BF16：**8×** GB200/B200/GB300/B300，**16×** H100，**8×** H200
  - NVFP4：**4×** GB200/B200/GB300/B300，**8×** H100
- **上手：**
  - 权重：[BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)、[NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
  - cookbook：[Nemotron-3-Ultra/vllm_cookbook.ipynb](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb)
  - 技术报告：[NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)

## 用 vLLM 跑优化过的 agent 推理

BF16 和 NVFP4。OpenAI-compatible API，接 agent 框架、编码系统、研究流水线、企业自动化。

省事：cookbook，或 NVFP4 的 NVIDIA Brev [launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-3EPQRUP8Sl27sxp1fMvXt3Lor8T)。

### Install vLLM

```bash
docker pull vllm/vllm-openai:v0.22.0

docker run --rm -it --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.22.0
```

### Serve the model

按 **8× B200** 写的。硬件不同就改并行相关旗。NVFP4 细节看 cookbook。

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

vllm serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --enable-flashinfer-autotune \
  --async-scheduling \
  --speculative_config.method mtp \
  --speculative_config.num_speculative_tokens 5 \
  --mamba-backend triton \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

和规格表对一下：`--max-model-len 262144` **不是** 规格里的 1M；这里 `--mamba-backend triton`，Lightning 的 H100 菜谱用 `flashinfer`。MTP `num_speculative_tokens` 是 **5**（Lightning 的 MTP 例子是 **3**）。

客户端 `api_key="EMPTY"`，推理字段用 `getattr(msg, "reasoning", None)`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 bullet points about vLLM"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)

msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning", None))
print("Content:", msg.content)
```

NVFP4 部署还是看 [cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb)。

## 长跑 agent 的高吞吐推理

Figure 1、2、3：agent 生产力、instruction following、长上下文准确率领先；吞吐领先；相对其他领先开源模省 **30%** 成本。正文 **没有** TPS / TTFT 表。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/nemotron-3-ultra/02-figure1.svg)

![figure2](../../../../assets/vllm/blog/serving/nemotron-3-ultra/03-figure2.svg)

![figure3](../../../../assets/vllm/blog/serving/nemotron-3-ultra/04-figure3.svg)

**Figure 1。** 开源模里，agent 生产力、coding、instruction following 的 agentic 榜领先。

**Figure 2。** 准确率和吞吐都领先的象限。配置：vLLM，**10k/2k** ISL/OSL，**BS 1**。

**Figure 3。** 成本最高省 **30%**，站在成本效率前沿。

为了缓高容量推理模常见的效率–准确率对打，原文点名的架构：

- **Post-Trained for Agent Harness。** [NeMo RL](https://github.com/nvidia-nemo/rl) 和 [Gym](https://github.com/NVIDIA-NeMo/gym) 跨很多 harness。优化的是领先的开源 agent harness，不是单轮聊天：规划、调工具、读观察、把活交给子代理、校验输出、多轮里从错误恢复。
- **Hybrid Mamba-Transformer。** Mamba 扛长上下文的序列效率；Transformer 在大窗口里精确召回具体事实。
- **Latent MoE。** 更省的 expert 路由，覆盖推理、代码、工具、领域逻辑。
- **Multi-Token Prediction (MTP)。** 一次前向预测多个未来 token；长输出和多轮工作流的吞吐。
- **NVFP4 precision。** 同一份 NVFP4 checkpoint，靠专用量化 kernel 在 Hopper 和 Blackwell 上都能跑。

## Summary

开源前沿推理模，给长跑自治 agent：高吞吐推理、长上下文、工具、开放部署。

上手三件套同 TL;DR。页上的订阅入口：[NVIDIA Nemotron](https://developer.nvidia.com/nemotron)，NVIDIA AI 的 [LinkedIn](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)、[X](https://x.com/NVIDIAAIDev)、[YouTube](https://www.youtube.com/@NVIDIADeveloper)，Discord 上的 [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) / [invite](https://discord.com/invite/nvidiadeveloper)。

## Acknowledgement

感谢把 Nemotron 3 Ultra 接到 vLLM 的所有人。

NVIDIA：Nirmal Kumar Juluru, Anusha Pant, Alex Steiner, Tomer Asida, Daniel Afrimi, Shaun Kotek, Roi Koren, Daniel Serebrenik, Amir Klein, Omer Ullman Argov, Netanel Haber, Amit Zuker, Shahar Mor, Tomer Bar Natan。

vLLM team and community：Michael Goin, Kaichao You, Yongye Zhu, Roger Wang, Simon Mo, Woosuk Kwon, Yasong Wang, Nick Hill, Zachary Xi。
