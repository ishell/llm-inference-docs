---
source: https://vllm.ai/blog/2025-04-05-llama4
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Llama 4：Scout 16E / Maverick 128E，iRoPE 1:3，v0.8.3+

英文对照：[en/vllm/blog/serving/llama4.md](../../../../en/vllm/blog/serving/llama4.md)  
原文：https://vllm.ai/blog/2025-04-05-llama4  
2025-04-05。署名 **The vLLM Team**。每 token 只激活 **1** expert（17B active）。前身 405B：[llama31.md](llama31.md)。集群 pitch 里点名的 V1：[v1-alpha.md](../architecture/v1-alpha.md)。分布式：[distributed-inference.md](distributed-inference.md)。`VLLM_DISABLE_COMPILE_CACHE=1` 是当时的开工旗。图上的 TPS 是他们的盘子，不是你的 SLA。

**原文 TL;DR：**

- Scout 17B-16E，Maverick 17B-128E。原生多模态（8–10 张图「效果不错」）。
- 8×H100：Scout `--max-model-len 1000000`（他们建议 `attn_temperature_tuning: true`）；Maverick-FP8 约 **430K**。
- 8×H200：Scout 3.6M，Maverick 1M。
- 多图：`--limit-mm-per-prompt image=10`（默认 1）。`--kv-cache-dtype fp8` 可把窗口再翻倍量级，他们说评测几乎不掉。
- iRoPE：无 RoPE 全局 attention 与分块局部 RoPE **1:3**。Maverick MMLU-Pro 官方 80.5，H100 FP8 **80.4**。

## Usage guide

[Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)：Scout 和 Maverick。装 `v0.8.3` 或更新：`pip install -U vllm`。CLI、[docker](https://docs.vllm.ai/en/latest/deployment/docker.html)、或 Python 的 [`LLM` class](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference)。Meta 的 1M 上下文 demo：[llama-cookbook notebook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynb)。

### 8× H100

Scout（最高 1M；`attn_temperature_tuning: true`）：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1000000 --override-generation-config='{"attn_temperature_tuning": true}'
```

Maverick-FP8（最高约 430K）：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 430000
```

### 8× H200

Scout（最高 3.6M）：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 3600000
```

Maverick（最高 1M）。页上 `--tensor-parallel-size 8` 后面少了一个 `\`：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8
  --max-model-len 1000000
```

**Multimodality。** 8–10 张图。默认每请求 1 张。OpenAI-compatible API 要 `--limit-mm-per-prompt image=10`。离线多图例子（v0.8.3）：[vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/v0.8.3/examples/offline_inference/vision_language_multi_image.py)。

**Performance。** 上面配置下 Scout-BF16 和 Maverick-FP8 的 output tok/s：

本地图（原文版权仍归原站；学习对照用）：

![perf](../../../../assets/vllm/blog/serving/llama4/01-perf.png)

**Figure。** Output tokens/s；数字在图里，不在表里。更多优化「还在路上」；架构 + 相对小的体积被写成已经能规模用。

**Tips for performance and long context：**

- `--kv-cache-dtype fp8` — 可用窗口可能翻倍，还有性能；声称评测几乎不掉。
- Scout 到 **10M**：多机 TP 或 PP。指南：[distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)。

**Other hardware and quantizations：**

- A100：BF16 验过。
- 单卡 H100 的 INT4 Scout：当时还在做。
- AMD MI300X：从 [源码编](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=rocm)，命令同上。

**Inference accuracy** 对照 Meta（lm-eval-harness），模型 [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)：

| | MMLU Pro | ChartQA |
|----------|---------|---------|
| Reported | 80.5 | 90 |
| H100 FP8 | 80.4 | 89.4 |
| AMD MI300x BF16 | 80.4 | 89.4 |
| H200 BF16 | 80.2 | 89.3 |

## Efficient architecture and cluster-scale serving

- **MoE：** Scout 16 expert，Maverick 128；激活 **17B**；每 token **一只** expert。
- **iRoPE：** 无 RoPE 的全局 attention 和带 RoPE 的分块局部 attention **1:3** 交错。局部层只看不重叠的 chunk——二次代价涨得慢。

V1 引擎：单机加速 + 原生 torch.compile。当时 Q2 roadmap：多机放大——拆开的集群 serving、expert parallelism、多机 data parallelism、集群级 prefill 分离。跟踪 issue：[vllm#15735](https://github.com/vllm-project/vllm/issues/15735)。

## Acknowledgement

Meta（架构、精度、压测）：Lucia (Lu) Fang, Ye (Charlotte) Qi, Lu Fang, Yang Chen, Zijing Liu, Yong Hoon Shin, Zhewen Li, Jon Swenson, Kai Wu, Xiaodong Wang, Shiyan Deng, Wenchen Wang, Lai Wei, Matthias Reso, Chris Thi, Keyun Tong, Jinho Hwang, Driss Guessous, Aston Zhang。

AMD MI300X：Hongxia Yang, Weijun Jiang。

vLLM 压测硬件来自 Nebius 和 NVIDIA。
