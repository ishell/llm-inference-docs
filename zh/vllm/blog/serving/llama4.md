---
source: https://vllm.ai/blog/2025-04-05-llama4
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Llama 4：Scout 16E / Maverick 128E，iRoPE 1:3，v0.8.3+

英文对照：[en/vllm/blog/serving/llama4.md](../../../../en/vllm/blog/serving/llama4.md)  
原文：https://vllm.ai/blog/2025-04-05-llama4  
2025-04-05。**The vLLM Team**。需要 **v0.8.3+**。Scout **17B-16E**，Maverick **17B-128E**。原生多模态（他们说 8–10 张图效果还行）。Meta 的长上下文 demo：[llama-cookbook notebook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynb)。Docker / `LLM` 类是备选；下面是 CLI。开工旗他们写成 `VLLM_DISABLE_COMPILE_CACHE=1`。

安装：`pip install -U vllm`。

## 他们印的硬件窗口

**8×H100：** Scout 到 **1M**；Maverick 大约 **430K**。

Scout：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1000000 --override-generation-config='{"attn_temperature_tuning": true}'
```

Maverick FP8：

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 430000
```

**8×H200：** Scout **3.6M**；Maverick **1M**。

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 3600000
```

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8
  --max-model-len 1000000
```

（第二条在原文 `8` 后面少了 `\`；两面旗标都要留。）

## 多模态

默认服务：**每请求 1 张图**。到 10 张：`--limit-mm-per-prompt image=10`。离线多图例子：[vision_language_multi_image.py @ v0.8.3](https://github.com/vllm-project/vllm/blob/v0.8.3/examples/offline_inference/vision_language_multi_image.py)。

## 性能图

本地图（原文版权仍归原站；学习对照用）：

![perf](../../../../assets/vllm/blog/serving/llama4/01-perf.png)

上面那套配置下 Scout-BF16 与 Maverick-FP8 的 output tok/s。正文没有数字表。

## 提示

- `--kv-cache-dtype fp8` — 他们说可用上下文大约能 **翻倍**，速度也有帮助；评测里精度几乎不掉
- Scout **到 10M**：多机 TP 或 PP；[distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

## 别的硬件 / 量化

- A100：BF16 验过
- 单卡 H100 上的 INT4 Scout：当时还在做
- AMD MI300X：从源码编 vLLM（ROCm GPU 安装），命令同上

## 他们印的精度（Maverick Instruct）

| | MMLU Pro | ChartQA |
|---|---|---|
| Reported | 80.5 | 90 |
| H100 FP8 | **80.4** | **89.4** |
| AMD MI300x BF16 | **80.4** | **89.4** |
| H200 BF16 | **80.2** | **89.3** |

lm-eval-harness 对 Meta 报告。

## 他们强调的架构

- MoE：Scout 16 expert，Maverick 128；**激活 17B**；**每 token 一只 expert**
- **iRoPE：** 无 RoPE 的全局 attention 与带 RoPE 的分块局部 attention 按 **1:3** 交错。局部层只看互不重叠的 chunk——平方代价不跟着全长涨。

点名 V1 引擎和 torch.compile。Q2 roadmap 指向 [issue 15735](https://github.com/vllm-project/vllm/issues/15735)——expert parallelism、多机 DP、集群 prefill 分离。

## 致谢

页上 Meta 一长串名字（Lucia Fang、Ye Qi……）。AMD：Hongxia Yang、Weijun Jiang。vLLM 评测硬件：Nebius 与 NVIDIA。
