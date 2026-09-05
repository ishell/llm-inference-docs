---
source: https://vllm.ai/blog/2025-06-30-minimax-m1
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# MiniMax-M1：Lightning Attention + MoE，当时 Docker 示例还钉 V0

英文对照：[en/vllm/blog/serving/minimax-m1.md](../../../../en/vllm/blog/serving/minimax-m1.md)  
原文：https://vllm.ai/blog/2025-06-30-minimax-m1  
2025-06-30。署名 **MiniMax**。论文：[arXiv:2506.13585](https://arxiv.org/pdf/2506.13585)。456B 总、约 45.9B 激活。后来的 MSA / 1M 亲戚：[minimax-m3.md](minimax-m3.md)。再后来 Omni：[minimax-h3.md](minimax-h3.md)。当时计划进 V1：[v1-alpha.md](../architecture/v1-alpha.md)。PagedAttention 背景：[paged-attention.md](../architecture/paged-attention.md)。架构介绍加当时部署。文内 Docker 钉了 **`VLLM_USE_V1=0`**——**历史**；后来 hybrid allocator 进了 V1。

这篇讲 MiniMax-M1 的 hybrid 架构怎么在 vLLM 里伺候：模型特点、推理难点、当时那条技术路径。

本地图（原文版权仍归原站；学习对照用）：

![benchmark](../../../../assets/vllm/blog/serving/minimax-m1/01-benchmark.png)

![moe](../../../../assets/vllm/blog/serving/minimax-m1/02-moe.png)

![lightning attention](../../../../assets/vllm/blog/serving/minimax-m1/03-lightning_attention.png)

## 引言

[MiniMax-M1](https://arxiv.org/pdf/2506.13585) 是开源的大规模 MoE 推理模型。Hybrid 架构对着长上下文推理和复杂任务。推荐 serving 路径是 vLLM。

**Figure（左）。** 商业和开源模型在数学、代码、软件工程、工具、长上下文理解上的对比。那张图上 MiniMax-M1 在开源里领跑。

**Figure（右）。** 理论推理 FLOPs 随 token 长度。相对 DeepSeek R1，生成 **100k** token 时 MiniMax-M1 大约只用 **25%** FLOPs。

## 用 vLLM 部署 MiniMax-M1

他们测试里报的好处：吞吐、内存管理、batched 请求、后端性能。

### 下模型

Hugging Face：`MiniMaxAI/MiniMax-M1-40k`（或 `MiniMax-M1-80k`）。

```bash
pip install -U huggingface-hub
huggingface-cli download MiniMaxAI/MiniMax-M1-40k
# huggingface-cli download MiniMaxAI/MiniMax-M1-80k
```

### 部署

当时的 Docker + vLLM 片段。**`VLLM_USE_V1=0` 是历史**——别当成今天的默认。

```bash
IMAGE=vllm/vllm-openai:latest
MODEL_DIR=<model storage path>
NAME=MiniMaxImage
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --rm --gpus all --ulimit stack=67108864"

sudo docker run -it \
    -v $MODEL_DIR:$MODEL_DIR \
    --name $NAME \
    $DOCKER_RUN_CMD \
    $IMAGE /bin/bash

export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=0
vllm serve \
--model <model storage path> \
--tensor-parallel-size 8 \
--trust-remote-code \
--quantization experts_int8 \
--max_model_len 4096 \
--dtype bfloat16
```

## MiniMax-M1 Hybrid 架构要点

### Mixture-of-Experts（MoE）

总参 **456B**。路由按 token 语义激活稀疏子集（约 **45.9B**，约 **10%**）。门控网络算专家选择概率。

分类任务：他们称相对稠密模型最多少 **90%** 计算，准确率相当。

**Figure。** Isoflop：MoE vs Dense，都在 1T token 上训。灰虚线：两边达到同一表现所需计算的差。

### Lightning Attention

二次的 softmax attention → 线性化近似：softmax 变成 **矩阵乘的线性组合**，再配动态内存 tiling 和梯度近似。

代码补全、**100k** token 序列上的声称：内存 **−83%**，推理延迟 **−67%**。

**Figure。** Lightning Attention 算法总览——长序列上的内存和延迟。

### 高效计算和激活策略

Lightning Attention 管运行时；稀疏专家激活跳过不必做的计算。架构深读论文：[arXiv:2506.13585](https://arxiv.org/pdf/2506.13585)。

## vLLM 里的高效推理

### 内存管理

PagedAttention：KV 分页，而不是一整块连续分配。他们称碎片 **<4%**，传统 **60–80%**。超长上下文靠这个，才不那么容易撞上内存墙。

### Kernel 级优化

FlashAttention、FlashInfer；量化 GPTQ、AWQ、INT4、INT8、FP8。量化砍内存和计算，（声称）准确率损失很小；FlashAttention 加速 attention 本身。

### vLLM 里的 Lightning Attention

走 **Triton**。Triton 执行覆盖 Lightning Attention 的核心计算，hybrid 路径坐进 vLLM。

## 当时的下一步

给 MiniMax-M1 这类模型做 hybrid allocator。当时计划完整支持 [vLLM V1](../architecture/v1-alpha.md)，hybrid 架构迁进 V1。后来发生了；本页是 2025-06 的快照。

## 结语

MiniMax-M1 的 hybrid 路径对着长上下文推理和复杂任务。vLLM 当时的说法：paged KV、batch、kernel 级后端。合在一起：代码生成、文档分析、对话。M3 后来用 MSA 换掉 Lightning Attention——[minimax-m3.md](minimax-m3.md)。

## 致谢

vLLM 社区，尤其 Tyler Michael Smith、Simon Mo、Cyrus Leung、Roger Wang、Zifeng Mo、Kaichao You。MiniMax 工程：Gangying Qing、Jun Qing、Jiaren Cai。
