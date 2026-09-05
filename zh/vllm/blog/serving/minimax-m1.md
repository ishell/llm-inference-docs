---
source: https://vllm.ai/blog/2025-06-30-minimax-m1
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# MiniMax-M1：Lightning Attention + MoE，当时 Docker 示例还钉 V0

英文对照：[en/vllm/blog/serving/minimax-m1.md](../../../../en/vllm/blog/serving/minimax-m1.md)  
原文：https://vllm.ai/blog/2025-06-30-minimax-m1  
2025-06-30。署名 **MiniMax**。论文：[arXiv 2506.13585](https://arxiv.org/pdf/2506.13585)。权重：[`MiniMaxAI/MiniMax-M1-40k`](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k)、[`MiniMaxAI/MiniMax-M1-80k`](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)。**456B** 总、约 **45.9B** 激活。后来的 MSA / 1M 多模：[minimax-m3.md](minimax-m3.md)。再后来的 Omni：[minimax-h3.md](minimax-h3.md)。当时还在迁的 V1：[../architecture/v1-alpha.md](../architecture/v1-alpha.md)。hybrid KV 亲戚：[qwen3-next.md](qwen3-next.md)、[hybrid-ssm.md](hybrid-ssm.md)。这篇是架构介绍加当时部署，**不是** M3 的 MSA。文内 Docker 写了 `VLLM_USE_V1=0`——**历史**；后来 hybrid allocator 进 V1。

他们报 100k 生成相对 DeepSeek R1 约 **25%** FLOPs。`--quantization experts_int8`。Lightning Attention 走 Triton。PagedAttention 把碎片从传统 60–80% 压到他们说的 <4%。

本地图（原文版权仍归原站；学习对照用）：

![benchmark](../../../../assets/vllm/blog/serving/minimax-m1/01-benchmark.png)

![moe](../../../../assets/vllm/blog/serving/minimax-m1/02-moe.png)

![lightning attention](../../../../assets/vllm/blog/serving/minimax-m1/03-lightning_attention.png)

## Introduction

[MiniMax-M1](https://arxiv.org/pdf/2506.13585) 是开源大 MoE 推理模型。混合架构：稀疏专家 + 线性化注意力，冲长上下文推理。这篇是 MiniMax 自己写当时 vLLM 怎么伺候它——模型特点、推理痛点、用到的 vLLM 零件。不是跟后来 MiniMax 代际做 bake-off。

**Figure（左）。** 页上的板：math、code、SWE、tool use、long-context；MiniMax-M1 在那块板上自称「开源里领先」。

**Figure（右）。** 理论推理 FLOPs 对序列长度。相对 DeepSeek R1，生成 **100k** token 时 MiniMax-M1 只用 **25%** FLOPs。页上的理论曲线，不是 vLLM 测出来的 TPS 表。

## 用 vLLM 部署 MiniMax-M1

他们列的好处（「outstanding throughput」旁边没有数字）：

- 吞吐
- 内存管理
- 批量请求
- 后端优化

### Model download

```bash
pip install -U huggingface-hub

huggingface-cli download MiniMaxAI/MiniMax-M1-40k
# huggingface-cli download MiniMaxAI/MiniMax-M1-80k
```

### Deployment（按当时印的 Docker）

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
--quantization experts_int8  \
--max_model_len 4096 \
--dtype bfloat16
```

**原文写死的坑，以及后来的事实：** `VLLM_USE_V1=0` 和 `--max_model_len 4096` 是 **2025-06 这段示例钉死的**。同一篇把 hybrid allocator 和 V1 支持写成下一步；后来才落地。不要把这段 Docker 当成现在的 M3 菜谱——那是 [minimax-m3.md](minimax-m3.md)。

## 混合架构要点

### Mixture-of-Experts (MoE)

总共 **456B**。路由按 token 语义激活稀疏子集，约 **45.9B**（约 **10%**）。

他们称分类类任务相对 dense 最多少 **90%** 计算、精度相当。这是架构主张，不是 vLLM kernel 微基准。

**Figure.** Isoflop：MoE 对 dense，两边都训 **1T** token。灰虚线：打到同一成绩所需计算差。

### Lightning Attention

二次 softmax 注意力换成 **矩阵乘的线性组合**，再加动态 memory tiling 和梯度近似。

他们报的 code-completion、**100k-token** 序列：内存 **−83%**，推理延迟 **−67%**。仍是模型论文数字，不是 vLLM serving 表。

**Figure.** Lightning Attention 算法总览。

### 计算与激活

Lightning Attention 管运行时；稀疏专家跳过不用的计算。细节回 [论文](https://arxiv.org/pdf/2506.13585)。

## vLLM 上的高效推理

### Advanced memory management

PagedAttention：KV 分页，不是一整块连续。他们引用 vLLM 常说的：浪费 **<4%**，传统碎片 **60–80%**。M1 超长上下文靠这个才不会被超额分配卡死。

### Deep kernel-level optimizations

当时菜单：FlashAttention、FlashInfer、GPTQ / AWQ / INT4 / INT8 / FP8。量化换内存/计算；FlashAttention 加速注意力本身。**这篇真正给 M1 的 flag 是 `--quantization experts_int8`**，不是完整 GPTQ/AWQ 菜谱。

### Lightning Attention in vLLM

用 **Triton** 实现。Triton 路径在 vLLM 里跑 Lightning Attention 的核心运算——不是另一套引擎。

## Future work（截至 2025-06-30）

两条，都是 **当时路线图**：

- **Hybrid allocator**，给混用注意力风格的模型（M1 这一类）。
- **完整 vLLM V1**；把 hybrid 架构迁进 V1（[v1-alpha.md](../architecture/v1-alpha.md)）。

所以 Docker 示例还写着 `VLLM_USE_V1=0`。历史。M3 day-0 是另一种注意力（MSA）和另一套菜谱。

## Conclusion

M1 hybrid = 长上下文 + 稀疏 MoE。vLLM 这边 = paged KV、batching、Triton Lightning Attention、`experts_int8`。合在一起的意思是「能端上去」，不是 25k TPS/GPU 的 Pareto。后来的栈：[minimax-m3.md](minimax-m3.md)。

## Acknowledgement

vLLM：[Tyler Michael Smith](https://github.com/tlrmchlsmth)、[Simon Mo](https://github.com/simon-mo)、[Cyrus Leung](https://github.com/DarkLight1337)、[Roger Wang](https://github.com/ywang96)、[Zifeng Mo](https://github.com/Isotr0py)、[Kaichao You](https://github.com/youkaichao)。MiniMax：[Gangying Qing](https://github.com/ZZBoom)、[Jun Qing](https://github.com/qscqesze)、[Jiaren Cai](https://github.com/sriting)。
