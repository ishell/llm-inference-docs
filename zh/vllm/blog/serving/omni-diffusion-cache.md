---
source: https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 扩散 cache：相邻 timestep 不必重算

英文对照：[en/vllm/blog/serving/omni-diffusion-cache.md](../../../../en/vllm/blog/serving/omni-diffusion-cache.md)  
原文：https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration  
2025-12-19。vLLM-Omni 团队。接在 [vLLM-Omni](vllm-omni.md) 立项之后。Cache-DiT 库：[vipshop/cache-dit](https://github.com/vipshop/cache-dit)。Omni 文档：[Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/)、[TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/)。数字是发布时的演示，不是你机器上的 SLA。

## 给扩散推理加油

vLLM-Omni 给扩散推理接上了 cache 加速：**Cache-DiT** 和 **TeaCache**。相邻 timestep 的中间结果可以留下来，后面几步就不必把 Transformer 再走一遍。

他们写的范围：图像生成大约 **1.5× 到超过 2×**，配置很少，质量损失他们称为可以忽略。不改权重，不重新训练——吃的是时间上的冗余。

## 瓶颈：扩散里的重复劳动

一张图仍要走**几十步**推理。相邻两步看见的特征往往很像。这份 **temporal redundancy** 就是杠杆：把贵的中间量缓存下来，下一步直接跳过。

## 两套后端

### Cache-DiT：旋钮多，峰值快

这是一只**外部库**，不是 Omni 自己长出来的。文中点了三件兵器：

- **DBCache（Dual Block Cache）：** 按残差差，缓存 Transformer **block** 的输出。
- **TaylorSeer：** 用泰勒展开去预报特征，算得更少。
- **SCM（Step Computation Masking）：** 自适应掩码，整步都可以跳。

### TeaCache：简单、会看脸色

**写在 vLLM-Omni 里面。** Hook。盯输入之间的差别，每一步自己决定：要不要复用上一个 timestep 的 Transformer 计算。

## 基准（NVIDIA H200，Qwen-Image 1024×1024）

| 模型 | 后端 | 配置 | 耗时 | 加速 |
|---|---|---|---|---|
| Qwen-Image | Baseline | 无 | 20.0s | 1.0× |
| Qwen-Image | TeaCache | `rel_l1_thresh=0.2` | 10.47s | **1.91×** |
| Qwen-Image | Cache-DiT | DBCache + TaylorSeer | 10.8s | **1.85×** |

本地图（原文版权仍归原站；学习对照用）。图注与原页一致：No Cache / TeaCache / Cache-DiT。

![No Cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/01-cat.png)

![TeaCache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/02-cat_tea_cache.png)

![Cache-DiT](../../../../assets/vllm/blog/serving/omni-diffusion-cache/03-cat_cache_dit.png)

## Edit 模型

**Qwen-Image-Edit** 上，Cache-DiT 更亮：**51.5s → 21.6s**，**2.38×**。TeaCache 是 **35.0s**，**1.47×**。

| 模型 | 后端 | 配置 | 耗时 | 加速 |
|---|---|---|---|---|
| Qwen-Image-Edit | Baseline | 无 | 51.5s | 1.0× |
| Qwen-Image-Edit | TeaCache | `rel_l1_thresh=0.2` | 35.0s | **1.47×** |
| Qwen-Image-Edit | Cache-DiT | DBCache + TaylorSeer | 21.6s | **2.38×** |

![No Cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/04-qwen_bear_base.png)

![TeaCache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/05-qwen_bear_tea_cache.png)

![Cache-DiT](../../../../assets/vllm/blog/serving/omni-diffusion-cache/06-qwen_bear_cache_dit.png)

同一套办法在 **Ascend NPU** 上也成立：Qwen-Image-Edit + Cache-DiT，**142.38s → 64.07s**，略高于 **2.2×**。异构硬件，cache 的故事不变。

## 当时支持的模型

| 模型 | TeaCache | Cache-DiT |
|---|---|---|
| Qwen-Image | 有 | 有 |
| Z-Image | **无** | 有 |
| Qwen-Image-Edit | 有 | 有 |

Z-Image 在这篇里**只有 Cache-DiT**。

## 怎么开

在构造 `Omni` 时设 `cache_backend`。

### TeaCache

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)

outputs = omni.generate(
    prompt="A cat sitting on a windowsill",
    num_inference_steps=50,
)
```

### Cache-DiT

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 8,
        "enable_taylorseer": True,  # 泰勒展开预报
        "taylorseer_order": 1,
    },
)

outputs = omni.generate(
    prompt="A cat sitting on a windowsill",
    num_inference_steps=50,
)
```

## 再往下读

旋钮细节在文档：[Cache-DiT 加速](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/)、[TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/)。文末还点了并行、kernel 融合、量化——后来的 Omni 笔记把这些接走（[omni-autoround](omni-autoround.md)、[omni-layerwise-offload](omni-layerwise-offload.md)）。
