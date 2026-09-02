---
source: https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 扩散 cache：相邻 timestep 不必重算

英文对照：`en/vllm/blog/serving/omni-diffusion-cache.md`  
原文：https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration  
H200、Qwen-Image 1024²。

TeaCache：`cache_backend="tea_cache"`，`rel_l1_thresh=0.2`，20.0s → 10.47s（约 **1.91×**）。Cache-DiT：DBCache + TaylorSeer，约 **1.85×**。编辑任务 Qwen-Image-Edit：Cache-DiT 51.5s → 21.6s（约 **2.38×**）；TeaCache 约 1.47×。Ascend 上 Edit 142.38s → 64.07s（约 2.2×）。Z-Image 当时只有 Cache-DiT。不改权重，吃的是时间冗余。接 [vLLM-Omni](vllm-omni.md)。

本地图（原文版权仍归原站；学习对照用）：

![cat](../../../../assets/vllm/blog/serving/omni-diffusion-cache/01-cat.png)

![cat tea cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/02-cat_tea_cache.png)

![cat cache dit](../../../../assets/vllm/blog/serving/omni-diffusion-cache/03-cat_cache_dit.png)

![qwen bear base](../../../../assets/vllm/blog/serving/omni-diffusion-cache/04-qwen_bear_base.png)

![qwen bear tea cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/05-qwen_bear_tea_cache.png)

![qwen bear cache dit](../../../../assets/vllm/blog/serving/omni-diffusion-cache/06-qwen_bear_cache_dit.png)
