---
source: https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration
lang: en
fetched: 2026-09-01
---

# Diffusion cache: skip redundant timesteps

Chinese: `../../zh/vllm/blog/serving/omni-diffusion-cache.md`  
H200, Qwen-Image 1024².

TeaCache: `cache_backend="tea_cache"`, `rel_l1_thresh=0.2`, 20.0s → 10.47s (~**1.91×**). Cache-DiT: DBCache + TaylorSeer, ~**1.85×**. Edit (Qwen-Image-Edit): Cache-DiT 51.5s → 21.6s (~**2.38×**); TeaCache ~1.47×. Ascend Edit 142.38s → 64.07s (~2.2×). Z-Image had Cache-DiT only then. No retraining — temporal redundancy. Follows [vLLM-Omni](vllm-omni.md).

Local figures (copyright remains with the original site; study copies):

![cat](../../../../assets/vllm/blog/serving/omni-diffusion-cache/01-cat.png)

![cat tea cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/02-cat_tea_cache.png)

![cat cache dit](../../../../assets/vllm/blog/serving/omni-diffusion-cache/03-cat_cache_dit.png)

![qwen bear base](../../../../assets/vllm/blog/serving/omni-diffusion-cache/04-qwen_bear_base.png)

![qwen bear tea cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/05-qwen_bear_tea_cache.png)

![qwen bear cache dit](../../../../assets/vllm/blog/serving/omni-diffusion-cache/06-qwen_bear_cache_dit.png)
