---
source: https://vllm.ai/blog/2025-12-19-vllm-omni-diffusion-cache-acceleration
lang: en
fetched: 2026-09-04
---

# vLLM-Omni Diffusion Cache Acceleration

Chinese: [zh/vllm/blog/serving/omni-diffusion-cache.md](../../../../zh/vllm/blog/serving/omni-diffusion-cache.md)

2025-12-19. vLLM-Omni Team. Follows the [vLLM-Omni](vllm-omni.md) launch. Cache-DiT library: [vipshop/cache-dit](https://github.com/vipshop/cache-dit). Omni docs: [Cache-DiT guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/), [TeaCache guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/). Demo numbers, not your SLA.

## Turbocharge diffusion inference

vLLM-Omni adds cache acceleration for diffusion inference: **Cache-DiT** and **TeaCache**. Intermediate computations are reused across timesteps so later steps skip redundant Transformer work.

Claimed range: **1.5× to over 2×** on image generation, small config, quality loss they call negligible. No weight change, no retraining.

## The bottleneck: redundancy in diffusion

A single image still takes **dozens of inference steps**. Adjacent timesteps often see very similar features. That **temporal redundancy** is the lever: cache the expensive intermediates, skip them on the next step.

## Two backends

### Cache-DiT — more knobs, peak speed

A **library** (not in-tree). Three techniques named in the post:

- **DBCache (Dual Block Cache):** cache Transformer **block** outputs from residual differences.
- **TaylorSeer:** Taylor-expansion forecast of features, so you compute even less.
- **SCM (Step Computation Masking):** adaptive mask that skips whole steps.

### TeaCache — simple and adaptive

**Native** in vLLM-Omni. Hook-based. Watches the difference between inputs and decides, per step, whether to reuse the previous timestep's Transformer computation.

## Benchmarks (NVIDIA H200, Qwen-Image 1024×1024)

| Model | Backend | Configuration | Time | Speedup |
|---|---|---|---|---|
| Qwen-Image | Baseline | none | 20.0s | 1.0× |
| Qwen-Image | TeaCache | `rel_l1_thresh=0.2` | 10.47s | **1.91×** |
| Qwen-Image | Cache-DiT | DBCache + TaylorSeer | 10.8s | **1.85×** |

Local figures (copyright remains with the original site; study copies). Captions as on the page: No Cache / TeaCache / Cache-DiT.

![No Cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/01-cat.png)

![TeaCache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/02-cat_tea_cache.png)

![Cache-DiT](../../../../assets/vllm/blog/serving/omni-diffusion-cache/03-cat_cache_dit.png)

## The Edit model

On **Qwen-Image-Edit**, Cache-DiT is the larger win: **51.5s → 21.6s**, **2.38×**. TeaCache is **35.0s**, **1.47×**.

| Model | Backend | Configuration | Time | Speedup |
|---|---|---|---|---|
| Qwen-Image-Edit | Baseline | none | 51.5s | 1.0× |
| Qwen-Image-Edit | TeaCache | `rel_l1_thresh=0.2` | 35.0s | **1.47×** |
| Qwen-Image-Edit | Cache-DiT | DBCache + TaylorSeer | 21.6s | **2.38×** |

![No Cache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/04-qwen_bear_base.png)

![TeaCache](../../../../assets/vllm/blog/serving/omni-diffusion-cache/05-qwen_bear_tea_cache.png)

![Cache-DiT](../../../../assets/vllm/blog/serving/omni-diffusion-cache/06-qwen_bear_cache_dit.png)

Same idea on **Ascend NPU**: Qwen-Image-Edit + Cache-DiT **142.38s → 64.07s**, a bit over **2.2×**. Heterogeneous hardware, same cache story.

## Supported models (then)

| Model | TeaCache | Cache-DiT |
|---|---|---|
| Qwen-Image | yes | yes |
| Z-Image | **no** | yes |
| Qwen-Image-Edit | yes | yes |

Z-Image had **Cache-DiT only** in this post.

## Quick start

Set `cache_backend` on the `Omni` constructor.

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
        "enable_taylorseer": True,  # Taylor expansion forecasting
        "taylorseer_order": 1,
    },
)

outputs = omni.generate(
    prompt="A cat sitting on a windowsill",
    num_inference_steps=50,
)
```

## Learn more

Docs for the extra knobs: [Cache-DiT acceleration](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/), [TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/teacache/). The post also flags work in flight: parallelization, kernel fusion, quantization — later Omni notes pick those up ([omni-autoround](omni-autoround.md), [omni-layerwise-offload](omni-layerwise-offload.md)).
