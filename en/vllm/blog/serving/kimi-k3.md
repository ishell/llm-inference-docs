---
source: https://vllm.ai/blog/2026-07-27-k3
lang: en
fetched: 2026-09-01
---

# Kimi K3 day-0

Chinese: [zh/vllm/blog/serving/kimi-k3.md](../../../../zh/vllm/blog/serving/kimi-k3.md)  
2026-07-27. Demo: GB300 NVL72. KDA prefix-cache design: [preview](kimi-k3-preview.md). Docker-only then (prerelease FlashInfer).

`moonshotai/Kimi-K3`: 2.8T MoE, 16 of 896 experts, 1M context, native vision, MXFP4 weights. Hybrid KDA + periodic full attention, AttnRes, Stable LatentMoE. Chat template is a Python renderer, not Jinja. **Not a new engine** — hybrid cache, kernels, recipes.

**HW:** ≥8× B300 or 8× MI355X (16× B200 also). Prefix cache **off by default** — pass `--enable-prefix-caching`.

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 --trust-remote-code \
  --load-format fastsafetensors --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
```

DSpark: `--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,...}'`. DEP → `deep_gemm_mega_moe`; TP>1 → `flashinfer_trtllm`. All-to-all: NVLink `flashinfer_nvlink_one_sided`, RDMA `deepep_v2`. ViT default `--mm-encoder-tp-mode=data`. `VLLM_USE_RUST_FRONTEND=1`.

Demo bs=1: no spec TP8 **111** / TP16 **118 tok/s**; DSpark ~**3.14×** → **331 / 370**. Low-entropy ~4.73 accept/step, high-entropy ~2.61. Max reasoning: GSM8K 0.976, GPQA-Diamond 0.939. KDA metadata prep **870 µs → 34 µs** at bs=1.

Local figures (copyright remains with the original site; study copies):

![architecture](../../../../assets/vllm/blog/serving/kimi-k3/01-architecture.png)

![hybrid cache](../../../../assets/vllm/blog/serving/kimi-k3/02-hybrid-cache.png)

![dspark acceptance rates](../../../../assets/vllm/blog/serving/kimi-k3/03-dspark-acceptance-rates.png)

![dspark schematic](../../../../assets/vllm/blog/serving/kimi-k3/04-dspark-schematic.png)

![sequence parallelism](../../../../assets/vllm/blog/serving/kimi-k3/05-sequence-parallelism.jpg)

![pd disaggregation animation](../../../../assets/vllm/blog/serving/kimi-k3/06-pd-disaggregation-animation.gif)

![interval cache retention](../../../../assets/vllm/blog/serving/kimi-k3/07-interval-cache-retention.png)

![selective cache retention](../../../../assets/vllm/blog/serving/kimi-k3/08-selective-cache-retention.gif)

![kda decode](../../../../assets/vllm/blog/serving/kimi-k3/09-kda-decode.png)

![kda metadata builder](../../../../assets/vllm/blog/serving/kimi-k3/10-kda-metadata-builder.png)

![latent moe tail fusion](../../../../assets/vllm/blog/serving/kimi-k3/11-latent-moe-tail-fusion.png)

![serving performance](../../../../assets/vllm/blog/serving/kimi-k3/12-serving-performance.png)

![pareto gb300](../../../../assets/vllm/blog/serving/kimi-k3/13-pareto-gb300.png)
