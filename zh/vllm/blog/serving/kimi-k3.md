---
source: https://vllm.ai/blog/2026-07-27-k3
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Kimi K3 day-0：2.8T 怎么端上桌

英文对照：[en/vllm/blog/serving/kimi-k3.md](../../../../en/vllm/blog/serving/kimi-k3.md)  
原文：https://vllm.ai/blog/2026-07-27-k3  
2026-07-27。数字是 GB300 NVL72 上的演示。KDA 前缀缓存设计见 [preview](kimi-k3-preview.md)。

`moonshotai/Kimi-K3`：2.8T MoE，896 expert 里激活 16，1M 上下文，原生视觉，权重 MXFP4。注意力是 KDA（定长 recurrent）夹 periodic full attention，再加 AttnRes、Stable LatentMoE。聊天模板是 Python 渲 token，不是 Jinja。**引擎骨架没换**；换的是 hybrid cache、kernel、配方。当时只能 Docker（含预发布 FlashInfer）。


本地图（原文版权仍归原站；学习对照用）：

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

## 旗

最少 8×B300 或 8×MI355X。`--tensor-parallel-size 8`、`--load-format fastsafetensors`、`--tool-call-parser kimi_k3`、`--reasoning-parser kimi_k3`。Prefix cache **默认关**，要显式 `--enable-prefix-caching`。DSpark：`--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,...}'`。DEP 用 `deep_gemm_mega_moe`；TP>1 用 `flashinfer_trtllm`。All-to-all：NVLink `flashinfer_nvlink_one_sided`，RDMA `deepep_v2`。ViT 默认 `--mm-encoder-tp-mode=data`（`head_size=12` 切不匀 TP8）。`VLLM_USE_RUST_FRONTEND=1`。

## 数字（演示）

bs=1：无投机 TP8 **111** / TP16 **118 tok/s**；DSpark 约 **3.14×**，到 **331 / 370 tok/s**。低熵任务约 4.73 accept/step，高熵约 2.61。准确率（最大 reasoning）：GSM8K 0.976、GPQA-Diamond 0.939。KDA metadata builder 把准备延迟 **870 µs → 34 µs**（bs=1）。早期 DCP 原型相对 TP8 约 **+40%** 吞吐——当时未合入。
