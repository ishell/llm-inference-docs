---
source: https://vllm.ai/blog/2026-04-02-gemma4
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Gemma 4 Day-0：NVIDIA / AMD / Intel / TPU 一起开，Apache 2.0

英文对照：[en/vllm/blog/serving/gemma4.md](../../../../en/vllm/blog/serving/gemma4.md)  
原文：https://vllm.ai/blog/2026-04-02-gemma4  
E2B / E4B / 26B MoE / 31B Dense。菜谱在 model card 和 GKE/GCE demo。

边端 128K，大号 256K。全尺寸原生图/视频；E2B/E4B 另有音频。function calling、structured JSON、system instruction。TPU Day-0 是卖点——接 [vllm-tpu](../architecture/vllm-tpu.md)。这篇几乎没有可复现 TPS；当矩阵看，别当基准。

本地图（原文版权仍归原站；学习对照用）：

![gemma4 elo score](../../../../assets/vllm/blog/serving/gemma4/01-gemma4-elo-score.png)
