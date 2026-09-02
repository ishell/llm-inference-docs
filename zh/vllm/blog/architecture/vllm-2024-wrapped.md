---
source: https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 2024 年报 / 2025 愿景：星标 2.3×，V1 重写，遥测可关

英文对照：`en/vllm/blog/architecture/vllm-2024-wrapped.md`  
原文：https://vllm.ai/blog/2025-01-10-vllm-2024-wrapped-2025-vision  
star 14k→32.6k，贡献者 190→740，月下载 6k→27k，GPU hours 约 10×。https://2024.vllm.ai

年末近 100 种架构；硬件从 A100 扩到 H100/MI300/TPU/Inferentia/Gaudi/XPU/CPU。量化进了 >20% 部署。2025 口头：单卡/单节点 GPT-4o 级、默认打开量化/prefix/投机、V1 可插拔。用法统计 UUID + 硬件/模型/量化；关：`VLLM_NO_USAGE_STATS=1` 或 `~/.config/vllm/do_not_track`。这是当时的愿景文档，后来的 V1/MRV2/Wide-EP 才是落地。

本地图（原文版权仍归原站；学习对照用）：

![vllm contributor groups](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/01-vllm-contributor-groups.png)

![model architecture serving usage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/02-model-architecture-serving-usage.png)

![gpu hours by vendor](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/03-gpu-hours-by-vendor.png)

![quantization deployment percentage](../../../../assets/vllm/blog/architecture/vllm-2024-wrapped/04-quantization-deployment-percentage.png)
