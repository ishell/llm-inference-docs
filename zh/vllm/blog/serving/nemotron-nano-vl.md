---
source: https://vllm.ai/blog/2025-10-31-run-multimodal-reasoning-agents-nvidia-nemotron
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Nemotron Nano 2 VL：12B 视频/文档，EVS 砍冗余帧

英文对照：`en/vllm/blog/serving/nemotron-nano-vl.md`  
原文：https://vllm.ai/blog/2025-10-31-run-multimodal-reasoning-agents-nvidia-nemotron  
128K。CRADIOH-V2 encoder + EVS + Nano 2 LLM。当时 nightly。图在原网页。

`--video-pruning-rate 0` 是不砍；FP8/FP4 走 `--quantization modelopt` / `modelopt_fp4`。系统提示 `/no_think` 关思考。他们报一串 VLM 榜均分 74 vs 当时顶 VL 64.2——营销对照，复测用 cookbook。后继 Omni 见 [nemotron-omni](nemotron-omni.md)。
