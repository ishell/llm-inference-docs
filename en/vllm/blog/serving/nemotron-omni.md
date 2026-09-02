---
source: https://vllm.ai/blog/2026-04-28-nemotron-omni
lang: en
fetched: 2026-09-01
---

# Nemotron 3 Nano Omni: one 30B/3B for image/audio/video; TPS compared at fixed per-user rate

Chinese: `../../zh/vllm/blog/serving/nemotron-omni.md`  
v0.20.0 `vllm[audio]`. 256K. BF16/FP8/NVFP4.

In: text/image/video/audio; out: text. Conv3D + Efficient Video Sampling. They hold **per-user token rate** and compare system throughput: multi-doc ~**7.4×**, video ~**9.2×** vs another open omni. `--media-io-kwargs '{"video":{"num_frames":512,"fps":1}}'` `--video-pruning-rate 0.5`, `--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`. A perception sub-agent — not the [vLLM-Omni](vllm-omni.md) diffusion/TTS stack.

Local figures (copyright remains with the original site; study copies):

![figure1](../../../../assets/vllm/blog/serving/nemotron-omni/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/nemotron-omni/02-figure2.png)

![figure3](../../../../assets/vllm/blog/serving/nemotron-omni/03-figure3.png)
