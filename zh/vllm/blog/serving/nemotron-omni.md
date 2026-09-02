---
source: https://vllm.ai/blog/2026-04-28-nemotron-omni
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Nemotron 3 Nano Omni：一只 30B/3B 吃图声视频，吞吐对比钉死每用户速率

英文对照：`en/vllm/blog/serving/nemotron-omni.md`  
原文：https://vllm.ai/blog/2026-04-28-nemotron-omni  
v0.20.0 `vllm[audio]`。256K。BF16/FP8/NVFP4。

输入 text/image/video/audio，输出文本。Conv3D + Efficient Video Sampling。他们在**固定每用户 token 速率**下比系统吞吐：多文档约 **7.4×**，视频约 **9.2×**（相对另一只 open omni）。`--media-io-kwargs '{"video":{"num_frames":512,"fps":1}}'` `--video-pruning-rate 0.5`，`--reasoning-parser nemotron_v3` `--tool-call-parser qwen3_coder`。这是 perception 子代理，不是 [vLLM-Omni](vllm-omni.md) 那条扩散/TTS 栈。

本地图（原文版权仍归原站；学习对照用）：

![figure1](../../../../assets/vllm/blog/serving/nemotron-omni/01-figure1.png)

![figure2](../../../../assets/vllm/blog/serving/nemotron-omni/02-figure2.png)

![figure3](../../../../assets/vllm/blog/serving/nemotron-omni/03-figure3.png)
