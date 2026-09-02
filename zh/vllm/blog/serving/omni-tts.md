---
source: https://vllm.ai/blog/2026-06-23-vllm-omni-tts
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# TTS：Talker 要 TTFP，Code2Wav 要吞吐

英文对照：`en/vllm/blog/serving/omni-tts.md`  
原文：https://vllm.ai/blog/2026-06-23-vllm-omni-tts  
覆盖 Qwen3-TTS、VoxCPM2、Higgs Audio V3、Fish Speech S2 Pro。

TTS 不是单只 LLM。Talker 自回归 codec（延迟）、Code2Wav 并行还原波形（吞吐）。调度当成同一类，两边都亏。块太小，跨块不连续；太大，**TTFP**（第一包音频）爆。Qwen3-TTS：阶段分离 + connector 切块、Talker decode 预处理批量化。VoxCPM2：整段 `torch.compile`、CFM/LocDiT 尾部批。Higgs：多 codebook 状态留 GPU。Fish：`q_len=1` 专用 decode attention。没有一只菜谱伺候所有 TTS。接 [Omni](vllm-omni.md) 与 [Qwen3-Omni](qwen3-omni.md)。

本地图（原文版权仍归原站；学习对照用）：

![tts serving pipeline](../../../../assets/vllm/blog/serving/omni-tts/01-tts-serving-pipeline.png)

![qwen3 tts connector chunking](../../../../assets/vllm/blog/serving/omni-tts/02-qwen3-tts-connector-chunking.png)

![qwen3 tts stage0 dispatch consolidation](../../../../assets/vllm/blog/serving/omni-tts/03-qwen3-tts-stage0-dispatch-consolidation.png)

![voxcpm2 single stage pipeline](../../../../assets/vllm/blog/serving/omni-tts/04-voxcpm2-single-stage-pipeline.png)

![voxcpm2 compile dispatch combined](../../../../assets/vllm/blog/serving/omni-tts/05-voxcpm2-compile-dispatch-combined.png)

![fish speech stage0 runtime shape](../../../../assets/vllm/blog/serving/omni-tts/06-fish-speech-stage0-runtime-shape.png)
