---
source: https://vllm.ai/blog/2026-06-23-vllm-omni-tts
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# TTS：Talker 要 TTFP，Code2Wav 要吞吐

英文对照：`en/vllm/blog/serving/omni-tts.md`  
原文：https://vllm.ai/blog/2026-06-23-vllm-omni-tts  
覆盖 Qwen3-TTS、VoxCPM2、Higgs Audio V3、Fish Speech S2 Pro。图在原网页。

TTS 不是单只 LLM。Talker 自回归 codec（延迟）、Code2Wav 并行还原波形（吞吐）。调度当成同一类，两边都亏。块太小，跨块不连续；太大，**TTFP**（第一包音频）爆。Qwen3-TTS：阶段分离 + connector 切块、Talker decode 预处理批量化。VoxCPM2：整段 `torch.compile`、CFM/LocDiT 尾部批。Higgs：多 codebook 状态留 GPU。Fish：`q_len=1` 专用 decode attention。没有一只菜谱伺候所有 TTS。接 [Omni](vllm-omni.md) 与 [Qwen3-Omni](qwen3-omni.md)。
