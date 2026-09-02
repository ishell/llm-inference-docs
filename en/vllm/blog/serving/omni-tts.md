---
source: https://vllm.ai/blog/2026-06-23-vllm-omni-tts
lang: en
fetched: 2026-09-01
---

# TTS: Talker wants TTFP; Code2Wav wants throughput

Chinese: `../../zh/vllm/blog/serving/omni-tts.md`  
Qwen3-TTS, VoxCPM2, Higgs Audio V3, Fish Speech S2 Pro.

TTS is not one LLM. Talker is autoregressive codec (latency); Code2Wav reconstructs waveform in parallel (throughput). Same scheduler hurts both. Chunks too small break continuity; too large blow **TTFP** (time to first audio packet). Qwen3-TTS: stage split + connector chunks, batched Talker decode preprocess. VoxCPM2: whole-forward `torch.compile`, CFM/LocDiT tail batching. Higgs: multi-codebook state on GPU. Fish: `q_len=1` decode attention. No single recipe. Read with [Omni](vllm-omni.md) and [Qwen3-Omni](qwen3-omni.md).

Local figures (copyright remains with the original site; study copies):

![tts serving pipeline](../../../../assets/vllm/blog/serving/omni-tts/01-tts-serving-pipeline.png)

![qwen3 tts connector chunking](../../../../assets/vllm/blog/serving/omni-tts/02-qwen3-tts-connector-chunking.png)

![qwen3 tts stage0 dispatch consolidation](../../../../assets/vllm/blog/serving/omni-tts/03-qwen3-tts-stage0-dispatch-consolidation.png)

![voxcpm2 single stage pipeline](../../../../assets/vllm/blog/serving/omni-tts/04-voxcpm2-single-stage-pipeline.png)

![voxcpm2 compile dispatch combined](../../../../assets/vllm/blog/serving/omni-tts/05-voxcpm2-compile-dispatch-combined.png)

![fish speech stage0 runtime shape](../../../../assets/vllm/blog/serving/omni-tts/06-fish-speech-stage0-runtime-shape.png)
