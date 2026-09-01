---
source: https://vllm.ai/blog/2025-11-30-vllm-omni
lang: en
fetched: 2026-09-01
---

# vLLM-Omni: a pipeline past text

Chinese: `../../zh/vllm/blog/serving/vllm-omni.md`  
First cut on vLLM v0.11.0 (v0.11.0rc). Later posts cover TTS, diffusion cache, AutoRound, Qwen3-Omni.

LLM serving assumes autoregressive text. Omni models see, hear, speak, and mix DiT-class non-AR. vLLM-Omni splits a request into detachable **OmniStage**s: modality encoders (ViT / Whisper) → vLLM LLM core (text + hidden) → modality generators (DiT, …). Stages pipeline. Hugging Face models + OpenAI-compatible API. Throughput vs Transformers is in their figures.

Roadmap: full encoder/prefill/decode/generation disagg, diffusion DP/TP/TeaCache, hardware plugins. Slack `#sig-omni`. Not “one LLM does everything” — **heterogeneous stages** are the building.
