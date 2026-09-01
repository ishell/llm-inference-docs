---
source: https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Qwen3-Omni：Thinker / Talker / Code2Wav 三截

英文对照：`en/vllm/blog/serving/qwen3-omni.md`  
原文：https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization  
图在原网页。TTS 工程细节见 [omni-tts](omni-tts.md)。

```
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

`/v1/chat/completions` 出文本和音频。三阶段各自 batch + CUDA graph；async chunk / async output 避免等整包；Talker/Code2Wav 可 replica。平台段在 yaml 的 `platforms:`，CUDA/NPU/ROCm/XPU 自动合并。他们扫的是音频 TTFP、RTF、吞吐——数字看原图。和文本 LLM 的 TTFT 不是同一只表。
