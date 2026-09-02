---
source: https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Qwen3-Omni：Thinker / Talker / Code2Wav 三截

英文对照：`en/vllm/blog/serving/qwen3-omni.md`  
原文：https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization  
TTS 工程细节见 [omni-tts](omni-tts.md)。

```
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

`/v1/chat/completions` 出文本和音频。三阶段各自 batch + CUDA graph；async chunk / async output 避免等整包；Talker/Code2Wav 可 replica。平台段在 yaml 的 `platforms:`，CUDA/NPU/ROCm/XPU 自动合并。他们扫的是音频 TTFP、RTF、吞吐——数字看原图。和文本 LLM 的 TTFT 不是同一只表。

本地图（原文版权仍归原站；学习对照用）：

![qwen3 omni serving flow](../../../../assets/vllm/blog/serving/qwen3-omni/01-qwen3-omni-serving-flow.svg)

![qwen3 omni optimization stack](../../../../assets/vllm/blog/serving/qwen3-omni/02-qwen3-omni-optimization-stack.svg)

![qwen3 omni cuda graph stages](../../../../assets/vllm/blog/serving/qwen3-omni/03-qwen3-omni-cuda-graph-stages.svg)

![qwen3 omni async chunk timeline](../../../../assets/vllm/blog/serving/qwen3-omni/04-qwen3-omni-async-chunk-timeline.svg)

![qwen3 omni async output step gap](../../../../assets/vllm/blog/serving/qwen3-omni/05-qwen3-omni-async-output-step-gap.svg)

![qwen3 omni async replica](../../../../assets/vllm/blog/serving/qwen3-omni/06-qwen3-omni-async-replica.svg)

![qwen3 omni bench reqps](../../../../assets/vllm/blog/serving/qwen3-omni/07-qwen3-omni-bench-reqps.svg)

![qwen3 omni bench rtf](../../../../assets/vllm/blog/serving/qwen3-omni/08-qwen3-omni-bench-rtf.svg)

![qwen3 omni bench ttfp](../../../../assets/vllm/blog/serving/qwen3-omni/09-qwen3-omni-bench-ttfp.svg)
