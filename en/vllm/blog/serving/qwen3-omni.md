---
source: https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
lang: en
fetched: 2026-09-01
---

# Qwen3-Omni: Thinker / Talker / Code2Wav

Chinese: [zh/vllm/blog/serving/qwen3-omni.md](../../../../zh/vllm/blog/serving/qwen3-omni.md)  
TTS engineering: [omni-tts](omni-tts.md).

```
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

`/v1/chat/completions` emits text and audio. Each stage batches and CUDA-graphs; async chunk / async output avoid waiting on full payloads; Talker/Code2Wav can replica. `platforms:` in the yaml merges CUDA/NPU/ROCm/XPU. They sweep audio TTFP, RTF, throughput —  Not the same stopwatch as text TTFT.

Local figures (copyright remains with the original site; study copies):

![qwen3 omni serving flow](../../../../assets/vllm/blog/serving/qwen3-omni/01-qwen3-omni-serving-flow.svg)

![qwen3 omni optimization stack](../../../../assets/vllm/blog/serving/qwen3-omni/02-qwen3-omni-optimization-stack.svg)

![qwen3 omni cuda graph stages](../../../../assets/vllm/blog/serving/qwen3-omni/03-qwen3-omni-cuda-graph-stages.svg)

![qwen3 omni async chunk timeline](../../../../assets/vllm/blog/serving/qwen3-omni/04-qwen3-omni-async-chunk-timeline.svg)

![qwen3 omni async output step gap](../../../../assets/vllm/blog/serving/qwen3-omni/05-qwen3-omni-async-output-step-gap.svg)

![qwen3 omni async replica](../../../../assets/vllm/blog/serving/qwen3-omni/06-qwen3-omni-async-replica.svg)

![qwen3 omni bench reqps](../../../../assets/vllm/blog/serving/qwen3-omni/07-qwen3-omni-bench-reqps.svg)

![qwen3 omni bench rtf](../../../../assets/vllm/blog/serving/qwen3-omni/08-qwen3-omni-bench-rtf.svg)

![qwen3 omni bench ttfp](../../../../assets/vllm/blog/serving/qwen3-omni/09-qwen3-omni-bench-ttfp.svg)
