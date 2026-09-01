---
source: https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
lang: en
fetched: 2026-09-01
---

# Intel Arc Pro B-Series

2025-11-11. XPU / SYCL. Study note; figures on the original page. Demos on 4–8× Arc Pro B60. Feature list is long: DeepSeek distill, >50K context, embed/rerank/pooling, multimodal, MoE, per-layer online quant, DP/TP/PP, FP16/BF16 `torch.compile`, n-gram/EAGLE/EAGLE3, async scheduling, P/D, LoRA, sleep mode, structured output, tool calling. See [sleep-mode.md](sleep-mode.md), [spec-decode.md](../performance/spec-decode.md), [hardware-plugin.md](hardware-plugin.md).

Naive MoE GEMM: one kernel per expert after the gate. Persistent **zero-gap** kernel claimed **>80%** of B60 capacity: single persistent loop (20 XeCores × 2 SYCL groups); atomic steal of the next GEMM block (static stride wasted ~15%); MXFP4→BF16 prepack (~**+30%** load) via `Bitcast-bf16 ((x << 12) >> 6 & 0x81c0) * 2^126`.

Demos: 8× B60 DeepSeek-distill FP8 throughput in the figures; Qwen-32B next-token **<100 ms** under load; Llama-70B 1K–40K single-batch TTFT/TPOT. GPT-OSS MXFP4: 20b TP1 1024/1024 conc=75 → TTFT **7.614 s**, TPOT **53.96 ms**, **1210.74 tok/s**; 120b TP4 same shape conc=100 → **1495.12 tok/s**. MLPerf Inference v5.1 Llama 8B price/perf mention.

```bash
docker pull intel/vllm:0.10.2-xpu
vllm serve openai/gpt-oss-120b --dtype=bfloat16 --enforce-eager \
  --gpu-memory-util=0.9 --no-enable-prefix-caching \
  --max-num-batched-tokens=8192 --max-model-len=16384 --block-size 64 -tp 4
```

Host then: Ubuntu 25.04, KMD 6.14.0. gpt-oss from the 0.10.2 XPU image.
