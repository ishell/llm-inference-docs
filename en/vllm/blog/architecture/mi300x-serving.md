---
source: https://vllm.ai/blog/2024-10-23-vllm-serving-amd
lang: en
fetched: 2026-09-01
---

# Serving Llama on MI300X (2024 knobs)

2024-10-23. Embedded LLM / Hot Aisle. vLLM **0.6.2**. Study note; figures on the original page. Demos: 8× MI300X BF16 ShareGPT vs TGI. Later attention routing: [rocm-attention.md](rocm-attention.md). Plugins: [hardware-plugin.md](hardware-plugin.md). Flags are of that vintage.

vs TGI: 405B ~**1.5×** throughput / **1.7×** TTFT; 70B ~**1.8×** / **5.1×**. At 16 QPS, optimized 405B TTFT ~**3.8×**. Demos.

Heuristics then: **disable chunked prefill** on MI300X unless you know otherwise; `--num-scheduler-steps` **10–15**; if prefix-cache hit is low, disable both chunked prefill and prefix cache; `--max-seq-len-to-capture 16384` (larger can coarsen CUDA-graph buckets); `numa_balancing=0`, `NCCL_MIN_NCHANNELS=112`; leave KV dtype auto (FP8 trades a little 70B throughput for room); min TP + many replicas for throughput, full-node TP for latency; `--max-num-seqs` 512+; `VLLM_USE_TRITON_FLASH_ATTN=0` for **CK Flash Attention**.

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  -tp 4 --max-num-seqs 1024 --max-seq-len-to-capture 16384 \
  --enable-chunked-prefill=False --num-scheduler-steps 15
```

Image then: `ghcr.io/embeddedllm/vllm-rocm:cb3b2b9`. Chatbot-length ShareGPT; summarization/long-form not covered.
