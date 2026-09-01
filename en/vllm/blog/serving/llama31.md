---
source: https://vllm.ai/blog/2024-07-23-llama31
lang: en
fetched: 2026-09-01
---

# Llama 3.1: 128K auto chunked prefill; 405B FP8 on one node, PP across nodes

Chinese: `../../zh/vllm/blog/serving/llama31.md`  
Numbers are **early** reference points; the post says weeks of headroom remain.

128K turns on chunked prefill: bounds memory and reduces long-prompt interruption of in-flight decode. 405B-Instruct-FP8: `--tensor-parallel-size 8` on 8×H100/A100. Their 1024/128 load: **2.82 req/s**, input **2884.86 tok/s**, output **291.53 tok/s**. GSM8K 8-shot CoT: FP8 **95.38%** vs BF16 official 96.8%. Unquantized: `--pipeline-parallel-size 2 --tensor-parallel-size 8` on 16 GPUs; without IB, PP+TP ~**6.6×** vs 16-way TP; with IB they match. Also 8×MI300x / 8×H200 or CPU offload. More quant and PP throughput still incoming then.
