---
source: https://vllm.ai/blog/2024-07-23-llama31
lang: en
fetched: 2026-09-04
---

# Announcing Llama 3.1 Support in vLLM

Chinese: [zh/vllm/blog/serving/llama31.md](../../../../zh/vllm/blog/serving/llama31.md)

2024-07-23. **vLLM Team**. Partner post with Meta. **Early** numbers — they say performance is preliminary and should rise in weeks. Llama 4 later: [llama4.md](llama4.md). Chunked prefill / distributed cousins in docs, not copied here. Docker then: `vllm/vllm-openai`. Install: `pip install -U vllm`.

Llama 3.1: up to **128K** context, up to **405B**. vLLM adds chunked prefill, FP8, pipeline parallelism so the longer/larger Llamas fit.

128K: vLLM **automatically enables chunked prefill**. Controls memory; also reduces interruption of in-flight decode by a long prompt.

## 405B: methods they list

- **FP8:** official FP8 on **8×A100 or 8×H100**
- **Pipeline parallelism:** official BF16, layers on different nodes
- **Tensor parallelism:** shard across nodes and GPUs
- **AMD MI300x or NVIDIA H200:** single **8×MI300x** (192 GB) or **8×H200** (141 GB)
- **CPU offloading:** last resort, full precision on tight GPU memory

Recommendation then: **FP8 for a single node**, **pipeline parallelism for multiple nodes**. They were still exploring more quantization and PP throughput.

## FP8

H100 / MI300x native FP8 tensor cores. vLLM then: FP8 for KV cache, attention, and MLP. Official Meta Llama 3.1 405B FP8 via FBGEMM: **per-channel** quant on MLP up/gate/down with a static scale; skip first and last layer; static upper bound. Minimal accuracy drop is the claim.

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tensor-parallel-size 8
```

Load they quote: avg input **1024**, avg output **128**:

| Metric | Value |
|---|---|
| Requests | **2.82 req/s** |
| Input | **2884.86 tok/s** |
| Output | **291.53 tok/s** |

GSM8K, lm-eval-harness, 8-shot CoT, exact match: FP8 **95.38% (±0.56%)** vs BF16 official **96.8%**.

## Pipeline parallelism

Unquantized 405B on **16×H100 or 16×A100**. PP splits by layer groups; P2P instead of all-reduce. Useful when nodes lack InfiniBand.

Combine PP + TP. Example, 16 GPUs / 2 nodes: **PP2 × TP8** — half the model per node, NVLink all-reduce inside the node:

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2
```

With InfiniBand, 16-way TP:

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 16
```

Local figures (copyright remains with the original site; study copies):

![perf llama3](../../../../assets/vllm/blog/serving/llama31/01-perf_llama3.png)

Serving throughput on **16×H100**, synthetic avg 1024 / 128.

**Without InfiniBand**, PP2+TP8 vs 16-way TP: about **6.6×**. **With InfiniBand**, similar.

Docs they point at: [distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html), [CPU offload example](https://docs.vllm.ai/en/latest/getting_started/examples/cpu_offload.html).

## Acknowledgements

Meta pre-release. Neural Magic (FP8), CentML + Snowflake AI Research (PP), Anyscale (chunked prefill). Eval on Lambda 1-Click Clusters with InfiniBand.
