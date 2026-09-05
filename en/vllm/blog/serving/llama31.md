---
source: https://vllm.ai/blog/2024-07-23-llama31
lang: en
fetched: 2026-09-04
---

# Llama 3.1: 128K auto chunked prefill; 405B FP8 on one node, PP across nodes

Chinese: [zh/vllm/blog/serving/llama31.md](../../../../zh/vllm/blog/serving/llama31.md)

2024-07-23. **vLLM Team**. Numbers are **early** reference points; the post says weeks of headroom remain. Later herd: [llama4.md](llama4.md). Llama Stack neighbor: [llama-stack.md](llama-stack.md). Distributed serving: [distributed-inference.md](distributed-inference.md). 1024/128 load numbers are their plate, not your SLA.

**TL;DR from the page:**

- Full **128K** window turns on chunked prefill: bounds memory; less long-prompt interruption of in-flight decode.
- 405B-Instruct-FP8: `--tensor-parallel-size 8` on 8×H100 / A100. Their 1024/128 load: **2.82 req/s**, input **2884.86 tok/s**, output **291.53 tok/s**.
- GSM8K 8-shot CoT: FP8 **95.38%** (±0.56) vs BF16 official **96.8%**.
- Unquantized: `--pipeline-parallel-size 2 --tensor-parallel-size 8` on 16 GPUs; without IB, PP+TP ~**6.6×** vs 16-way TP; with IB they match.

## Introduction

vLLM partnered with Meta on Llama 3.1: longer context (128K), larger size (405B), more capability. Enhancements named: chunked prefill, FP8 quantization, pipeline parallelism.

Then: 40+ LLM types, Nvidia / AMD / Inferentia / TPU / Intel / Gaudi. Docs: [docs.vllm.ai](https://docs.vllm.ai/).

128K: vLLM automatically enables [chunked prefill](https://www.linkedin.com/posts/joinanyscale_recently-weve-contributed-chunked-prefill-activity-7201277641490849792-lGqZ). Memory stays bounded; long prompts interrupt in-flight decode less.

Install then: `pip install -U vllm` or `vllm/vllm-openai`.

405B methods on the page:

- **FP8:** official FP8 on 8×A100 or 8×H100
- **Pipeline Parallelism:** BF16 across nodes, layers on different nodes
- **Tensor Parallelism:** shard across nodes and GPUs
- **AMD MI300x or NVIDIA H200:** one 8×MI300x (192 GB) or 8×H200 (141 GB)
- **CPU Offloading:** last resort, full precision on limited GPU memory

Recommend: FP8 for a single node; pipeline parallelism for multiple nodes. Performance still preliminary; more quant and PP throughput still incoming then.

## FP8

8-bit float. H100 / MI300x: native tensor cores. Then: FP8 for KV cache, attention, and MLP. Smaller footprint, higher throughput, lower latency, claimed minimal accuracy drop.

Official Meta Llama 3.1 405B FP8 via FBGEMM: per-channel quantization on MLP up/gate/down with a static scale. Skip first and last layer; static upper bound. Command:

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tensor-parallel-size 8
```

Load: avg input **1024**, avg output **128** → **2.82** requests/s; **2884.86** input tok/s, **291.53** output tok/s.

Accuracy check (lm-eval-harness, GSM8K 8-shot CoT, exact match): **95.38%** (±0.56 stddev) vs BF16 official **96.8%**.

## Pipeline parallelism

Unquantized 405B: 16×H100 or 16×A100. PP splits layers across nodes; point-to-point instead of expensive all-reduce. Useful when nodes lack InfiniBand.

Combine PP and TP. 16 GPUs / 2 nodes: 2-way PP + 8-way TP — half the model per node, NVLink all-reduce inside the node:

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2
```

With InfiniBand, 16-way TP is also on the page:

```shell
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 16
```

Local figures (copyright remains with the original site; study copies):

![perf llama3](../../../../assets/vllm/blog/serving/llama31/01-perf_llama3.png)

**Figure.** Serving throughput on 16×H100, synthetic dataset (avg input 1024, avg output 128). Without IB: PP+TP ~**6.6×** vs 16-way TP. With IB: similar.

Distributed docs: [distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). CPU offload example: [cpu_offload](https://docs.vllm.ai/en/latest/getting_started/examples/cpu_offload.html).

## Acknowledgements

Meta (pre-release partnership). Neural Magic (FP8). CentML and Snowflake AI Research (pipeline parallelism). Anyscale (chunked prefill). Eval on [Lambda 1-Click Clusters](https://lambdalabs.com/service/gpu-cloud/1-click-clusters) with InfiniBand.
