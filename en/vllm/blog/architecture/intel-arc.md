---
source: https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
lang: en
fetched: 2026-09-04
---

# Fast and Affordable LLM Serving on Intel Arc Pro B-Series GPUs with vLLM

Chinese: [zh/vllm/blog/architecture/intel-arc.md](../../../../zh/vllm/blog/architecture/intel-arc.md)

2025-11-11. **Intel vLLM Team**. XPU / SYCL. Study note; demos on **4–8× Intel Arc Pro B60**, not your SLA. Sleep: [sleep-mode.md](sleep-mode.md). Spec: [spec-decode.md](../performance/spec-decode.md). Hardware out of tree: [hardware-plugin.md](hardware-plugin.md). W4A16 on this card via AutoRound: [autoround-llmc.md](autoround-llmc.md). CPU cousin: [arm-cpus.md](arm-cpus.md). `torch.compile` FP16/BF16 path: [torch-compile.md](torch-compile.md). Docker they name: `intel/vllm:0.10.2-xpu`. Host then: Ubuntu 25.04, KMD 6.14.0. MoE / gpt-oss from the **0.10.2** XPU image.

Fits: serving DeepSeek-distill / Qwen / Llama / GPT-OSS on Arc Pro B60 with persistent MoE kernels, TP, `--enforce-eager`. Does not fit: treating table **1210.74 / 1495.12 tok/s** as a promise — Intel’s own disclaimer is on the page.

[Intel Arc Pro B-Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html): large memory, multi-GPU, price/perf pitch for local LLMs. vLLM is the serving core they name. Months of Intel × vLLM work on features, multi-GPU scaling, PCIe P2P.

Hardware they print for the card: **24 GB** HBM-class VRAM, **456 GB/s** bandwidth, **160** Intel XMX engines. Supported-model list at posting: [intel/ai-containers vllm 0.10.2-xpu](https://github.com/intel/ai-containers/blob/main/vllm/0.10.2-xpu.md#supported-models).

Feature list on the page:

- DeepSeek distilled Llama / Qwen, solid output-token throughput
- Long context **>50K**, batch-size scaling
- Embedding, reranker, pooling
- Multimodal
- MoE (GPT-OSS, DeepSeek-v2-lite, Qwen3-30B-A3B, …)
- Per-layer online quantization (cut GPU memory)
- Data / tensor / pipeline parallelism
- FP16 and BF16 paths for `torch.compile`
- Speculative decoding: n-gram, EAGLE, EAGLE3
- Async scheduling
- Prefill/Decode disaggregation
- LoRA
- Reasoning output
- Sleep mode
- Structured outputs
- Tool calling
- Mixed precision: BF16, FP16, INT4, FP8 recipes

## MoE: don’t launch one GEMM per expert

MoE: a gate picks a subset of expert FFNs per token. Equivalent work is many parallel GEMMs, structured sparsity. Besides GEMM and Flash Attention, experts + gate dominate.

![moe](../../../../assets/vllm/blog/architecture/intel-arc/01-moe.png)

**Figure.** MoE experts and gating (study copy; copyright remains with the original site).

Naive path: a `for` loop launching one GEMM kernel per expert per iteration — launch tax, scheduling latency. Gate must finish before GEMMs start → pipeline stalls, device idle.

They designed a **persistent zero-gap kernel**, claimed **>80%** of Arc Pro B60 hardware capacity.

### 1. Single kernel, persistent loop

One launch; a persistent loop so launch parameters need not wait on the routing network. Keeps device parallelism.

Before: device idle while the host waits.

![persistent kernel1](../../../../assets/vllm/blog/architecture/intel-arc/02-persistent-kernel1.png)

**Figure.** Kernel trace before persistence — host wait / device idle (study copy).

After: device stays busy.

![persistent kernel2](../../../../assets/vllm/blog/architecture/intel-arc/03-persistent-kernel2.png)

**Figure.** Persistent loop keeps the device busy (study copy).

B60 has **20 XeCores**, identical resources, multiple SYCL groups each. Design: **two groups per XeCore**, balancing compute vs memory bandwidth.

### 2. Dynamic steal among compute groups

Expert routing is imbalanced, so groups do unequal work. Fixed stride: the slowest group sets the pace; the gap accumulates up to **~15%** of total MoE GEMM time. Better: whoever finishes a loop takes the next available block.

Concrete: 40 groups, 200 GEMM blocks. Static stride → group 0 does 0, 40, 80, …; group 1 does 1, 41, 81, …. MoE blocks are not equal intensity; random access lets some groups finish early and sit idle.

| Before | After |
| --- | --- |
| ![thread load1](../../../../assets/vllm/blog/architecture/intel-arc/04-thread-load1.png) | ![thread load2](../../../../assets/vllm/blog/architecture/intel-arc/05-thread-load2.png) |

**Figure.** Thread load before / after atomic steal (study copy).

Fix: groups compete for the next job through an **atomic counter**. Finish one GEMM block → take a rank from the atomic → that rank is the next block. Small loop gaps gone; they claim even scheduling across expert-routing patterns.

### 3. Fast MXFP4 → BF16, with prepack

Prepack for load efficiency. For 4-bit loads, a hardware-friendly layout increased efficiency up to **~30%** in their case. Naive FP4→BF16 is instruction-heavy. Alternative (borrowed from oneDNN: stride E2M1 encoding onto single-precision E/M positions, multiply by the scale gap):

`Bitcast-bf16 ((x << 12) >> 6 & 0x81c0) * 2^126`

Minimizes the convert.

## Performance (demos on the page)

DeepSeek distilled **8B–70B**, FP8, eight Arc Pro cards — output-token throughput in Figure 1.

![perf figure1](../../../../assets/vllm/blog/architecture/intel-arc/06-perf-figure1.png)

**Figure 1.** FP8 output-token throughput at max concurrency under SLA, **8× Arc Pro B60** (study copy).

Next-token latency held **<100 ms** under load (Figure 2, Qwen-32B, **4× B60**, increasing prompt count).

![perf figure2](../../../../assets/vllm/blog/architecture/intel-arc/07-perf-figure2.png)

**Figure 2.** Qwen-32B next-token latency vs number of prompts, **4× Arc Pro B60** (study copy).

Llama-70B, single batch, input **1K–40K**: TTFT / TPOT stay consistent. They credit Flash Attention kernels that parallelize along the sequence dimension.

![perf figure3](../../../../assets/vllm/blog/architecture/intel-arc/08-perf-figure3.png)

**Figure 3.** Llama-70B single-batch TTFT / TPOT, 1K–40K input, **8× Arc Pro B60** (study copy).

GPT-OSS MXFP4 on an x8 Arc Pro B-series system (1–4 GPUs):

| Model | Data type | TP | Input/output seq | Concurrency | TTFT (s) | TPOT (ms) | Output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS-20b | MXFP4 | 1 | 1024/1024 | 75 | **7.614** | **53.96** | **1210.74** |
| GPT-OSS-20b | MXFP4 | 1 | 2048/2048 | 38 | 7.823 | 42.35 | 818.92 |
| GPT-OSS-20b | MXFP4 | 1 | 5120/5120 | 15 | 8.36 | 34.27 | 416.94 |
| GPT-OSS-120b | MXFP4 | 4 | 1024/1024 | 100 | 8.04 | 58.78 | **1495.12** |
| GPT-OSS-120b | MXFP4 | 4 | 2048/2048 | 50 | 8.11 | 41.98 | 1085.58 |
| GPT-OSS-120b | MXFP4 | 4 | 5120/5120 | 20 | 8.60 | 30.60 | 619.10 |

**Table 1.** GPT-OSS vLLM inference on 1–4 GPUs, x8 Arc Pro B-series system (page numbers).

MLPerf Inference **v5.1**: Llama 8B price/perf mention for B60, vLLM as the serving framework. Link they print: [MLCommons inference datacenter](https://mlcommons.org/benchmarks/inference-datacenter/).

## How to set up

Image: [intel/vllm on Docker Hub](https://hub.docker.com/r/intel/vllm). MoE / gpt-oss since **vllm 0.10.2** docker. Examples assume host **Ubuntu 25.04**, KMD **6.14.0**, Xeon with **4× Arc Pro B60** in PCIe slots.

```bash
docker pull intel/vllm:0.10.2-xpu
```

```bash
docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
  -v /dev/dri/by-path:/dev/dri/by-path --name=vllm-test \
  --device /dev/dri:/dev/dri --entrypoint= intel/vllm:0.10.2-xpu /bin/bash
```

gpt-oss-120b on 4× B60:

```bash
vllm serve openai/gpt-oss-120b --dtype=bfloat16 --enforce-eager \
  --port 8000 --host 0.0.0.0 --trust-remote-code \
  --gpu-memory-util=0.9 --no-enable-prefix-caching \
  --max-num-batched-tokens=8192 --disable-log-requests \
  --max-model-len=16384 --block-size 64 -tp 4
```

Another shell, bench:

```bash
vllm bench serve --model openai/gpt-oss-120b \
  --dataset-name sonnet --dataset-path="./benchmarks/sonnet.txt" \
  --sonnet-input-len=1024 --sonnet-output-len=1024 --ignore-eos \
  --num-prompt 1 --trust_remote_code --request-rate inf \
  --backend vllm --port=8000 --host 0.0.0.0
```

Validated models: [Supported Models](https://github.com/intel/ai-containers/blob/main/vllm/0.10.2-xpu.md#supported-models).

## Looking ahead

Deeper integration with core vLLM. Roadmap on the page: full upstream feature support; SOTA opts for popular LLMs on Intel hardware; contribute back upstream.

## Acknowledgement

vLLM team — partnership named.

## Notices they print

Performance varies by use, configuration, and other factors: [Intel Performance Index](http://www.intel.com/PerformanceIndex). Results as of the dates shown; may not include later updates. See [MLCommons](https://mlcommons.org/). No product is absolutely secure. Intel technologies may need enabled hardware, software, or service activation.
