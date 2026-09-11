---
source: https://vllm.ai/blog/2025-04-05-llama4
lang: en
fetched: 2026-09-04
---

# Llama 4: Scout 16E / Maverick 128E, iRoPE 1:3, v0.8.3+

Chinese: [zh/vllm/blog/serving/llama4.md](../../../../zh/vllm/blog/serving/llama4.md)

2025-04-05. **The vLLM Team**. **One** expert per token (17B active). Prior 405B: [llama31.md](llama31.md). V1 engine named in the cluster-scale pitch: [v1-alpha.md](../architecture/v1-alpha.md). Distributed: [distributed-inference.md](distributed-inference.md). `VLLM_DISABLE_COMPILE_CACHE=1` is a then-launch flag. Chart TPS is their plate, not your SLA.

**TL;DR from the page:**

- Scout 17B-16E, Maverick 17B-128E. Native multimodal (8–10 images “with good results”).
- 8×H100: Scout `--max-model-len 1000000` (they suggest `attn_temperature_tuning: true`); Maverick-FP8 ~**430K**.
- 8×H200: Scout 3.6M, Maverick 1M.
- Multi-image: `--limit-mm-per-prompt image=10` (default 1). `--kv-cache-dtype fp8` can roughly double the window; they saw little eval drop.
- iRoPE: global no-RoPE vs chunked local RoPE at **1:3**. Maverick MMLU-Pro reported 80.5, H100 FP8 **80.4**.

## Usage guide

[Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/): Scout and Maverick. Install `v0.8.3` or later: `pip install -U vllm`. CLI, [docker](https://docs.vllm.ai/en/latest/deployment/docker.html), or the Pythonic [`LLM` class](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference). Meta 1M-context demo: [llama-cookbook notebook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynb).

### 8× H100

Scout (up to 1M; `attn_temperature_tuning: true`):

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1000000 --override-generation-config='{"attn_temperature_tuning": true}'
```

Maverick-FP8 (up to ~430K):

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --max-model-len 430000
```

### 8× H200

Scout (up to 3.6M):

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 3600000
```

Maverick (up to 1M). Page is missing a `\` after `--tensor-parallel-size 8`:

```
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 8
  --max-model-len 1000000
```

**Multimodality.** 8–10 images. Default: 1 image per request. Pass `--limit-mm-per-prompt image=10` for the OpenAI-compatible API. Offline multi-image example (v0.8.3): [vision_language_multi_image.py](https://github.com/vllm-project/vllm/blob/v0.8.3/examples/offline_inference/vision_language_multi_image.py).

**Performance.** Scout-BF16 and Maverick-FP8 output tok/s under the configs above:

Local figures (copyright remains with the original site; study copies):

![perf](../../../../assets/vllm/blog/serving/llama4/01-perf.png)

**Figure.** Output tokens/s; numbers live in the chart, not a table. More enhancements “on the way”; architecture + relatively small size billed as already practical.

**Tips for performance and long context:**

- `--kv-cache-dtype fp8` — potentially double the usable window and a performance boost; little-to-no eval drop claimed.
- Scout up to **10M**: multi-node TP or PP. Guide: [distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html).

**Other hardware and quantizations:**

- A100: BF16 verified.
- INT4 Scout on a single H100: work in progress then.
- AMD MI300X: build [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html?device=rocm), same commands.

**Inference accuracy** vs Meta (lm-eval-harness) for [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct):

| | MMLU Pro | ChartQA |
|----------|---------|---------|
| Reported | 80.5 | 90 |
| H100 FP8 | 80.4 | 89.4 |
| AMD MI300x BF16 | 80.4 | 89.4 |
| H200 BF16 | 80.2 | 89.3 |

## Efficient architecture and cluster-scale serving

- **MoE:** Scout 16 experts, Maverick 128; **17B** activated; **one** expert per token.
- **iRoPE:** global attention (no RoPE) interleaved with chunked local attention (with RoPE) at **1:3**. Local layer attends inside non-overlapping chunks — quadratic cost grows slower.

V1 engine: single-node speedups + native torch.compile. Then-Q2 roadmap: multi-node scale — disaggregated cluster serving, expert parallelism, multi-node data parallelism, cluster-wide prefill disaggregation. Tracking issue named: [vllm#15735](https://github.com/vllm-project/vllm/issues/15735).

## Acknowledgement

Meta (architecture, accuracy, benches): Lucia (Lu) Fang, Ye (Charlotte) Qi, Lu Fang, Yang Chen, Zijing Liu, Yong Hoon Shin, Zhewen Li, Jon Swenson, Kai Wu, Xiaodong Wang, Shiyan Deng, Wenchen Wang, Lai Wei, Matthias Reso, Chris Thi, Keyun Tong, Jinho Hwang, Driss Guessous, Aston Zhang.

AMD MI300X: Hongxia Yang, Weijun Jiang.

vLLM benches on hardware from Nebius and NVIDIA.
