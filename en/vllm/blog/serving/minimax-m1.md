---
source: https://vllm.ai/blog/2025-06-30-minimax-m1
lang: en
fetched: 2026-09-04
---

# MiniMax-M1 Hybrid Architecture Meets vLLM: Long Context, Fast Inference

Chinese: [zh/vllm/blog/serving/minimax-m1.md](../../../../zh/vllm/blog/serving/minimax-m1.md)

2025-06-30. **MiniMax**. Paper: [arXiv 2506.13585](https://arxiv.org/pdf/2506.13585). Checkpoints: [`MiniMaxAI/MiniMax-M1-40k`](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k), [`MiniMaxAI/MiniMax-M1-80k`](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k). **456B** total, ~**45.9B** active. Later MSA / 1M multimodal: [minimax-m3.md](minimax-m3.md). Later Omni: [minimax-h3.md](minimax-h3.md). V1 alpha (the thing they were still migrating toward): [../architecture/v1-alpha.md](../architecture/v1-alpha.md). Hybrid KV relatives: [qwen3-next.md](qwen3-next.md), [hybrid-ssm.md](hybrid-ssm.md). Study note. Architecture + then-deploy; **not** M3 MSA. Docker sample pins `VLLM_USE_V1=0` — **historical**; hybrid allocator later landed in V1.

They quote ~**25%** FLOPs vs DeepSeek R1 at 100k generated tokens. `--quantization experts_int8`. Lightning Attention via Triton. PagedAttention: they claim fragmentation <4% vs traditional 60–80%.

Local figures (copyright remains with the original site; study copies):

![benchmark](../../../../assets/vllm/blog/serving/minimax-m1/01-benchmark.png)

![moe](../../../../assets/vllm/blog/serving/minimax-m1/02-moe.png)

![lightning attention](../../../../assets/vllm/blog/serving/minimax-m1/03-lightning_attention.png)

## Introduction

[MiniMax-M1](https://arxiv.org/pdf/2506.13585) is an open-source large MoE inference model. Hybrid architecture: sparse experts + linearized attention, aimed at long-context reasoning. This post is MiniMax's own write-up of how vLLM served it then — model features, inference pain, and the vLLM pieces they used. Not a bake-off against later MiniMax generations.

**Figure (left).** Benchmarks they print: math, code, SWE, tool use, long-context; MiniMax-M1 “leads among open-source” on that board.

**Figure (right).** Theoretical inference FLOPs vs sequence length. Versus DeepSeek R1, MiniMax-M1 uses **25%** of the FLOPs when generating **100k** tokens. Theoretical curve on the page, not a vLLM measured TPS table.

## Deploying MiniMax-M1 with vLLM

Benefits they list (no numbers beside “outstanding throughput”):

- Throughput
- Memory management
- Batched requests
- Backend optimizations

### Model download

```bash
pip install -U huggingface-hub

huggingface-cli download MiniMaxAI/MiniMax-M1-40k
# huggingface-cli download MiniMaxAI/MiniMax-M1-80k
```

### Deployment (Docker sample as printed)

```bash
IMAGE=vllm/vllm-openai:latest
MODEL_DIR=<model storage path>
NAME=MiniMaxImage

DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --rm --gpus all --ulimit stack=67108864"

sudo docker run -it \
    -v $MODEL_DIR:$MODEL_DIR \
    --name $NAME \
    $DOCKER_RUN_CMD \
    $IMAGE /bin/bash

export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=0
vllm serve \
--model <model storage path> \
--tensor-parallel-size 8 \
--trust-remote-code \
--quantization experts_int8  \
--max_model_len 4096 \
--dtype bfloat16
```

**Caveat they print in the sample, and the later fact:** `VLLM_USE_V1=0` and `--max_model_len 4096` are **what the 2025-06 snippet pinned**. Hybrid allocator + V1 support are named as future work in this same post; they landed later. Do not treat this Docker block as current M3 recipe — that is [minimax-m3.md](minimax-m3.md).

## Hybrid architecture highlights

### Mixture-of-Experts (MoE)

**456B** total. Routing activates a sparse subset ~**45.9B** (~**10%**) per token, from a gating network over token semantics.

They claim classification-style compute cut **up to 90%** versus dense at similar accuracy. That is their architecture claim, not a vLLM kernel microbenchmark.

**Figure.** Isoflop MoE vs dense, both trained on **1T** tokens. Gray dashed lines: compute difference to match performance.

### Lightning Attention

Quadratic softmax attention replaced by a **linear combination of matrix multiplications**, plus dynamic memory tiling and gradient approximation.

Code-completion numbers they print for **100k-token** sequences: memory **−83%**, inference latency **−67%**. Again model-paper numbers, not a vLLM serving table.

**Figure.** Lightning Attention algorithm overview.

### Efficient computation and activation

Lightning Attention for runtime; sparse expert activation to skip unused compute. Pointer back to the [paper](https://arxiv.org/pdf/2506.13585).

## Efficient inference with vLLM

### Advanced memory management

PagedAttention: KV in pages, not one contiguous slab. They quote vLLM's usual claim: waste **<4%** versus traditional **60–80%** fragmentation. Needed for M1's long context so the run does not die on over-allocation.

### Deep kernel-level optimizations

They list vLLM's then-menu: FlashAttention, FlashInfer, GPTQ / AWQ / INT4 / INT8 / FP8. Quantization for memory/compute; FlashAttention for the attention op. **This post's actual M1 flag is `--quantization experts_int8`**, not a full GPTQ/AWQ recipe.

### Lightning Attention in vLLM

Implemented via **Triton**. Triton path runs Lightning Attention's core math inside vLLM — no separate serving engine.

## Future work (as of 2025-06-30)

Two items, both **then-roadmap**:

- **Hybrid allocator** for models that mix attention styles (M1 class).
- **Full vLLM V1** support; migrate the hybrid architecture into V1 ([v1-alpha.md](../architecture/v1-alpha.md)).

Those are the reason the Docker sample still says `VLLM_USE_V1=0`. Historical. M3 day-0 is a different attention (MSA) and a different recipe.

## Conclusion

M1 hybrid = long-context + sparse MoE. vLLM side = paged KV, batching, Triton Lightning Attention, `experts_int8`. Together they meant “can actually serve,” not a 25k TPS/GPU Pareto. Later stack: [minimax-m3.md](minimax-m3.md).

## Acknowledgement

vLLM: [Tyler Michael Smith](https://github.com/tlrmchlsmth), [Simon Mo](https://github.com/simon-mo), [Cyrus Leung](https://github.com/DarkLight1337), [Roger Wang](https://github.com/ywang96), [Zifeng Mo](https://github.com/Isotr0py), [Kaichao You](https://github.com/youkaichao). MiniMax: [Gangying Qing](https://github.com/ZZBoom), [Jun Qing](https://github.com/qscqesze), [Jiaren Cai](https://github.com/sriting).
