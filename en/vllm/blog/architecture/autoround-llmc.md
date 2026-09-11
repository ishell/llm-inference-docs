---
source: https://vllm.ai/blog/2025-12-09-intel-autoround-llmc
lang: en
fetched: 2026-09-04
---

# Advancing Low-Bit Quantization for LLMs: AutoRound × LLM Compressor

Chinese: [zh/vllm/blog/architecture/autoround-llmc.md](../../../../zh/vllm/blog/architecture/autoround-llmc.md)

2025-12-09. **Intel Neural Compressor Team, Red Hat AI Model Optimization Team**. Study note of the first `AutoRoundModifier` drop into [LLM Compressor](https://github.com/vllm-project/llm-compressor). Omni sibling (quantize once, serve W4A16 in vLLM-Omni): [omni-autoround.md](../serving/omni-autoround.md). Arm INT8 / W4A8 cousin: [arm-cpus.md](arm-cpus.md). Arc XPU serving: [intel-arc.md](intel-arc.md). GSM8K **0.911** is their `lm_eval` demo (5-shot, `--limit 1000`) — they print that it fluctuates; not an SLA.

Fits: W4A16 PTQ of dense Llama / Qwen into `compressed-tensors`, then `vllm serve`. Quantize on a workstation GPU, serve on another box (they name a single **Intel Arc Pro B60**). Does not fit: XPU without `--enforce-eager`, or expecting this first stage to cover FP8 / MXFP4 / NVFP4 / MoE — those were roadmap.

Paper: [AutoRound (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.662.pdf). Code: [intel/auto-round](https://github.com/intel/auto-round).

## TL;DR they print

AutoRound — Intel’s tuning-based post-training quantization (PTQ) — lands in LLM Compressor:

- Higher accuracy at low bit-width
- Lightweight tuning (hundreds of steps, not thousands)
- Zero extra inference overhead
- `compressed-tensors` checkpoints, served directly in [vLLM](https://github.com/vllm-project/vllm)
- Quantize and serve in a few lines

Broader schemes and model coverage were “coming next” on the page.

## What AutoRound is

PTQ for LLMs and VLMs. Three trainable parameters **per quantized tensor**:

- `V` — rounding offset / adjustment
- `α` and `β` — learned clipping-range controls

Decoder layers go **sequentially**. Signed gradient descent jointly tunes rounding and clipping to minimize **block-wise output reconstruction error**.

Strengths they name:

- Accuracy at very low bit-width
- Multiple dtypes: W4A16, MXFP8, MXFP4, FP8, NVFP4, more on the way
- Mixed-bit, layer-wise precision search
- LLMs **and** VLMs

Target hardware on the page: Intel Xeon, Intel Gaudi, Intel Data Center GPUs, Intel Arc B-Series, and other GPUs (CUDA named). Looking forward: native FP8 / MXFP8 / MXFP4 on next-gen Data Center GPUs, codenamed **Crescent Island** — AutoRound checkpoints meant to ride that path.

## Why LLM Compressor

LLM Compressor already owns unified compression primitives (quantization, pruning). Putting AutoRound there:

- Matches existing modifier architecture (e.g. `GPTQModifier`)
- Reuses sequential calibration and layer-onloading
- Future multi-modifier recipes
- Checkpoints ready for vLLM — compression to deployment in one workflow

## Integration (first stage)

New `AutoRoundModifier` in LLM Compressor, producing `W{n}A16` (they demo **W4A16**) that vLLM loads. First-stage PR: [llm-compressor#1994](https://github.com/vllm-project/llm-compressor/pull/1994). Config is model + calibration data. Dense LLMs in the **Llama** and **Qwen** families.

## Quickstart they print

### 1. Install

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

### 2. Load model and tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_ID = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 3. Calibration data

```python
from auto_round.calib_dataset import get_dataset
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ds = get_dataset(tokenizer=tokenizer,
                 seqlen=MAX_SEQUENCE_LENGTH,
                 nsamples=NUM_CALIBRATION_SAMPLES)
```

### 4. Quantize

Runs on CPU or GPU. Quantization and serving need not share a device — workstation GPU now, AIPC later, their example.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    iters=200,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

SAVE_DIR = MODEL_ID.split("/")[-1] + "-W4A16-G128-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Practice on the page: **128 calibration samples + ~200 iterations** often converges. Raise samples or iters for extremely low bits or tighter accuracy.

### 5. Serve in vLLM

Same compressed checkpoint on different hardware than tuning. Example: quantized `Qwen3-8B-W4A16-G128-AutoRound` on one **Intel Arc Pro B60**:

```bash
vllm serve Qwen3-8B-W4A16-G128-AutoRound \
    --dtype=bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-num-batched-tokens 8192
```

**Caveats they print:** install vLLM from [PR #29484](https://github.com/vllm-project/vllm/pull/29484/). On **XPU**, `--enforce-eager` is required.

### 6. Evaluate (GSM8K via `lm_eval`)

```bash
lm_eval --model vllm \
  --model_args pretrained="./Qwen3-8B-W4A16-G128-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enforce_eager=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 1000 \
  --batch_size 128
```

They print:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.911|±  | 0.009|
|     |       |strict-match    |     5|exact_match|↑  |0.911|±  | 0.009|
```

Note on the page: results may fluctuate (non-determinism).

## Conclusion and plans

First integration: W4A16 end-to-end, simple config, dense Llama / Qwen. Roadmap they name: FP8, MXFP4, MXFP8, NVFP4; automatic mixed-bit search; MoE families; deeper multi-modifier recipes with other LLM Compressor algorithms.

Priorities: [RFC #1968](https://github.com/vllm-project/llm-compressor/issues/1968), or the Intel Community.

## Acknowledgements

LLM Compressor and vLLM community. Named: Kyle Sayers, Dipika Sikka, Brian Dellabetta, Charles Hernandez, Robert Shaw, Kunshang Ji — early proposal and PR review.

### Related RFCs and PRs

[llm-compressor#1968](https://github.com/vllm-project/llm-compressor/issues/1968), [llm-compressor#1994](https://github.com/vllm-project/llm-compressor/pull/1994), [llm-compressor#2055](https://github.com/vllm-project/llm-compressor/pull/2055), [llm-compressor#2062](https://github.com/vllm-project/llm-compressor/pull/2062), [auto-round#993](https://github.com/intel/auto-round/pull/993), [auto-round#1053](https://github.com/intel/auto-round/pull/1053), [auto-round#1055](https://github.com/intel/auto-round/pull/1055), [auto-round#1072](https://github.com/intel/auto-round/pull/1072), [vllm#29484](https://github.com/vllm-project/vllm/pull/29484).
