---
source: https://vllm.ai/blog/2026-06-02-vllm-omni-autoround
lang: en
fetched: 2026-09-04
---

# Accelerating vLLM-Omni Inference with AutoRound Quantization

Chinese: [zh/vllm/blog/serving/omni-autoround.md](../../../../zh/vllm/blog/serving/omni-autoround.md)

2026-06-02. **vLLM-Omni Community, Intel AutoRound Team**. [AutoRound](https://github.com/intel/auto-round) PTQ into [vLLM-Omni](https://github.com/vllm-project/vllm-omni): quantize once, serve W4A16 (4-bit weight / 16-bit activation). LLM Compressor sibling (compressed-tensors into vLLM): [autoround-llmc.md](../architecture/autoround-llmc.md). Same Omni line: [vllm-omni.md](vllm-omni.md), [qwen3-omni.md](qwen3-omni.md). Study note; OmniBench / TIIF / B60 numbers on the page, not your SLA.

**TL;DR from the page:**

- Reads `quantization_config.quant_method = "auto-round"` — **no** extra `--quantization` at serve time.
- Qwen3-Omni-30B-A3B: **66 GB → 25 GB** (~**62%** checkpoint shrink).
- OmniBench W4A16 slightly **above** BF16 on 100 image+audio tasks (evalscope). TIIF average drift ~**1.3%** across 9 sub-attributes.
- Intel B60: FLUX.1-dev BF16 transformer **23 GB** needs TP=4 on **24.4 GB** cards; W4A16 **7 GB** fits one card (~**19%** headroom). Leftover GPUs run CFG Parallel → guided gen ~**1.55–1.67×** vs sequential BF16.
- Wan2.2 / GLM-Image / FLUX live; BAGEL / Ovis had checkpoints, runtime still landing. Quantize offline; the hot path only infers. Verified on Intel XPU and NVIDIA GPU.

Figures stay on the original page (no local copies). Captions:

**Figure 1.** OmniBench: BF16 vs W4A16 AutoRound on Qwen3-Omni-30B-A3B-Instruct.

**Figure 2.** TIIF-Bench across 9 structural sub-attributes for multi-stage T2I.

**Figure 3.** Wan2.2 T2V-A14B text-to-video under W4A16 AutoRound.

**Figure 4.** Wan2.2 I2V-A14B image-to-video under W4A16 AutoRound.

**Figure 5.** VRAM footprint, BF16 vs W4A16 AutoRound, across vLLM-Omni families.

**Figure 6.** Latency/memory tradeoff: W4A16 drops FLUX min hardware from 4 GPUs to 1, enabling CFG Parallel.

**Figure 7.** CFG Parallel on Intel XPU B60: **1.55–1.67×** vs sequential BF16.

Body tables do **not** reprint OmniBench/TIIF/VRAM cell values; those live in the figures.

## 1. Introduction: vLLM-Omni meets AutoRound

vLLM-Omni serves diffusion, multimodal Omni, and multi-stage generation. Quantization here is not “shrink one transformer” — it is making a **diverse runtime portfolio** fit real cards.

AutoRound (Intel; EMNLP 2024, weight rounding via signed gradient descent) is tuning-based PTQ. Per quantized tensor, three learnables: `V` (rounding offset), `alpha` / `beta` (clip range). Stronger low-bit accuracy than naive round-to-nearest; the checkpoint is static — **zero** extra quantization work on the inference path. Three layers on the page: algorithm (AutoRound) + runtime (Omni) + INT4 catalog on Hugging Face.

Runtime is checkpoint-driven. Omni reads metadata, sees `quantization_config.quant_method = "auto-round"`, remaps blocks to runtime modules, picks the compute backend. Serving API matches a normal load.

## 2. Model coverage

Three paradigms on the page.

### 2.1 Omni multimodal

Unified text / vision / audio; cross-modal embedding alignment is the quantization hazard.

| Model | Checkpoint | Status |
|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | [Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) | Integrated and validated |
| Qwen2.5-Omni-7B | [Intel/Qwen2.5-Omni-7B-int4-AutoRound](https://huggingface.co/Intel/Qwen2.5-Omni-7B-int4-AutoRound) | Integrated and validated |

### 2.2 Diffusion and multi-stage image

| Model | Checkpoint | Status |
|---|---|---|
| GLM-Image | [Intel/GLM-Image-int4-AutoRound](https://huggingface.co/Intel/GLM-Image-int4-AutoRound) | Integrated and validated |
| FLUX.1-dev | [vllm-project-org/FLUX.1-dev-AutoRound-w4a16](https://huggingface.co/vllm-project-org/FLUX.1-dev-AutoRound-w4a16) | Integrated and validated |
| BAGEL-7B-MoT | [Intel/BAGEL-7B-MoT-int4-AutoRound](https://huggingface.co/Intel/BAGEL-7B-MoT-int4-AutoRound) | Checkpoint; runtime in progress |
| Ovis-Image-7B | [Intel/Ovis-Image-7B-int4-AutoRound](https://huggingface.co/Intel/Ovis-Image-7B-int4-AutoRound) | Checkpoint; runtime in progress |

### 2.3 Video diffusion (Wan2.2)

Spatio-temporal video; INT4 checkpoints validated in Omni:

- [Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound)
- [Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound)
- [Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound](https://huggingface.co/Intel/Wan2.2-TI2V-5B-Diffusers-int4-AutoRound)

## 3. Usage

Quantize and tune **offline**. Production code only infers. No calibration on the serving path.

### 3.1 Inference with a quantized model

FLUX.1-dev Python API is a normal Omni load — only the checkpoint path changes:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == '__main__':
    omni = Omni(model="vllm-project-org/FLUX.1-dev-AutoRound-w4a16")
    outputs = omni.generate(
        "A cat sitting on a windowsill",
        OmniDiffusionSamplingParams(num_inference_steps=28, guidance_scale=3.5),
    )
    outputs[0].images[0].save("output.png")
```

Wan2.2: standard `vllm serve`, same video endpoint as BF16.

```bash
vllm serve Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound --omni --port 8091
```

```bash
curl -X POST "http://127.0.0.1:8091/v1/videos/sync" \
  -F 'prompt=Cherry blossoms swaying gently in the breeze, cinematic motion' \
  -F 'width=832' -F 'height=480' -F 'num_frames=48' \
  -F 'num_inference_steps=40' -F 'guidance_scale=5.0' \
  --output t2v_output.mp4
```

Qwen2.5-Omni: OpenAI-compatible chat, unchanged.

```bash
vllm serve Intel/Qwen2.5-Omni-7B-int4-AutoRound --omni --port 8091
```

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/Qwen2.5-Omni-7B-int4-AutoRound",
    "messages": [{"role": "user", "content": "What is 2 + 3?"}],
    "max_tokens": 128
  }'
```

Omni auto-detects quantization metadata. Pre-quantized AutoRound: **do not** add `--quantization`.

### 3.2 Quantizing a new model

Offline AutoRound → serve. The three recipes on the page are **not** the same recipe: FLUX uses `--iters 0` and `--disable_opt_rtn` (no signed-GD tuning loop); Wan uses `--iters 100` / `--nsamples 32`; Qwen3-Omni uses `--bits 4 --group_size 128 --iters 200 --lr 5e-3`.

```bash
# FLUX.1-dev
auto-round \
  --model black-forest-labs/FLUX.1-dev \
  --scheme W4A16 \
  --batch_size 1 \
  --disable_opt_rtn \
  --dataset coco2014 \
  --iters 0

# Wan2.2-T2V-A14B
auto-round \
  --model_name Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --format auto_round \
  --scheme W4A16 \
  --iters 100 \
  --nsamples 32 \
  --batch_size 1 \
  --num-inference-steps 3 \
  --guidance-scale 5.0 \
  --dataset coco2014 \
  --output_dir Wan2.2-T2V-A14B-Diffusers-int4-AutoRound

# Qwen3-Omni-30B-A3B-Instruct
auto-round \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --bits 4 \
  --group_size 128 \
  --format auto_round \
  --iters 200 \
  --lr 5e-3 \
  --output_dir tmp_qwen3_omni_w4a16 \
  --trust_remote_code
```

Checkpoint metadata in `config.json`:

```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "packing_format": "auto_round:auto_gptq"
  }
}
```

Guidance on the page: **128** calibration samples and ~**200** optimization iterations are often enough; larger / more sensitive models may need more. Exact settings depend on family, task, and deployment. That “often enough” line is **not** what the FLUX snippet uses (`iters 0`).

### 3.3 Quality validation

Diffusion: same-seed regression vs BF16.

```bash
python -m vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity \
  --task t2i \
  --reference-model black-forest-labs/FLUX.1-dev \
  --candidate-model vllm-project-org/FLUX.1-dev-AutoRound-w4a16 \
  --prompt "a cup of coffee on the table" \
  --height 512 --width 512 \
  --num-inference-steps 20 \
  --seed 142 \
  --output-json /tmp/flux_similarity/result.json
```

The post does not print the JSON fields or a pass/fail threshold for that tool.

## 4. Quantitative evaluation: accuracy and quality

### 4.1 Omni multimodal (OmniBench)

evalscope, **100** multimodal tasks, image **and** audio together. W4A16 aggregate OmniBench score slightly **higher** than BF16. No numeric OmniBench totals in the body. **Figure 1.**

### 4.2 Multi-stage diffusion (TIIF-Bench)

Nine structural sub-attributes: alignment, composition, fidelity. Average accuracy degradation ~**1.3%**. The nine axis names are not listed in prose. **Figure 2.**

### 4.3 Video (Wan2.2)

Naive scalar quant tends to break temporal consistency. Objective metrics on T2V-A14B (**Figure 3**) and I2V-A14B (**Figure 4**). Under W4A16 AutoRound, T2V-A14B showed **marginal gains** on structural-consistency metrics — the page’s hypothesis: clip optimization can act as regularization. Metric names and deltas are in the figures, not a table.

## 5. Performance, footprint, serving

### 5.1 VRAM footprint

First-order win: checkpoint size and execution memory. W4A16 takes quantized weights from BF16 down to roughly **¼** of the weight footprint. End-to-end speedup then depends on whether the workload was capacity- or bandwidth-bound. **Figure 5.**

Not every stage is quantized. VAE decode, auxiliary stages, parts of multi-stage systems may stay higher precision — weight-compression ratio is usually **larger** than E2E latency speedup. Do not read the 66→25 GB Omni shrink as “every byte of every stage is INT4.”

### 5.2 Trading memory headroom for latency

Benches in this case study: **Intel XPU B60**. Not a NVIDIA-GPU CFG-Parallel table.

**Min hardware: 4 GPUs → 1 GPU.** BF16 FLUX.1-dev transformer **23 GB** does not fit a single B60 (**24.4 GB**) once activations are in — TP=4. W4A16 transformer **7 GB** fits one GPU with ~**19%** headroom.

**W4A16 + CFG Parallel = 1.55×–1.67× guided generation.** Classifier-Free Guidance runs two denoising passes per step (prompt + negative). BF16 occupies all 4 GPUs for tensor parallelism → those passes run **sequentially** (2× latency). W4A16 fits TP=2, freeing 2 GPUs, so both guidance branches run at once across two GPU groups. **Figure 6** (hardware drop + CFG Parallel). **Figure 7** (B60 CFG Parallel latency). The **1.55–1.67×** is vs sequential BF16 on that B60 layout, not vs a 1-GPU BF16 that does not fit.

The claim is not only “fits”: memory headroom lets diffusion **run differently**, unlocking parallelism larger than raw dequantization savings would predict.

## 6. Conclusion

Operator-shaped path: offline checkpoints, automatic detection, predictable memory, a quality check before rollout. Coverage named: FLUX, Wan, GLM, BAGEL, Ovis, Qwen Omni. BAGEL and Ovis were checkpoint-ready, runtime not yet.

Ongoing: **MXFP4** and **MXFP8** for Linear and MoE; low-bit attention (e.g. SageAttention).

## 7. Acknowledgements

Hongsheng Liu, Shunyang Li, WeiQing Chen (vLLM-Omni); Chendi Xue (Intel). vLLM-Omni community for adopting AutoRound.
