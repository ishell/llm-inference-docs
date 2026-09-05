---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: en
fetched: 2026-09-04
---

# GLM-4.5 / 4.5V: hybrid thinking, glm45 parsers, V0 not supported then

Chinese: [zh/vllm/blog/serving/glm45.md](../../../../zh/vllm/blog/serving/glm45.md)

2025-08-19. **Yuxuan Zhang**. Nightly then, plus `transformers-v4.55.0-GLM-4.5V-preview`. 355B/32B and Air 106B/12B. FP8/BF16 same serve. Later 5.2 production: [glm52-b300.md](glm52-b300.md). Benchmarks **63.2** / **59.8** are the page’s scores, not your SLA.

**TL;DR from the page:**

- `--tool-call-parser glm45` `--reasoning-parser glm45`. Disable thinking: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- 4.5V: `--allowed-local-media-path` and video `num_frames: -1`.
- Full 4.5 on 8×H100 then often `--cpu-offload-gb 16`. FlashInfer issues: `VLLM_ATTENTION_BACKEND=XFORMERS`.
- **V0 unsupported.** Grounding: `<|begin_of_box|>` boxes; xy normalized by W/H then ×1000.

## Introduction

[GLM](https://aclanthology.org/2022.acl-long.26/) from Zhipu.ai (now [Z.ai](https://z.ai/)). Long vLLM collaboration back to [ChatGLM](https://github.com/zai-org/ChatGLM-6B). This post: [GLM-4.5](https://arxiv.org/abs/2508.06471) and [GLM-4.5V](https://arxiv.org/abs/2507.01006) for intelligent agents — then top-trending on Hugging Face.

GLM-4.5: **355B** total / **32B** active. GLM-4.5-Air: **106B** / **12B**. Unify reasoning, coding, agent capabilities.

Both are hybrid reasoning models: **thinking** mode (complex reasoning + tools) and **non-thinking** (immediate responses).

Page eval across 12 industry benches: GLM-4.5 **63.2** (3rd among proprietary + open); Air **59.8** at higher efficiency.

**Figure bench_45 (not scraped; remote on the original).** Caption: GLM-4.5 / Air composite score vs other models. Source: `https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/bench.png`.

GLM-4.5V is based on Air; continues GLM-4.1V-Thinking. Claimed SOTA among same-scale models on **42** public vision-language benches.

**Figure bench_45v (not scraped; remote on the original).** Caption: GLM-4.5V vs same-scale VLMs. Source: `https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_45v.jpeg`.

Repos: [GLM-4.5](https://github.com/zai-org/GLM-4.5), [GLM-V](https://github.com/zai-org/GLM-V). This post: vLLM on NVIDIA Blackwell and Hopper.

## Installation

`main` then. Nightly vLLM + a preview Transformers:

```shell
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install transformers-v4.55.0-GLM-4.5V-preview
```

## Usage

FP8 and BF16: **same** `vllm serve` command.

GLM-4.5 (Air example, TP 4):

```shell
vllm serve zai-org/GLM-4.5-Air \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice
```

GLM-4.5V:

```shell
vllm serve zai-org/GLM-4.5V \
     --tensor-parallel-size 4   \
     --tool-call-parser glm45   \
     --reasoning-parser glm45   \
     --enable-auto-tool-choice  \
     --allowed-local-media-path / \
     --media-io-kwargs '{"video": {"num_frames": -1}}'
```

### Important notes

- Reasoning wrapped in `reasoning_content`; `content` is the final answer. Disable: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- 8× H100 running full GLM-4.5, OOM → `--cpu-offload-gb 16`.
- `flash_infer` issues: `VLLM_ATTENTION_BACKEND=XFORMERS` as a temporary replacement. Or `TORCH_CUDA_ARCH_LIST='9.0+PTX'` to keep flash_infer — arch string is GPU-dependent.
- **vLLM V0 is not supported** (page wording: “vLLM v0 is not support our model”).

### Grounding in GLM-4.5V

Ask for an object location; the model reasons then emits bounding boxes. Example prompts:

- Help me to locate `<expr>` in the image and give me its bounding boxes.
- Please pinpoint the bounding box `[[x1,y1,x2,y2], …]` in the image as per the given description. `<expr>`

Box is `[x1, y1, x2, y2]` (top-left / bottom-right). Each value: normalize by image width (x) or height (y), then **×1000**.

Special tokens `<|begin_of_box|>` / `<|end_of_box|>` mark the box. Bracket style may vary (`[]`, `[[]]`, `()`, `<>`); meaning is the same.

## Cooperation with vLLM and GLM Team

Before release: vLLM worked with GLM on launch issues so `main` had full GLM-4.5 series support on day-0.

## Acknowledgement

vLLM: Kaichao You, Simon Mo, Zifeng Mo, Lucia Fang, Rui Qiao, Jie Li, Ce Gao, Roger Wang, Lu Fang, Wentao Ye, Zixi Qi.
