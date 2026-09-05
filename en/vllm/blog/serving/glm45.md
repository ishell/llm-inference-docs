---
source: https://vllm.ai/blog/2025-08-19-glm45-vllm
lang: en
fetched: 2026-09-04
---

# GLM-4.5 Meets vLLM: Built for Intelligent Agents

Chinese: [zh/vllm/blog/serving/glm45.md](../../../../zh/vllm/blog/serving/glm45.md)

2025-08-19. **Yuxuan Zhang**. Production follow-up: [glm52-b300.md](glm52-b300.md). Nightly install; **vLLM V0 is not supported**. Figures on GitHub (`bench.png`, `bench_45v.jpeg`) — not copied here.

[GLM](https://aclanthology.org/2022.acl-long.26/) from Zhipu.ai (now [Z.ai](https://z.ai/)). Long collaboration, back to ChatGLM. This post: [GLM-4.5](https://arxiv.org/abs/2508.06471) and [GLM-4.5V](https://arxiv.org/abs/2507.01006) on NVIDIA Blackwell and Hopper.

| Model | Total / active |
|---|---|
| GLM-4.5 | 355B / 32B |
| GLM-4.5-Air | 106B / 12B |

Hybrid reasoning: **thinking** (complex reasoning + tools) vs **non-thinking** (immediate replies). Their 12-benchmark score: GLM-4.5 **63.2** (3rd among proprietary + open); Air **59.8**. GLM-4.5V is based on Air; they claim SOTA among same-scale models on **42** public VL benchmarks. Repos: [zai-org/GLM-4.5](https://github.com/zai-org/GLM-4.5), [zai-org/GLM-V](https://github.com/zai-org/GLM-V).

## Install (then)

Latest `main`. Nightly vLLM + a preview transformers package:

```shell
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install transformers-v4.55.0-GLM-4.5V-preview
```

## Usage

FP8 and BF16 use the **same** `vllm serve` command.

GLM-4.5 / Air:

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

### Notes they print

- Reasoning lives in `reasoning_content`; `content` is the final answer. Disable thinking: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- 8× H100 OOM on full GLM-4.5: `--cpu-offload-gb 16`
- `flash_infer` trouble: `VLLM_ATTENTION_BACKEND=XFORMERS` as a temp fallback; or set `TORCH_CUDA_ARCH_LIST` (e.g. `'9.0+PTX'`) so FlashInfer can run — **arch string is GPU-specific**
- **vLLM V0 does not support these models**

### Grounding in GLM-4.5V

Prompt for a box; model reasons then returns boxes. Example prompts:

- Help me to locate `<expr>` in the image and give me its bounding boxes.
- Please pinpoint the bounding box `[[x1,y1,x2,y2], …]` in the image as per the given description. `<expr>`

Box is \([x_1,y_1,x_2,y_2]\) top-left / bottom-right; each coordinate normalized by width (x) or height (y) then **×1000**. Special tokens `<|begin_of_box|>` / `<|end_of_box|>`. Bracket style may vary (`[]`, `[[]]`, `()`, `<>`); meaning is the same.

## Cooperation / thanks

vLLM worked with GLM before release so `main` had support on day 0. Names: Kaichao You, Simon Mo, Zifeng Mo, Lucia Fang, Rui Qiao, Jie Li, Ce Gao, Roger Wang, Lu Fang, Wentao Ye, Zixi Qi.
