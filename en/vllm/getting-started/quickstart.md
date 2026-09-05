---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: en
fetched: 2026-09-04
---

# Quickstart — vLLM

Chinese: [zh/vllm/getting-started/quickstart.md](../../../zh/vllm/getting-started/quickstart.md)  
Knob order: `../optimization/optimization.md`. Perf-related `vllm serve` flags: `serve.md` (the generated CLI page is not copied).

Linux; Python **3.10–3.13**. Official NVIDIA path uses `uv`:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` lets the installer pick a PyTorch CUDA build. No permanent env:

```bash
uv run --with vllm vllm --help
```

Other stacks (wrong stack means every later stopwatch is measuring an install accident):

- **AMD ROCm:** `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`. Page: Python 3.12, ROCm 7.0, `glibc >= 2.35`. `rocm/vllm-dev` images are being deprecated; nightly `vllm/vllm-openai-rocm:nightly`.
- **Intel GPU (XPU):** prebuilt wheels “soon”. Official Docker from **v0.26.0**; nightly `vllm/vllm-openai-xpu:nightly`. Details: GPU install guide, Intel XPU tab.
- **Google TPU:** `uv pip install vllm-tpu`. Docker / source / troubleshooting: [vLLM on TPU](https://docs.vllm.ai/projects/tpu/en/latest/).
- **Ascend NPU:** community plugin [vLLM Ascend](https://github.com/vllm-project/vllm-ascend). Hardware + CANN: [Ascend docs](https://docs.vllm.ai/projects/ascend/en/latest/).
- **Apple Silicon:** [vLLM-Metal](https://github.com/vllm-project/vllm-metal) via Metal; compute backend is **MLX**, not PyTorch; models from [mlx-community](https://huggingface.co/mlx-community).

Conda works too: create the env, `pip install --upgrade uv`, then the same `uv pip install vllm --torch-backend=auto`. Non-CUDA hub: [installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/). macOS Metal is also under the GPU install guide’s Apple Silicon tab. `--torch-backend=auto` inspects the CUDA driver; pin e.g. `cu126` with `--torch-backend=cu126` (or `UV_TORCH_BACKEND`).

## Offline batched inference

No HTTP server. `generate` in-process:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt, output.outputs[0].text)
```

`opt-125m` exists so the example lives on a laptop, not as a production model. Supported-model table is on the official Models page. Weights default to Hugging Face; ModelScope: `export VLLM_USE_MODELSCOPE=True` before constructing the engine.

Default sampling reads Hugging Face `generation_config.json` when present. That is usually the creator’s best default — especially if you omit `SamplingParams`. If that file pins temperature=0, your `0.8` in code may lose. Force vLLM defaults: `LLM(..., generation_config="vllm")`.

Instruct / Chat models need a chat template or `llm.chat(...)`. `llm.generate` does **not** apply the template. Feeding dialogue as raw completion measures a different model. Two paths on the page: `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)` then `generate`, or the same OpenAI-shaped `messages` into `llm.chat`. Script: `examples/basic/offline_inference/basic.py`.

## Online serving

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Default `http://localhost:8000`. `--host` / `--port` move it. One model per process. OpenAI-compatible: `/v1/models`, chat completions, completions.

```bash
curl http://localhost:8000/v1/models
```

Auth: `--api-key` or `VLLM_API_KEY`. `--api-key` accepts several keys; any one passes — for rotation. No key is convenient in the lab and dangerous in the lobby.

The server also applies repo `generation_config.json` by default. Disable: `--generation-config vllm`. Default chat template lives in the tokenizer; override path is in the online-serving docs.

### Completions

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Python: `openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")` then `client.completions.create(...)`.

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

Same `openai` client, `chat.completions.create`.

Attention backend is usually auto. To pin:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN
# offline: python script.py --attention-backend FLASHINFER
```

Named options on the page: NVIDIA CUDA `FLASH_ATTN` / `FLASHINFER`; ROCm `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `TRITON_MLA`, `ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`; Intel XPU `FLASH_ATTN`, `TRITON_ATTN`, `TRITON_MLA`, `XPU_MLA_SPARSE`, `TORCH_SDPA`, `TURBOQUANT`. There are **no** prebuilt vLLM wheels that already contain FlashInfer — install it first ([FlashInfer docs](https://docs.flashinfer.ai/) or repo `docker/Dockerfile`). A wrong backend shows up as worse ITL or a failed start — read which one the log actually loaded.

Next is not more CLI: `optimization.md` (CPU cores → `-O*` → `max_num_batched_tokens` → parallelism and cache).
