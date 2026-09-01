---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: en
fetched: 2026-09-01
---

# Quickstart — vLLM

Chinese: `../../zh/vllm/getting-started/quickstart.md`  
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

Other stacks have their own install pages: ROCm, Intel XPU, TPU (`vllm-tpu`), Ascend, Apple Silicon (vLLM-Metal / MLX). Wrong stack means every later stopwatch is measuring an install accident.

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

`opt-125m` exists so the example lives on a laptop, not as a production model.

Default sampling reads Hugging Face `generation_config.json` when present. If that file pins temperature=0, your `0.8` in code may lose. Force vLLM defaults: `LLM(..., generation_config="vllm")`.

Instruct / Chat models need a chat template or `llm.chat(...)`. Feeding dialogue as raw completion without the template measures a different model.

## Online serving

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Default `http://localhost:8000`. `--host` / `--port` move it. One model per process. OpenAI-compatible: `/v1/models`, chat completions, completions.

```bash
curl http://localhost:8000/v1/models
```

Auth: `--api-key` or `VLLM_API_KEY`. `--api-key` accepts several keys; any one passes — for rotation. No key is convenient in the lab and dangerous in the lobby.

Attention backend is usually auto. To pin:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN
```

Same idea for `FLASHINFER`. A wrong backend shows up as worse ITL or a failed start — read which one the log actually loaded.

Next is not more CLI: `optimization.md` (CPU cores → `-O*` → `max_num_batched_tokens` → parallelism and cache).
