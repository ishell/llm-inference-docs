---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: en
fetched: 2026-08-31
---

# vLLM Quickstart

Linux, Python 3.10–3.13. NVIDIA install:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Also: ROCm, Intel XPU, TPU (`vllm-tpu`), Ascend, Apple Silicon (vLLM-Metal / MLX).

**Offline:**

```python
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8, top_p=0.95))
```

Default sampling comes from HF `generation_config.json` if present. Force vLLM defaults: `generation_config="vllm"`. Chat/Instruct models need a chat template or `llm.chat(...)`.

**Online:** `vllm serve Qwen/Qwen2.5-1.5B-Instruct` → OpenAI Completions + Chat at `localhost:8000`. `--api-key` / `VLLM_API_KEY` for auth. Attention backend is auto; override `--attention-backend FLASH_ATTN` / `FLASHINFER`.
