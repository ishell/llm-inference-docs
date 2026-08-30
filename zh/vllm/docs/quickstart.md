---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: zh
fetched: 2026-08-30
---

# Quickstart — vLLM（中文摘译）

英文全文：尚未单独精修，官方页：https://docs.vllm.ai/en/stable/getting_started/quickstart/

## 安装（NVIDIA GPU）

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

也支持 ROCm、Intel XPU、TPU（`vllm-tpu`）、昇腾、Apple Silicon（vLLM-Metal/MLX）。详见官方安装页。

## 离线批推理

```python
from vllm import LLM, SamplingParams
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
```

默认会用 Hugging Face 上的 `generation_config.json`。想用 vLLM 默认采样：创建 LLM 时 `generation_config="vllm"`。Instruct/Chat 模型要用 chat template，或走 `llm.chat(...)`。

## 在线 serving

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
curl http://localhost:8000/v1/models
```

OpenAI Completions / Chat Completions 都兼容。`--api-key` 或 `VLLM_API_KEY` 可开鉴权。

Attention backend 一般自动选，也可 `--attention-backend FLASH_ATTN` / `FLASHINFER` 等。
