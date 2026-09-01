---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Quickstart — vLLM

英文对照：`en/vllm/getting-started/quickstart.md`  
旋钮顺序：`../optimization/optimization.md`。`vllm serve` 的性能相关旗标：`serve.md`（整页 CLI 不搬）。

Linux；Python **3.10–3.13**。NVIDIA GPU 官方推荐 `uv`：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` 让安装器自己配 PyTorch CUDA。也可以不建永久环境：

```bash
uv run --with vllm vllm --help
```

其它后端各有安装页，这里不抄：ROCm、Intel XPU、TPU（包名 `vllm-tpu`）、昇腾、Apple Silicon（vLLM-Metal / MLX）。装错栈，后面所有秒表都在测安装事故。

## 离线批推理

服务还没起来。进程里直接 `generate`：

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

`opt-125m` 只是为了让例子在笔记本上活下来，不是生产模型。

默认采样会去读 Hugging Face 上的 `generation_config.json`。仓库里写了 temperature=0 时，你在代码里设 0.8 也可能被盖掉。要 vLLM 自己的默认：`LLM(..., generation_config="vllm")`。

Instruct / Chat 模型必须走 chat template，或 `llm.chat(...)`。把对话当 raw completion 喂进去，模板没套上，测到的是另一种模型。

## 在线 serving

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

默认 `http://localhost:8000`。`--host` / `--port` 改地址。一次进程一只模型。OpenAI 兼容：`/v1/models`、chat completions、completions。

```bash
curl http://localhost:8000/v1/models
```

鉴权：`--api-key` 或环境变量 `VLLM_API_KEY`。`--api-key` 可跟多个 key，任何一个过——给轮换用的。没设 key 时，实验室里方便，门厅里危险。

Attention backend 一般自动选。要钉死：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN
```

`FLASHINFER` 等同理。选错 backend 会表现为 ITL 变差或直接起不来，先看启动日志里实际加载的是哪一个。

下一步不是再抄一遍 CLI，而是 `optimization.md` 的调优顺序：CPU 核 → `-O*` → `max_num_batched_tokens` → 并行与 cache。
