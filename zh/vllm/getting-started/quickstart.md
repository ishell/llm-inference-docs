---
source: https://docs.vllm.ai/en/stable/getting_started/quickstart/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Quickstart — vLLM

英文对照：[en/vllm/getting-started/quickstart.md](../../../en/vllm/getting-started/quickstart.md)  
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

其它栈（装错等于后面所有秒表都在测安装事故）：

- **AMD ROCm：** `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`。页上口径：Python 3.12、ROCm 7.0、`glibc >= 2.35`。旧的 `rocm/vllm-dev` 镜像在弃用；夜间镜像 `vllm/vllm-openai-rocm:nightly`。
- **Intel GPU (XPU)：** 预编译轮子「即将有」。官方 Docker 从 **v0.26.0** 起；夜间 `vllm/vllm-openai-xpu:nightly`。细节走 GPU 安装页的 Intel XPU 标签。
- **Google TPU：** `uv pip install vllm-tpu`。Docker / 源码 / 排错：[vLLM on TPU](https://docs.vllm.ai/projects/tpu/en/latest/)。
- **昇腾 NPU：** 社区插件 [vLLM Ascend](https://github.com/vllm-project/vllm-ascend)。硬件和 CANN 版本见 [Ascend 文档](https://docs.vllm.ai/projects/ascend/en/latest/)。
- **Apple Silicon：** [vLLM-Metal](https://github.com/vllm-project/vllm-metal) 走 Metal；计算后端是 **MLX** 不是 PyTorch，模型从 [mlx-community](https://huggingface.co/mlx-community) 取。

也可以 conda 建环境再 `pip install --upgrade uv`，然后同一条 `uv pip install vllm --torch-backend=auto`。非 CUDA 平台总入口：[installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/)。macOS 上 Metal 那条也在 GPU 安装页的 Apple Silicon 标签。`--torch-backend=auto` 会看本机 CUDA 驱动；要钉死例如 `cu126` 就写 `--torch-backend=cu126`（或 `UV_TORCH_BACKEND`）。

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

`opt-125m` 只是为了让例子在笔记本上活下来，不是生产模型。支持模型表在官方 Models 页。默认从 Hugging Face 拉权重；改 ModelScope：初始化引擎前 `export VLLM_USE_MODELSCOPE=True`。

默认采样会去读 Hugging Face 上的 `generation_config.json`。多数时候那是模型作者推荐的最好默认——**没**写 `SamplingParams` 时尤其如此。仓库里写了 temperature=0 时，你在代码里设 0.8 也可能被盖掉。要 vLLM 自己的默认：`LLM(..., generation_config="vllm")`。

Instruct / Chat 模型必须走 chat template，或 `llm.chat(...)`。`llm.generate` **不会**自动套模板。把对话当 raw completion 喂进去，测到的是另一种模型。页上给了两条路：`AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)` 再 `generate`，或把同一份 OpenAI 格式 `messages` 交给 `llm.chat`。脚本对照：`examples/basic/offline_inference/basic.py`。

## 在线 serving

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

默认 `http://localhost:8000`。`--host` / `--port` 改地址。一次进程一只模型。OpenAI 兼容：`/v1/models`、chat completions、completions。

```bash
curl http://localhost:8000/v1/models
```

鉴权：`--api-key` 或环境变量 `VLLM_API_KEY`。`--api-key` 可跟多个 key，任何一个过——给轮换用的。没设 key 时，实验室里方便，门厅里危险。

服务端默认也吃仓库里的 `generation_config.json`。关掉：`--generation-config vllm`。默认 chat template 在 tokenizer 里；覆盖方式见 online serving 文档。

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

Python：`openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")` 再 `client.completions.create(...)`。

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

同一套 `openai` 客户端走 `chat.completions.create`。

Attention backend 一般自动选。要钉死：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN
# 离线脚本：python script.py --attention-backend FLASHINFER
```

页上点名的选项：NVIDIA CUDA 上 `FLASH_ATTN` / `FLASHINFER`；ROCm 上 `TRITON_ATTN`、`ROCM_ATTN`、`ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN`、`TRITON_MLA`、`ROCM_AITER_MLA`、`ROCM_AITER_TRITON_MLA`；Intel XPU 上 `FLASH_ATTN`、`TRITON_ATTN`、`TRITON_MLA`、`XPU_MLA_SPARSE`、`TORCH_SDPA`、`TURBOQUANT`。**没有**预编译的带 FlashInfer 的 vLLM 轮子，要先按 [FlashInfer 文档](https://docs.flashinfer.ai/) 或仓库 `docker/Dockerfile` 装。选错 backend 会表现为 ITL 变差或直接起不来，先看启动日志里实际加载的是哪一个。

下一步不是再抄一遍 CLI，而是 `optimization.md` 的调优顺序：CPU 核 → `-O*` → `max_num_batched_tokens` → 并行与 cache。
