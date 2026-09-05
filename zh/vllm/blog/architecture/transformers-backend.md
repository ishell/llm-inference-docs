---
source: https://vllm.ai/blog/2025-04-11-transformers-backend
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Transformers modeling backend：新架构先跑起来

英文对照：[en/vllm/blog/architecture/transformers-backend.md](../../../../en/vllm/blog/architecture/transformers-backend.md)  
原文：https://vllm.ai/blog/2025-04-11-transformers-backend  
2025-04-11。署名 **The Hugging Face Team**。2025-07-21 起支持视觉语言模型。原文没有机制图。这是 **覆盖面**，不是性能默认：原生 `modeling_*.py` 已经能跑，就走原生。

适用：Hub 上还没有 vLLM 原生实现的架构，想先用 PagedAttention 和 continuous batching 伺候起来。不适合：把 `model_impl="transformers"` 当成吞吐开关。

[Transformers](https://huggingface.co/docs/transformers/main/en/index) 是模型生态那一层：研究、微调、统一接口。[vLLM](https://docs.vllm.ai/en/latest/) 是部署那一层：从 Hub 拉模型，为吞吐和时延优化。modeling backend 把 Transformers 的实现接到 vLLM 底下——架构已经在 Transformers 里了，就先用 vLLM 的调度和 KV 去跑。

## 更新：Vision Language Models（2025-07-21）

这篇 2025-04-11 首发。本节是后来补上的。`model_impl="transformers"` 之后，vLLM 会自己推断该加载文本类还是多模态类。

Serving：

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
--model_impl transformers \
```

OpenAI 客户端（原文用 `localhost:8000/v1`，key 写 `"EMPTY"`）：

```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
chat_response = client.chat.completions.create(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                },
            },
        ],
    }],
)
print("Chat response:", chat_response)
```

也可以不走 HTTP，直接 `LLM`。多模态要自己用 `AutoProcessor` 套 chat template，再把 PIL 图塞进 `multi_modal_data`：

```python
from vllm import LLM, SamplingParams
from PIL import Image
import requests
from transformers import AutoProcessor

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
hf_processor = AutoProcessor.from_pretrained(model_id) # required to dynamically update the chat template

messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": "dummy_image.jpg"},
          {"type": "text", "text": "What is the content of this image?"},
        ],
    },
]
prompt = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

# initialize the vlm using the `model_impl="transformers"`
vlm = LLM(
    model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    model_impl="transformers",
)

outputs = vlm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    },
    sampling_params=SamplingParams(max_tokens=100)
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

原文贴了一段样例输出：两只猫在粉色沙发上睡觉。那是当时那次生成，不是保证。

## Transformers 和 vLLM：同一句 prompt

以 `meta-llama/Llama-3.2-1B` 做对照。

**Transformers `pipeline`：** 原型友好，不是为高并发、低时延准备的。

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
result = pipe("The future of AI is")

print(result[0]["generated_text"])
```

**vLLM `LLM`：** PagedAttention、dynamic batching。同一句 prompt：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("The future of AI is", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

原文的定性：vLLM 在负载下更快、更省资源；举例「每秒几千请求、GPU 显存更低」——没有表，当宣传口径，别当 SLA。

## 部署：OpenAI 兼容

vLLM 提供 OpenAI 兼容 API，可当本地替代。拉起：

```bash
vllm serve meta-llama/Llama-3.2-1B
```

curl：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

Python OpenAI 客户端：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
completion = client.completions.create(
    model="meta-llama/Llama-3.2-1B",
    prompt="San Francisco is a",
    max_tokens=7,
    temperature=0
)
print("Completion result:", completion.choices[0].text)
```

兼容的意思：推理收在自己机器上，还走同一套客户端。

## 为什么需要这层 backend

Transformers 为**加新模型**优化（[add a new model](https://huggingface.co/docs/transformers/en/add_new_model)）。往 vLLM 加一只原生实现要更绕（[contributing models](https://docs.vllm.ai/en/latest/contributing/model/index.html)）：调度、paged KV、CUDA graph 都要对齐。

理想世界：模型一进 Transformers，vLLM 就能伺候。这层 backend 往那个理想挪了一步。

兼容清单：[Custom models](https://docs.vllm.ai/en/latest/models/supported_models.html#custom-models)。他们按这份清单改过 `modeling_gpt2.py`，样板 PR：[huggingface/transformers#36934](https://github.com/huggingface/transformers/pull/36934)。

已经在 Transformers 里、也兼容这层集成的模型：

```python
llm = LLM(model="new-transformers-model", model_impl="transformers")
```

> **原文注：** `model_impl` **不是**硬性必填。vLLM 找不到原生实现时，会自己切到 Transformers 实现。

Hub 上的自定义模型还要 `trust_remote_code=True`：

```python
llm = LLM(model="custom-hub-model", model_impl="transformers", trust_remote_code=True)
```

桥的两端：Transformers 的即插即用，vLLM 的推理路径。原型仍在 Transformers 里写；部署走 vLLM。

## 案例：Kyutai Helium

写这篇时，[Kyutai Helium](https://huggingface.co/docs/transformers/en/model_doc/helium) **还没有** vLLM 原生实现。backend 的用处就是这一类：

```bash
vllm serve kyutai/helium-1-preview-2b --model-impl transformers
```

注意：Helium 这条 CLI 用的是 **`--model-impl`**（连字符）；上面 VLM 更新用的是 **`--model_impl`**（下划线）。原文两处拼法都在，以你那版 CLI 为准。

OpenAI 客户端：

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(model="kyutai/helium-1-preview-2b", prompt="What is AI?")
print("Completion result:", completion)
```

vLLM 负责吞吐路径，Transformers backend 负责把 `kyutai/helium-1-preview-2b` 加载进来。相对纯 Transformers 推理，原文仍只给定性：更低时延、更好的资源利用率——没有对照表。

收束也是同一句话：Transformers 的模型面，加上 vLLM 的推理优化。新架构、自定义 Hub 模型、后来的多模态，都是这条桥上的交通。原生路径一旦落地，就不必再写 `model_impl`。
