---
source: https://vllm.ai/blog/2025-04-11-transformers-backend
lang: en
fetched: 2026-09-04
---

# Transformers modeling backend integration in vLLM

Chinese: [zh/vllm/blog/architecture/transformers-backend.md](../../../../zh/vllm/blog/architecture/transformers-backend.md)

2025-04-11. **The Hugging Face Team**. Vision-language models from **2025-07-21**. No figures on the original. This is **coverage**, not the performance default: if a native `modeling_*.py` exists, use it.

Fits: Hub architectures without a native vLLM port, served under PagedAttention and continuous batching. Does not fit: treating `model_impl="transformers"` as a throughput switch.

[Transformers](https://huggingface.co/docs/transformers/main/en/index) is the model ecosystem: research, fine-tuning, one interface. [vLLM](https://docs.vllm.ai/en/latest/) is the serving layer: pull from the Hub, optimize for throughput and latency. The modeling backend sits Transformers implementations under vLLM — if the architecture already lives in Transformers, run it with vLLM’s scheduler and KV.

## Updates: Vision Language Models (21 July 2025)

The post shipped 11 April 2025. This section was added later. With `model_impl="transformers"`, vLLM infers the right class for text-only vs multimodal.

Serving:

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
--model_impl transformers \
```

OpenAI client (page uses `localhost:8000/v1`, key `"EMPTY"`):

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

Or skip HTTP and use `LLM` directly. Multimodal still needs `AutoProcessor` for the chat template, then a PIL image in `multi_modal_data`:

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

The page pastes a sample about two cats asleep on a pink couch. That is one generation, not a guarantee.

## Transformers and vLLM: the same prompt

`meta-llama/Llama-3.2-1B` as the foil.

**Transformers `pipeline`:** fine for prototypes, not built for high volume or low latency.

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
result = pipe("The future of AI is")

print(result[0]["generated_text"])
```

**vLLM `LLM`:** PagedAttention, dynamic batching. Same prompt:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("The future of AI is", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

Qualitative claim on the page: vLLM is faster and more resource-efficient under load; “thousands of requests per second” and lower GPU memory. No table — marketing contour, not an SLA.

## Deployment: OpenAI compatibility

vLLM exposes an OpenAI-compatible API as a local stand-in. Launch:

```bash
vllm serve meta-llama/Llama-3.2-1B
```

curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B", "prompt": "San Francisco is a", "max_tokens": 7, "temperature": 0}'
```

Python OpenAI client:

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

Compatibility means: keep inference on your machines, keep the same client.

## Why the Transformers modeling backend

Transformers is optimized for **adding models** ([add a new model](https://huggingface.co/docs/transformers/en/add_new_model)). A native vLLM port is more involved ([contributing models](https://docs.vllm.ai/en/latest/contributing/model/index.html)): scheduler, paged KV, CUDA graphs all have to line up.

Ideal world: the day a model lands in Transformers, vLLM can serve it. This backend steps toward that.

Compatibility checklist: [Custom models](https://docs.vllm.ai/en/latest/models/supported_models.html#custom-models). They followed it for `modeling_gpt2.py`. Template PR: [huggingface/transformers#36934](https://github.com/huggingface/transformers/pull/36934).

For a model already in Transformers and compatible with the integration:

```python
llm = LLM(model="new-transformers-model", model_impl="transformers")
```

> **Note:** `model_impl` is **not** strictly required. vLLM switches to the Transformers implementation on its own if the model is not natively supported.

Custom Hub models also need `trust_remote_code=True`:

```python
llm = LLM(model="custom-hub-model", model_impl="transformers", trust_remote_code=True)
```

The bridge: Transformers’ plug-and-play, vLLM’s serving path. Prototype in Transformers; deploy with vLLM.

## Case study: Kyutai Helium

When this post ran, [Kyutai Helium](https://huggingface.co/docs/transformers/en/model_doc/helium) was **not yet** natively supported in vLLM. That is the backend’s job:

```bash
vllm serve kyutai/helium-1-preview-2b --model-impl transformers
```

Caveat: the Helium CLI uses **`--model-impl`** (hyphen); the VLM update above uses **`--model_impl`** (underscore). Both spellings appear on the page; trust your version’s CLI.

OpenAI client:

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

vLLM owns the serving path; the Transformers backend loads `kyutai/helium-1-preview-2b`. Versus native Transformers inference the page only claims lower latency and better resource use — no comparison table.

Closing line of the post: Transformers’ model surface plus vLLM’s inference optimizations. New arches, custom Hub models, later multimodal — traffic on that bridge. Once a native path exists, you do not need `model_impl`.
