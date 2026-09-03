---
source: https://vllm.ai/blog/2025-04-11-transformers-backend
lang: en
fetched: 2026-09-01
---

# Transformers modeling backend: new arches without a native port

Chinese: [zh/vllm/blog/architecture/transformers-backend.md](../../../../zh/vllm/blog/architecture/transformers-backend.md)  
April 2025; VLMs from July 2025.

Native `modeling_*.py` must match the scheduler, paged KV, CUDA graphs — slow to land. `model_impl="transformers"` runs the Hugging Face module under PagedAttention and continuous batching. Architectures without a native port (Kyutai Helium then):

```
vllm serve kyutai/helium-1-preview-2b --model-impl transformers
```

Custom Hub models: `trust_remote_code=True`. Skip the flag when a native impl exists — vLLM falls back on its own. Multimodal:

```
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --model_impl transformers
```

Compatibility checklist lives in Transformers docs; GPT-2 was the template PR. This is **coverage**, not the performance default: use native when it exists.
