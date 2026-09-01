---
source: https://vllm.ai/blog/2025-04-11-transformers-backend
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Transformers modeling backend：新架构先跑起来

英文对照：`en/vllm/blog/architecture/transformers-backend.md`  
原文：https://vllm.ai/blog/2025-04-11-transformers-backend  
2025-04；2025-07 起支持视觉语言模型。图在原网页。

vLLM 原生 `modeling_*.py` 要跟调度、paged KV、CUDA graph 对齐，新模型进仓库慢。`model_impl="transformers"` 直接跑 Hugging Face 实现，上面仍用 PagedAttention 和 continuous batching。Hub 上还没有原生实现的架构（当时 Kyutai Helium）可以：

```
vllm serve kyutai/helium-1-preview-2b --model-impl transformers
```

自定义 Hub 模型加 `trust_remote_code=True`。原生已支持时不必写这个参数——找不到原生实现会自己切。多模态：

```
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --model_impl transformers
```

兼容清单在 Transformers 文档；GPT-2 当过样板 PR。这是 **覆盖面**，不是性能默认：能进原生路径就走原生。
