---
source: https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Docker Model Runner × vLLM：safetensors 走高吞吐，GGUF 仍 llama.cpp

英文对照：`en/vllm/blog/serving/docker-model-runner.md`  
原文：https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm  
当时 x86_64 + NVIDIA。

```
docker model install-runner --backend vllm --gpu cuda
docker model run ai/smollm2-vllm "..."
```

HTTP 仍是 localhost:12434 `/v1/chat/completions`，请求里不写引擎名——按模型格式路由：GGUF→llama.cpp，safetensors→vLLM。OCI 推拉两种格式。卖点是同一套 Docker 命令从笔记本原型切到吞吐 serve，不是 vLLM 内核新机制。当时启动比 llama.cpp 慢；WSL2 / DGX Spark 还在路上。
