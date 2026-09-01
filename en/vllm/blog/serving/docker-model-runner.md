---
source: https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm
lang: en
fetched: 2026-09-01
---

# Docker Model Runner × vLLM: safetensors for throughput, GGUF still llama.cpp

Chinese: `../../zh/vllm/blog/serving/docker-model-runner.md`  
Then: x86_64 + NVIDIA.

```
docker model install-runner --backend vllm --gpu cuda
docker model run ai/smollm2-vllm "..."
```

HTTP is still localhost:12434 `/v1/chat/completions`; the request does not name the engine — GGUF→llama.cpp, safetensors→vLLM. Both as OCI. The claim is one Docker command from laptop prototype to throughput serve, not a new vLLM kernel. Startup was slower than llama.cpp; WSL2 / DGX Spark still incoming.
