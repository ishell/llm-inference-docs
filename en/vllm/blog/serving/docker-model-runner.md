---
source: https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm
lang: en
fetched: 2026-09-04
---

# Docker Model Runner Integrates vLLM for High-Throughput Inferencing

Chinese: [zh/vllm/blog/serving/docker-model-runner.md](../../../../zh/vllm/blog/serving/docker-model-runner.md)

2025-11-19. **Docker Team**. Repo: [docker/model-runner](https://github.com/docker/model-runner). vLLM as one inference backend inside Docker Model Runner — not a new vLLM kernel. Then: **x86_64 + NVIDIA**. WSL2 / DGX Spark named as incoming; Spark itself: [dgx-spark.md](dgx-spark.md). Logo-only artwork on the page (not copied). No TPS table.

## Expanding Docker Model Runner's capabilities

Model Runner already ran LLMs with Docker. Multiple engines from day one, starting with **llama.cpp**. This post adds **vLLM** plus **safetensors**, so the same Docker workflow can move from low-end to high-end NVIDIA hardware.

## Why vLLM?

High-throughput open-source engine for production LLMs. What they list:

- **Optimized performance**: PagedAttention — less memory overhead, more GPU utilization.
- **Scalable serving**: batch requests and streaming outputs.
- **Model flexibility**: GPT-OSS, Qwen3, Mistral, Llama 3, and other **safetensors** open-weight models.

The claim: bridge fast local experimentation and production-style inference **without leaving Docker**.

## How vLLM works

Install the backend, then run a model. No extra engine-specific setup in the request.

```bash
docker model install-runner --backend vllm --gpu cuda
```

```bash
docker model run ai/smollm2-vllm "Can you read me?"
```

Example reply on the page: `Sure, I am ready to read you.`

HTTP (OpenAI-compatible chat):

```bash
curl --location 'http://localhost:12434/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "ai/smollm2-vllm",
  "messages": [
    {
      "role": "user",
      "content": "Can you read me?"
    }
  ]
}'
```

**Caveat they print:** the HTTP request and the CLI command **do not name vLLM**. Model Runner routes by model: GGUF → llama.cpp, safetensors → vLLM.

## Why multiple inference engines?

Until then: easy (Model Runner + llama.cpp) **or** max throughput (vLLM as its own stack). The pitch: prototype locally with llama.cpp; scale to production with vLLM; same Docker commands, CI/CD, and environments.

They call the unified interface a first: switch engines inside one portable, containerized workflow. From laptops to clusters — marketing claim on the page, not a bake-off.

## Safetensors (vLLM) vs GGUF (llama.cpp)

Two dominant open-source formats, both push/pull as **OCI images**.

- **GGUF**: native for llama.cpp. Portability and quantization. Commodity hardware, limited memory bandwidth. Architecture + weights in one file.
- **Safetensors**: native for vLLM; the high-throughput / high-end path.

Routing is by what you pull, not by a flag in `curl`.

## vLLM-compatible models on Docker Hub

Safetensors. Early Hub names on the page:

- [ai/smollm2-vllm](https://hub.docker.com/r/ai/smollm2-vllm)
- [ai/qwen3-vllm](https://hub.docker.com/r/ai/qwen3-vllm)
- [ai/gemma3-vllm](https://hub.docker.com/r/ai/gemma3-vllm)
- [ai/gpt-oss-vllm](https://hub.docker.com/r/ai/gpt-oss-vllm)

## Available now: x86_64 with NVIDIA

Initial release: **x86_64 + NVIDIA GPUs** only. No other arch/GPU matrix on the page.

## What's next?

Two roadmap buckets: platform access, performance.

- **WSL2 / Docker Desktop**: vLLM backend on Windows via WSL2, starting with NVIDIA Windows machines. Inner-loop on Desktop matching Linux.
- **DGX Spark compatibility**: “different kinds of hardware”; NVIDIA DGX systems named. No Spark flags or numbers here.
- **Performance**: vLLM **startup is currently slower than llama.cpp**. They want to improve “time-to-first-token” for rapid development cycles. **No delta in seconds** on the page.

## How you can get involved

Star [docker/model-runner](https://github.com/docker/model-runner). Issues / forks / PRs. Spread the word. Community ask, not an inference result.
