---
source: https://vllm.ai/blog/2026-01-02-introducing-vllm-playground
lang: en
fetched: 2026-09-01
---

# vLLM Playground: click-start a containerized server; CLI knobs unchanged

Chinese: [zh/vllm/blog/serving/playground.md](../../../../zh/vllm/blog/serving/playground.md)  
`pip install vllm-playground` → localhost:7860. Apache-2.0. Images then pinned v0.11.0.

Local Podman, cloud Kubernetes API, same UI. macOS ARM CPU, Linux GPU/CPU, OpenShift. Structured outputs: choice / regex / JSON schema / EBNF. Tool calling auto-picks Llama/Mistral/Hermes/Qwen… parsers. GuideLLM percentiles in-app. Recipes one-click fill flags. Does not replace `vllm serve` — a form over 140+ knobs. Community project, not the vLLM core repo.

Local figures (copyright remains with the original site; study copies):

![vllm playground newUI](../../../../assets/vllm/blog/serving/playground/01-vllm-playground-newUI.png)

![vllm playground structured outputs](../../../../assets/vllm/blog/serving/playground/02-vllm-playground-structured-outputs.png)

![guidellm](../../../../assets/vllm/blog/serving/playground/03-guidellm.png)

![vllm recipes 1](../../../../assets/vllm/blog/serving/playground/04-vllm-recipes-1.png)
