---
source: https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm
lang: en
fetched: 2026-09-04
---

# Introducing vLLM Inference Provider in Llama Stack

Chinese: [zh/vllm/blog/serving/llama-stack.md](../../../../zh/vllm/blog/serving/llama-stack.md)

2025-01-27. **Yuan Tang (Red Hat) and Ashwin Bharambe (Meta)**. Repos: [meta-llama/llama-stack](https://github.com/meta-llama/llama-stack), docs: [llama-stack.readthedocs.io](https://llama-stack.readthedocs.io). Demo: **Llama-3.2-1B-Instruct** on a **CPU** container. Sibling “vLLM as one backend”: [docker-model-runner.md](docker-model-runner.md). App-lifecycle packaging cousin: [production-stack.md](production-stack.md). Study note. **Inference is a swappable Provider, not a second engine.** Tutorial is 2025-01 `llama stack build` YAML — **APIs drift**. No TPS table.

Two providers: [`remote::vllm`](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html) (OpenAI-compatible `/v1`) and [inline](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline/inference/vllm) (same process as Stack). Safety, agents, vectors stay other Stack providers. This post demos **remote**. K8s: vLLM Service DNS `http://vllm-server.default.svc.cluster.local:8000/v1`; Stack only fills the URL.

Local figures (copyright remains with the original site; study copies):

![llama stack](../../../../assets/vllm/blog/serving/llama-stack/01-llama-stack.png)

**Figure.** Llama Stack: interoperable APIs, each implementation a Provider. vLLM backs **inference**.

## What is Llama Stack?

Llama Stack standardizes the building blocks for generative apps: interoperable APIs plus Service Providers that implement them. Pre-packaged “distributions” for local / mobile / desktop → on-prem / cloud, **same APIs** at every step. Models they name then: Llama 3.3, Llama Guard for safety, plus others. Swap providers in config. vLLM is the high-performance backing for the inference API — not a rewrite of Stack's agent/safety/vector layers.

## vLLM inference provider

1. **Remote** — Stack HTTP-calls vLLM's OpenAI-compatible server.
2. **Inline** — vLLM runs in-process with the Stack server.

Remote is what the rest of the page walks.

## Tutorial

Prerequisites they list: Linux; Hugging Face CLI; Podman or Docker (`CONTAINER_BINARY`); Kind for K8s; Conda.

Paths they use: `/tmp/test-vllm-llama-stack`. Model needs [HF access](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) + token.

### Start vLLM (CPU image)

```bash
mkdir /tmp/test-vllm-llama-stack
huggingface-cli login --token <YOUR-HF-TOKEN>
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct
```

Build the CPU image from source (`Dockerfile.cpu`). Other hardware images: [vLLM install docs](https://docs.vllm.ai/en/latest/getting_started/installation.html).

```bash
git clone git@github.com:vllm-project/vllm.git /tmp/test-vllm-llama-stack
cd /tmp/test-vllm-llama-stack/vllm
podman build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
```

Run (entrypoint is the **then** OpenAI API module):

```bash
podman run -it --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct:/app/model \
   --entrypoint='["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/app/model", "--served-model-name", "meta-llama/Llama-3.2-1B-Instruct", "--port", "8000"]' \
    vllm-cpu-env
```

**Caveat on the sample:** `--device /dev/kfd` is an AMD ROCm device node on a **CPU** Dockerfile demo. Historical copy-paste; not a ROCm bake-off. Entrypoint `vllm.entrypoints.openai.api_server` is the 2025-01 module path.

Smoke:

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

### Start Llama Stack

Clone, Conda **python=3.10**, `pip install .` from the Stack tree.

Build spec they print (`image_type: container`). **This YAML is the 2025-01 provider map** — names drift; read current docs before copying:

```yaml
name: vllm
distribution_spec:
  description: Like local, but use vLLM for running LLM inference
  providers:
    inference: remote::vllm
    safety: inline::llama-guard
    agents: inline::meta-reference
    vector_io: inline::faiss
    datasetio: inline::localfs
    scoring: inline::basic
    eval: inline::meta-reference
    post_training: inline::torchtune
    telemetry: inline::meta-reference
image_type: container
```

```bash
export CONTAINER_BINARY=podman
LLAMA_STACK_DIR=. PYTHONPATH=. python -m llama_stack.cli.llama stack build \
  --config /tmp/test-vllm-llama-stack/vllm-llama-stack-build.yaml \
  --image-name distribution-myenv
```

Edit generated `vllm-run.yaml` → `vllm-llama-stack-run.yaml`. Models block they want:

```yaml
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: vllm
  provider_model_id: null
```

Run:

```bash
export INFERENCE_ADDR=host.containers.internal
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-1B-Instruct
export LLAMA_STACK_PORT=5000

LLAMA_STACK_DIR=. PYTHONPATH=. python -m llama_stack.cli.llama stack run \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://$INFERENCE_ADDR:$INFERENCE_PORT/v1 \
  --env VLLM_MAX_TOKENS=8192 \
  --env VLLM_API_TOKEN=fake \
  --env LLAMA_STACK_PORT=$LLAMA_STACK_PORT \
  /tmp/test-vllm-llama-stack/vllm-llama-stack-run.yaml
```

Alternate `podman run` of `localhost/distribution-myenv:dev`, same env, entrypoint `python -m llama_stack.distribution.server.server --yaml-config /app/config.yaml`.

CLI check:

```bash
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

Page sample: `ChatCompletionResponse` with `role='assistant'`, `stop_reason='end_of_turn'`, empty `tool_calls`. Text content is a canned assistant reply — not a benchmark.

Python they print:

```python
import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")
models = client.models.list()
print(models)

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ]
)
print(response.completion_message.content)
```

Listed model: `identifier='meta-llama/Llama-3.2-1B-Instruct'`, `provider_id='vllm'`, `api_model_type='llm'`.

## Kubernetes

Kind: `kind create cluster --image kindest/node:v1.32.0 --name llama-stack-test`.

vLLM as Pod + Service: PVC `vllm-models` **50Gi**; Secret `hf-token-secret` with placeholder `"<YOUR-HF-TOKEN>"` (page prints it under `data.token`, not a real base64 blob); container image `localhost/vllm-cpu-env:latest`. Inside the container they `huggingface-cli download` then:

```text
python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL --port 8000
```

Service name **`vllm-server`**, port **8000**, `type: NodePort`. Stack does not scrape Pod IPs; it uses DNS.

Ready log they wait for: `Uvicorn running on http://0.0.0.0:8000`.

Run YAML inference provider for in-cluster Stack:

```yaml
providers:
  inference:
  - provider_id: vllm
    provider_type: remote::vllm
    config:
      url: http://vllm-server.default.svc.cluster.local:8000/v1
      max_tokens: 4096
      api_token: fake
```

**That URL is the whole integration.** Swap the engine by swapping the Service behind `/v1`.

They bake a `Containerfile` `FROM distribution-myenv:dev`, clone llama-stack source, `ADD` the k8s run YAML, tag `llama-stack-run-k8s`. Pod `llama-stack-pod` + Service `llama-stack-service` ClusterIP **5000**. PVC `llama-pvc` **1Gi** on `/root/.llama`.

**Caveat they print:** the “check Llama Stack logs” snippet is `$ kubectl logs vllm-server` — that is the **vLLM** Pod name. Treat as a copy-paste slip on the page; the Stack Pod is `llama-stack-pod`. Ready line they show for Stack: Uvicorn on `http://['::', '0.0.0.0']:5000`.

Forward and hit the **Stack** API, not vLLM directly:

```bash
kubectl port-forward service/llama-stack-service 5000:5000
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

More providers: [official docs](https://llama-stack.readthedocs.io).

## Acknowledgement

Red Hat AI Engineering (vLLM providers, fixes, design). Meta Llama Stack team and vLLM team (reviews).
