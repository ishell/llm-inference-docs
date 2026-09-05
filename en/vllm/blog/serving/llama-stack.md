---
source: https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm
lang: en
fetched: 2026-09-04
---

# Introducing vLLM Inference Provider in Llama Stack

Chinese: [zh/vllm/blog/serving/llama-stack.md](../../../../zh/vllm/blog/serving/llama-stack.md)

2025-01-27. **Yuan Tang (Red Hat) and Ashwin Bharambe (Meta)**. Demo: Llama-3.2-1B **CPU** container. Stack: [meta-llama/llama-stack](https://github.com/meta-llama/llama-stack). Docs then: [remote vLLM distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html), [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-completions-api-with-vllm). Container cousin: [docker-model-runner.md](docker-model-runner.md). Study note. Tutorial is **2025-01** `llama stack build` YAML — APIs drift. The point: one app API across the lifecycle; the engine underneath is a **swappable Provider**, not a second engine.

Local figures (copyright remains with the original site; study copies):

![llama stack](../../../../assets/vllm/blog/serving/llama-stack/01-llama-stack.png)

## What is Llama Stack?

Llama Stack defines core building blocks for generative-AI apps as interoperable APIs, each with Service Providers. Pre-packaged **distributions** run locally, on mobile/desktop, on-prem, or in public cloud — **same APIs**, same developer experience.

Models in the pitch: Llama 3.3 through specialized ones like Llama Guard. Safety, agents, vectors stay **other** Stack providers. Each implementation of an API is a **Provider**; users swap providers in config. vLLM is the high-performance backing for the **inference** API.

## vLLM Inference Provider

Two:

1. [Remote](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html) — `remote::vllm` against vLLM’s OpenAI-compatible `/v1`.
2. [Inline](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline/inference/vllm) — runs in the same process as the Stack server.

This tutorial demonstrates the **remote** path.

## Tutorial

### Prerequisites

Linux; [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) if you download via CLI; Podman or Docker (`CONTAINER_BINARY` for `llama stack` CLI); [Kind](https://kind.sigs.k8s.io/) for Kubernetes; [Conda](https://github.com/conda/conda) for the Python env.

Paths below are the **original post’s** tutorial scratch dir (`/tmp/test-vllm-llama-stack`), not a machine-local home path.

## Get Started via Containers

### Start vLLM Server

Need Hugging Face access to [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), then a token.

```bash
mkdir /tmp/test-vllm-llama-stack
huggingface-cli login --token <YOUR-HF-TOKEN>
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct
```

Build the vLLM **CPU** image from source (demo). Other hardware images: [installation](https://docs.vllm.ai/en/latest/getting_started/installation.html).

```bash
git clone git@github.com:vllm-project/vllm.git /tmp/test-vllm-llama-stack
cd /tmp/test-vllm-llama-stack/vllm
podman build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
```

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

Smoke test:

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

### Start Llama Stack Server

```bash
git clone git@github.com:meta-llama/llama-stack.git /tmp/test-vllm-llama-stack/llama-stack
cd /tmp/test-vllm-llama-stack/llama-stack
conda create -n stack python=3.10
conda activate stack
pip install .
```

Build config — inference is `remote::vllm`; safety / agents / vectors / etc. stay inline Stack providers:

```
cat > /tmp/test-vllm-llama-stack/vllm-llama-stack-build.yaml << "EOF"
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
EOF

export CONTAINER_BINARY=podman
LLAMA_STACK_DIR=. PYTHONPATH=. python -m llama_stack.cli.llama stack build \
  --config /tmp/test-vllm-llama-stack/vllm-llama-stack-build.yaml \
  --image-name distribution-myenv
```

Edit generated `vllm-run.yaml` → `/tmp/test-vllm-llama-stack/vllm-llama-stack-run.yaml`, `models` field:

```
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

Or `podman run` with the same env, mounting the YAML and source, entrypoint `python -m llama_stack.distribution.server.server --yaml-config /app/config.yaml`, image `localhost/distribution-myenv:dev`.

Client:

```bash
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

Python:

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

Sample `models.list()` on the page: `provider_id='vllm'`, identifier `meta-llama/Llama-3.2-1B-Instruct`. The haiku in the post is a demo completion, not a quality claim.

## Deployment on Kubernetes

Kind cluster for the demo:

```bash
kind create cluster --image kindest/node:v1.32.0 --name llama-stack-test
```

vLLM as Pod + Service (replace `<YOUR-HF-TOKEN>`; the post’s Secret `data.token` field is shown as that placeholder):

PVC `vllm-models` 50Gi; Secret `hf-token-secret`; Pod `vllm-server` image `localhost/vllm-cpu-env:latest`, downloads the model then:

```
python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL --port 8000
```

Service `vllm-server` port 8000. Logs (model download can take minutes): Uvicorn on `http://0.0.0.0:8000`.

Stack run YAML for K8s — the only inference URL that matters:

```
providers:
  inference:
  - provider_id: vllm
    provider_type: remote::vllm
    config:
      url: http://vllm-server.default.svc.cluster.local:8000/v1
      max_tokens: 4096
      api_token: fake
```

Build an image that clones Stack source and `ADD`s that YAML. Deploy Pod `llama-stack-pod` + Service `llama-stack-service` ClusterIP **5000**. The post’s log check command is `kubectl logs vllm-server` (same name as the vLLM pod); expected Stack lines: Uvicorn on port **5000**.

```bash
kubectl port-forward service/llama-stack-service 5000:5000
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

More providers: [Llama Stack docs](https://llama-stack.readthedocs.io).

## Acknowledgement

Red Hat AI Engineering: vLLM inference providers, bug fixes, design. Llama Stack team at Meta and the vLLM team: reviews and fixes.
