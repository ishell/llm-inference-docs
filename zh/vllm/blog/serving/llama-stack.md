---
source: https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Llama Stack × vLLM：inference 是可换的 Provider，不是另一套引擎

英文对照：[en/vllm/blog/serving/llama-stack.md](../../../../en/vllm/blog/serving/llama-stack.md)  
原文：https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm  
2025-01-27。署名 **Yuan Tang (Red Hat) and Ashwin Bharambe (Meta)**。仓库：[meta-llama/llama-stack](https://github.com/meta-llama/llama-stack)，文档：[llama-stack.readthedocs.io](https://llama-stack.readthedocs.io)。演示：**Llama-3.2-1B-Instruct** 跑在 **CPU** 容器。同类「vLLM 当一个后端」：[docker-model-runner.md](docker-model-runner.md)。应用生命周期打包亲戚：[production-stack.md](production-stack.md)。**inference 是可换的 Provider，不是另一套引擎。** 教程偏 2025-01 的 `llama stack build` YAML——**API 会漂**。没有 TPS 表。

两种：[`remote::vllm`](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html)（打 OpenAI-compatible `/v1`）和 [inline](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline/inference/vllm)（跟 Stack 同进程）。安全、agent、vector 仍是 Stack 自己的 provider。这篇演示 **remote**。K8s：vLLM Service DNS `http://vllm-server.default.svc.cluster.local:8000/v1`，Stack 只填 URL。

本地图（原文版权仍归原站；学习对照用）：

![llama stack](../../../../assets/vllm/blog/serving/llama-stack/01-llama-stack.png)

**Figure.** Llama Stack：可互换 API，每种实现叫 Provider。vLLM 托的是 **inference**。

## What is Llama Stack?

Llama Stack 把生成式应用的积木标准化：一套可互操作 API，外加实现它们的 Service Provider。预打包 “distributions”，从本地 / 手机 / 桌面迁到机房 / 公有云，**每一步同一套 API**。当时点名的模型：Llama 3.3、安全用的 Llama Guard，还有别的。配置里换 provider。vLLM 是 inference API 的高性能托底——不是把 Stack 的 agent/safety/vector 层重写一遍。

## vLLM inference provider

1. **Remote** —— Stack 用 HTTP 打 vLLM 的 OpenAI-compatible 服务。
2. **Inline** —— vLLM 和 Stack server 同进程。

后面整页走的是 remote。

## Tutorial

他们列的前置：Linux；Hugging Face CLI；Podman 或 Docker（`CONTAINER_BINARY`）；K8s 用 Kind；Conda。

路径：`/tmp/test-vllm-llama-stack`。模型要 [HF 权限](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) + token。

### 先起 vLLM（CPU 镜像）

```bash
mkdir /tmp/test-vllm-llama-stack
huggingface-cli login --token <YOUR-HF-TOKEN>
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/test-vllm-llama-stack/.cache/huggingface/hub/models/Llama-3.2-1B-Instruct
```

从源码打 CPU 镜像（`Dockerfile.cpu`）。别的硬件镜像见 [vLLM 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

```bash
git clone git@github.com:vllm-project/vllm.git /tmp/test-vllm-llama-stack
cd /tmp/test-vllm-llama-stack/vllm
podman build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .
```

跑（entrypoint 是 **当时** 的 OpenAI API 模块）：

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

**原文写死的坑：** `--device /dev/kfd` 是 AMD ROCm 设备节点，却套在 **CPU** Dockerfile 演示上。历史粘贴；不是 ROCm bake-off。entrypoint `vllm.entrypoints.openai.api_server` 是 2025-01 的模块路径。

冒烟：

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

### 再起 Llama Stack

clone，Conda **python=3.10**，在 Stack 树里 `pip install .`。

他们印的 build spec（`image_type: container`）。**这份 YAML 是 2025-01 的 provider 表**——名字会漂；抄之前看现行文档：

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

生成的 `vllm-run.yaml` 改成 `vllm-llama-stack-run.yaml`。他们要的 models 块：

```yaml
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: vllm
  provider_model_id: null
```

跑：

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

也可以 `podman run` `localhost/distribution-myenv:dev`，同一套环境变量，entrypoint `python -m llama_stack.distribution.server.server --yaml-config /app/config.yaml`。

CLI 检查：

```bash
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

页上样例：`ChatCompletionResponse`，`role='assistant'`，`stop_reason='end_of_turn'`，`tool_calls` 空。那段助手回复是演示文本——不是基准。

他们印的 Python：

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

列出来的模型：`identifier='meta-llama/Llama-3.2-1B-Instruct'`，`provider_id='vllm'`，`api_model_type='llm'`。

## Kubernetes

Kind：`kind create cluster --image kindest/node:v1.32.0 --name llama-stack-test`。

vLLM 做成 Pod + Service：PVC `vllm-models` **50Gi**；Secret `hf-token-secret`，占位 `"<YOUR-HF-TOKEN>"`（页上写在 `data.token` 下，不是真的 base64）；镜像 `localhost/vllm-cpu-env:latest`。容器里先 `huggingface-cli download`，再：

```text
python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL --port 8000
```

Service 名 **`vllm-server`**，端口 **8000**，`type: NodePort`。Stack 不扫 Pod IP，用 DNS。

他们等的就绪日志：`Uvicorn running on http://0.0.0.0:8000`。

集群内 Stack 的 run YAML inference provider：

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

**这个 URL 就是全部集成。** 换引擎 = 换 `/v1` 后面那条 Service。

再打一个 `Containerfile`：`FROM distribution-myenv:dev`，clone llama-stack 源码，`ADD` k8s 那份 run YAML，打成 `llama-stack-run-k8s`。Pod `llama-stack-pod` + Service `llama-stack-service` ClusterIP **5000**。PVC `llama-pvc` **1Gi** 挂 `/root/.llama`。

**原文写死的坑：** 「看 Llama Stack 日志」那条是 `$ kubectl logs vllm-server`——那是 **vLLM** Pod 名。当页上的粘贴笔误；Stack Pod 叫 `llama-stack-pod`。他们给 Stack 看的就绪行：Uvicorn 在 `http://['::', '0.0.0.0']:5000`。

转发打的是 **Stack** API，不是直接打 vLLM：

```bash
kubectl port-forward service/llama-stack-service 5000:5000
llama-stack-client --endpoint http://localhost:5000 inference chat-completion \
  --message "hello, what model are you?"
```

更多 provider：[官方文档](https://llama-stack.readthedocs.io)。

## Acknowledgement

Red Hat AI Engineering（vLLM provider、修复、设计）。Meta Llama Stack 团队和 vLLM 团队（review）。
