---
source: https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Docker Model Runner × vLLM：safetensors 走高吞吐，GGUF 仍 llama.cpp

英文对照：[en/vllm/blog/serving/docker-model-runner.md](../../../../en/vllm/blog/serving/docker-model-runner.md)  
原文：https://vllm.ai/blog/2025-11-19-docker-model-runner-vllm  
2025-11-19。**Docker Team**。仓库：[docker/model-runner](https://github.com/docker/model-runner)。vLLM 是 Docker Model Runner 里的一个推理后端——不是 vLLM 内核新机制。当时：**x86_64 + NVIDIA**。WSL2 / DGX Spark 写在下一步；Spark 正文见 [dgx-spark.md](dgx-spark.md)。页上只有 logo（未收录）。没有 TPS 表。

## 把 Model Runner 的能力往外扩

Model Runner 已经能用 Docker 跑 LLM。从第一天就打算接多种引擎，先接的是 **llama.cpp**。这篇补上 **vLLM** 和 **safetensors**，同一套 Docker 工作流可以从低端 NVIDIA 走到高端。

## 为什么是 vLLM？

开源高吞吐引擎，给生产级 LLM。他们列的三条：

- **Optimized performance**：PagedAttention——少占内存、多吃 GPU。
- **Scalable serving**：原生 batch 和流式输出。
- **Model flexibility**：GPT-OSS、Qwen3、Mistral、Llama 3 以及其它 **safetensors** 开源权重。

卖点：本地试验和生产向推理之间那条缝，**不必离开 Docker**。

## 怎么跑

装后端，再跑模型。请求里不用再写引擎名。

```bash
docker model install-runner --backend vllm --gpu cuda
```

```bash
docker model run ai/smollm2-vllm "Can you read me?"
```

页上的示例回复：`Sure, I am ready to read you.`

HTTP（OpenAI 兼容 chat）：

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

**原文写死的坑：** HTTP 和 CLI **都不出现 vLLM 这个名字**。Model Runner 按模型路由：GGUF → llama.cpp，safetensors → vLLM。

## 为什么要多种引擎？

在那之前：好跑（Model Runner + llama.cpp）**或者**冲吞吐（自己养一套 vLLM）。现在的说法：llama.cpp 在笔记本上原型；vLLM 上生产向吞吐；Docker 命令、CI/CD、环境还是同一套。

他们自称这是行业里第一份「一个便携容器工作流里切换多种推理引擎」。从笔记本到集群——页上的宣传，不是 bake-off。

## Safetensors（vLLM）对 GGUF（llama.cpp）

两种主流开源格式，都能当 **OCI 镜像** 推拉。

- **GGUF**：llama.cpp 的原生格式。便携、量化。商品硬件、带宽紧的时候。架构和权重打在一个文件里。
- **Safetensors**：vLLM 的原生格式；高吞吐、高端路径。

路由看你 pull 了什么，不看 `curl` 里有没有写引擎。

## Docker Hub 上当时的 vLLM 模型

safetensors。页上点名的早期名字：

- [ai/smollm2-vllm](https://hub.docker.com/r/ai/smollm2-vllm)
- [ai/qwen3-vllm](https://hub.docker.com/r/ai/qwen3-vllm)
- [ai/gemma3-vllm](https://hub.docker.com/r/ai/gemma3-vllm)
- [ai/gpt-oss-vllm](https://hub.docker.com/r/ai/gpt-oss-vllm)

## 当时能用的：x86_64 + NVIDIA

首发只覆盖 **x86_64 + NVIDIA GPU**。没有别的架构 / GPU 矩阵。

## 下一步

两块：平台覆盖，性能。

- **WSL2 / Docker Desktop**：经 WSL2 把 vLLM 后端带到 Windows，先从 NVIDIA Windows 机器起。Desktop 上的 inner loop 对齐 Linux。
- **DGX Spark**：写「不同硬件」；点名 NVIDIA DGX。这里没有 Spark flag 或数字。
- **性能**：vLLM **启动当时比 llama.cpp 慢**。他们想改进开发循环里的 “time-to-first-token”。页上**没有慢几秒**。

## 怎么参与

给 [docker/model-runner](https://github.com/docker/model-runner) 点星。Issue / fork / PR。把话传开。社区邀请，不是推理成绩。
