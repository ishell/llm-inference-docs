---
source: https://vllm.ai/blog/2026-01-02-introducing-vllm-playground
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vLLM Playground：容器里点一下起 server，CLI 仍是同一套旋钮

英文对照：[en/vllm/blog/serving/playground.md](../../../../en/vllm/blog/serving/playground.md)  
原文：https://vllm.ai/blog/2026-01-02-introducing-vllm-playground  
2026-01-02。**micytao**（社区项目，不是 vLLM 核心仓）。GitHub：[micytao/vllm-playground](https://github.com/micytao/vllm-playground)。PyPI：[vllm-playground](https://pypi.org/project/vllm-playground/)。Apache-2.0。当时镜像钉 **v0.11.0**。**不**替代 `vllm serve`——是 **140+** 旋钮上的表单。GuideLLM 亲戚：[deeplearning-ai-course.md](deeplearning-ai-course.md)。韩国场提到：[korea-meetup-2026.md](korea-meetup-2026.md)。

本地图（原文版权仍归原站；学习对照用）：

![vllm playground newUI](../../../../assets/vllm/blog/serving/playground/01-vllm-playground-newUI.png)

![vllm playground structured outputs](../../../../assets/vllm/blog/serving/playground/02-vllm-playground-structured-outputs.png)

![guidellm](../../../../assets/vllm/blog/serving/playground/03-guidellm.png)

![vllm recipes 1](../../../../assets/vllm/blog/serving/playground/04-vllm-recipes-1.png)

## 为什么要它

CLI + 容器 + 一片旗林。Playground 的卖点：

- **Zero setup** — 容器自己拉，不必手装 vLLM
- **一键** 起停、换模型、改配置
- **跨平台** — macOS Apple Silicon、Linux CPU/GPU、Kubernetes / OpenShift
- **同一套 UI** 从本地到云

他们印的 roadmap：跟官方 vLLM 齐步，新功能点得开。下一步：**MCP** server、**RAG**、持续把新能力补进 UI。

## 开工

```bash
pip install vllm-playground
vllm-playground pull          # 可选，GPU 镜像约 10GB
vllm-playground
```

打开 http://localhost:7860，点 Start Server。编排器自己选镜像。

## 他们列的功能

**UI：** 暗色、聊天、图标工具栏（设置、system prompt、structured outputs、工具）、每条回复的 token 数和生成速度、可拖面板。

**Structured outputs**（四种）：

| 模式 | 他们举的用途 |
|---|---|
| Choice | 封闭集合（情感……） |
| Regex | 邮箱 / 电话 / 日期 |
| JSON Schema | API 形状的 JSON |
| Grammar（EBNF） | DSL / 形式语言 |

**Tool calling：** 起服务前在 Server Configuration 里打开。自动 parser：Llama 3.x、Mistral、Hermes、Qwen、Granite、InternLM。预置：Weather、Calculator、Search。自定义工具走 JSON Schema。并行 tool call。

**容器：** 生命周期、健康检查、日志；配置没变就复用容器。当时镜像：

- GPU：`vllm/vllm-openai:v0.11.0`
- CPU x86：`quay.io/rh_ee_micyang/vllm-cpu:v0.11.0`
- macOS ARM64：`quay.io/rh_ee_micyang/vllm-mac:v0.11.0`

**GuideLLM：** 成功率、时长、均值；tok/s 均值/中位数；P50/P75/P90/P95/P99；负载模式；JSON 导出。

**Recipes：** [vllm-project/recipes](https://github.com/vllm-project/recipes) — 17+ 类（DeepSeek、Qwen、Llama……），可搜，一键填 flag，带 GPU 指引。

**OpenShift / K8s：**

```bash
cd openshift/
./deploy.sh --gpu
./deploy.sh --cpu
```

动态建 pod、自动认 GPU/CPU、RBAC，UI 不变。

## 架构

```
Web UI（FastAPI：app.py + index.html + static/）
  ├─ 本地：container_manager.py → Podman → vLLM 容器
  └─ 云：kubernetes_container_manager.py → K8s API → vLLM pod
```

编译期换 manager，界面两边一样。

## macOS ARM

给 Apple Silicon 的 CPU 优化镜像，自动认平台，rootless Podman。`python run.py` 或 `vllm-playground`。

## CLI

```bash
vllm-playground                    # 默认
vllm-playground --port 8080
vllm-playground pull               # GPU 约 10GB
vllm-playground pull --cpu
vllm-playground pull --all
vllm-playground stop
vllm-playground status
```

这篇没有 serving TPS——是给旗标做的 UI，不是基准。
