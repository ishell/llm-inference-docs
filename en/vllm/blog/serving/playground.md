---
source: https://vllm.ai/blog/2026-01-02-introducing-vllm-playground
lang: en
fetched: 2026-09-04
---

# Introducing vLLM Playground: A Modern Web Interface for Managing and Interacting with vLLM Servers

Chinese: [zh/vllm/blog/serving/playground.md](../../../../zh/vllm/blog/serving/playground.md)

2026-01-02. **micytao** (community, not the core vLLM repo). GitHub: [micytao/vllm-playground](https://github.com/micytao/vllm-playground). PyPI: [vllm-playground](https://pypi.org/project/vllm-playground/). Apache-2.0. Images then pinned **v0.11.0**. Does **not** replace `vllm serve` — a form over **140+** knobs. GuideLLM cousin: [deeplearning-ai-course.md](deeplearning-ai-course.md). Korea mention: [korea-meetup-2026.md](korea-meetup-2026.md).

Local figures (copyright remains with the original site; study copies):

![vllm playground newUI](../../../../assets/vllm/blog/serving/playground/01-vllm-playground-newUI.png)

![vllm playground structured outputs](../../../../assets/vllm/blog/serving/playground/02-vllm-playground-structured-outputs.png)

![guidellm](../../../../assets/vllm/blog/serving/playground/03-guidellm.png)

![vllm recipes 1](../../../../assets/vllm/blog/serving/playground/04-vllm-recipes-1.png)

## Why

CLI + containers + a forest of flags. Playground’s pitch:

- **Zero setup** — containers, no manual vLLM install
- **One-click** start/stop, model switch, config
- **Cross-platform** — macOS Apple Silicon, Linux CPU/GPU, Kubernetes / OpenShift
- **Same UI** local and cloud

Roadmap they print: keep pace with official vLLM so new features are clickable. Next: **MCP** server, **RAG**, continuous UI parity.

## Quick start

```bash
pip install vllm-playground
vllm-playground pull          # optional, ~10GB GPU image
vllm-playground
```

Open http://localhost:7860, click Start Server. Orchestrator picks the image.

## Features they list

**UI:** dark theme, chat, icon toolbar (settings, system prompt, structured outputs, tools), per-response token count and gen speed, resizable panels.

**Structured outputs** (four modes):

| Mode | Use they name |
|---|---|
| Choice | closed set (sentiment …) |
| Regex | email / phone / date |
| JSON Schema | API-shaped JSON |
| Grammar (EBNF) | DSLs / formal languages |

**Tool calling:** enable in Server Configuration before start. Auto parsers: Llama 3.x, Mistral, Hermes, Qwen, Granite, InternLM. Presets: Weather, Calculator, Search. Custom tools via JSON Schema. Parallel tool calls.

**Containers:** lifecycle, health, logs; reuse container if config unchanged. Images then:

- GPU: `vllm/vllm-openai:v0.11.0`
- CPU x86: `quay.io/rh_ee_micyang/vllm-cpu:v0.11.0`
- macOS ARM64: `quay.io/rh_ee_micyang/vllm-mac:v0.11.0`

**GuideLLM:** success rate, duration, averages; tok/s mean/median; P50/P75/P90/P95/P99; load patterns; JSON export.

**Recipes:** [vllm-project/recipes](https://github.com/vllm-project/recipes) — 17+ categories (DeepSeek, Qwen, Llama, …), search, one-click fill flags, GPU guidance.

**OpenShift / K8s:**

```bash
cd openshift/
./deploy.sh --gpu
./deploy.sh --cpu
```

Dynamic pods, GPU/CPU detect, RBAC, same UI.

## Architecture

```
Web UI (FastAPI: app.py + index.html + static/)
  ├─ local: container_manager.py → Podman → vLLM container
  └─ cloud: kubernetes_container_manager.py → K8s API → vLLM pods
```

Swap the manager at build time so the UI stays identical.

## macOS ARM

CPU-optimized Apple Silicon image, auto platform detect, rootless Podman. `python run.py` or `vllm-playground`.

## CLI

```bash
vllm-playground                    # defaults
vllm-playground --port 8080
vllm-playground pull               # GPU ~10GB
vllm-playground pull --cpu
vllm-playground pull --all
vllm-playground stop
vllm-playground status
```

No serving TPS in this post — a UI for flags, not a benchmark.
