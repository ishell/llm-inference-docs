---
source: https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Llama Stack × vLLM：inference 是可换的 Provider，不是另一套引擎

英文对照：`en/vllm/blog/serving/llama-stack.md`  
原文：https://vllm.ai/blog/2025-01-27-intro-to-llama-stack-with-vllm  
Red Hat + Meta。演示 Llama-3.2-1B CPU 容器。

两种：`remote::vllm`（打 OpenAI-compatible `/v1`）和 inline（跟 Stack 同进程）。安全、agent、vector 仍是 Stack 自己的 provider。K8s 示例：vLLM Service DNS `vllm-server.default.svc…:8000/v1`，Stack 只填 URL。教程偏 2025-01 的 `llama stack build` YAML，API 会漂。要点是应用生命周期同一套 API，底下换引擎。

本地图（原文版权仍归原站；学习对照用）：

![llama stack](../../../../assets/vllm/blog/serving/llama-stack/01-llama-stack.png)
