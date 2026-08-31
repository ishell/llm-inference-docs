---
source: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
lang: zh
fetched: 2026-08-31
---

# GenAI-Perf（中文导读）

英文全文：`en/nvidia/tools/genai-perf.md`

**已停更。新项目用 AIPerf**（`aiperf-load-generator.md`）。概念仍适用。

客户端打生成式模型：output token throughput、TTFT、TTST、ITL、request throughput。服务必须先起来。支持 OpenAI chat/completions、Triton TensorRT-LLM backend 等。

空内容的首包不算 TTFT。结果打表，并写 csv/json。
