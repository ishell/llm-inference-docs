# LLM 推理笔记（网站 × 主题）

个人学习库，**不是官方译本**。

- `en/` 英文（抓取或摘录）
- `zh/` 中文（全译或摘译，文件头 `source:` 是原文）

指标名、CLI、公式保留英文。图在网页里，本地只留图注。

## 目录怎么读

```
{en,zh}/
  nvidia/          # NVIDIA
    benchmarking/  # 压测 vs 性能测试、NIM 手册、AIPerf 实操
    performance-tuning/  # 推理优化、TensorRT-LLM
    cost/          # TCO / 每 token 成本
    tools/         # AIPerf、GenAI-Perf、Triton Perf Analyzer
  vllm/            # vLLM
    getting-started/
    optimization/
    benchmarking/
    metrics/
    features/      # prefix cache、spec decode、V1
    blog/          # CATALOG 全列表 + anatomy
```

## NVIDIA · benchmarking（压测 / 指标）

| zh | 原文 |
|---|---|
| [nim-01-overview](zh/nvidia/benchmarking/nim-01-overview.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html |
| [nim-02-metrics](zh/nvidia/benchmarking/nim-02-metrics.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html |
| [nim-03-parameters](zh/nvidia/benchmarking/nim-03-parameters.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html |
| [nim-04-aiperf](zh/nvidia/benchmarking/nim-04-aiperf.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/quickstart.html |
| [nim-05-lora](zh/nvidia/benchmarking/nim-05-lora.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/benchmarking-lora.html |
| [nim-product-benchmarking](zh/nvidia/benchmarking/nim-product-benchmarking.md) | https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html |
| [blog-01-fundamental-concepts](zh/nvidia/benchmarking/blog-01-fundamental-concepts.md) | https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/ |
| [blog-02-genai-perf-and-nim](zh/nvidia/benchmarking/blog-02-genai-perf-and-nim.md) | https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/ |

## NVIDIA · performance-tuning

| zh | 原文 |
|---|---|
| [mastering-llm-techniques](zh/nvidia/performance-tuning/mastering-llm-techniques.md) | https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ |
| [blog-03-tensorrt-llm](zh/nvidia/performance-tuning/blog-03-tensorrt-llm.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/ |
| [trtllm-tuning-guide](zh/nvidia/performance-tuning/trtllm-tuning-guide.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/ |
| [trtllm-build-flags](zh/nvidia/performance-tuning/trtllm-build-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html |
| [trtllm-runtime-flags](zh/nvidia/performance-tuning/trtllm-runtime-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html |
| [trtllm-kvcache](zh/nvidia/performance-tuning/trtllm-kvcache.md) | https://nvidia.github.io/TensorRT-LLM/features/kvcache.html |
| [trtllm-bench](zh/nvidia/performance-tuning/trtllm-bench.md) | https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html |

## NVIDIA · cost

| zh | 原文 |
|---|---|
| [blog-04-tco](zh/nvidia/cost/blog-04-tco.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ |

## NVIDIA · tools（压测工具）

| zh | 原文 |
|---|---|
| [aiperf](zh/nvidia/tools/aiperf.md) | https://github.com/ai-dynamo/aiperf |
| [aiperf-load-generator](zh/nvidia/tools/aiperf-load-generator.md) | https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference |
| [genai-perf](zh/nvidia/tools/genai-perf.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html |
| [perf-analyzer](zh/nvidia/tools/perf-analyzer.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html |
| [triton-performance-tuning](zh/nvidia/tools/triton-performance-tuning.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html |

## vLLM

| 主题 | zh | 原文 |
|---|---|---|
| getting-started | [quickstart](zh/vllm/getting-started/quickstart.md) | https://docs.vllm.ai/en/stable/getting_started/quickstart/ |
| optimization | [optimization](zh/vllm/optimization/optimization.md) | https://docs.vllm.ai/en/stable/configuration/optimization/ |
| benchmarking | [cli](zh/vllm/benchmarking/cli.md) | https://docs.vllm.ai/en/stable/benchmarking/cli/ |
| benchmarking | [auto-tune](zh/vllm/benchmarking/auto-tune.md) | https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md |
| metrics | [production-metrics](zh/vllm/metrics/production-metrics.md) | https://docs.vllm.ai/en/stable/usage/metrics/ |
| features | [prefix-caching](zh/vllm/features/prefix-caching.md) | https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/ |
| features | [prefix-caching-design](zh/vllm/features/prefix-caching-design.md) | https://docs.vllm.ai/en/stable/design/prefix_caching/ |
| features | [speculative-decoding](zh/vllm/features/speculative-decoding.md) | https://docs.vllm.ai/en/stable/features/speculative_decoding/ |
| features | [v1-guide](zh/vllm/features/v1-guide.md) | https://docs.vllm.ai/en/stable/usage/v1_guide/ |
| blog | [README + CATALOG](zh/vllm/blog/README.md) | https://vllm.ai/llms.txt |
| blog / architecture | [anatomy](zh/vllm/blog/architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |

`vllm serve` 完整 CLI 太长，未整页落地：https://docs.vllm.ai/en/stable/cli/serve/

## 建议阅读顺序

1. `zh/nvidia/benchmarking/blog-01-fundamental-concepts.md`
2. `nim-02-metrics` → `nim-03-parameters` → `nim-04-aiperf`
3. `zh/nvidia/tools/aiperf-load-generator.md`
4. `zh/vllm/optimization/optimization.md`
5. `zh/vllm/blog/architecture/anatomy.md`（导读；英文全文在 `en/`）

## 仍未逐页全译

- vLLM 博客除 Anatomy 外的 100+ 篇（目录在 `en/vllm/blog/CATALOG.md`）
- `Mastering LLM Techniques`、Anatomy、GenAI-Perf、trtllm-bench：英文全文 + 中文导读/摘译
- TensorRT-LLM「tuning max batch / max num tokens」独立页、paged-attention-IFB 专页：要点已并入 runtime/kvcache/build-flags

抓取：2026-08-30 / 2026-08-31。
