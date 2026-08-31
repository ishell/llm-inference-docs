# LLM 推理笔记（网站 × 主题）

个人学习库，**不是官方译本**。

- `en/` 英文（抓取或摘录）
- `zh/` 中文（全译或摘译；文件头 `source:` 是原文）

指标名、CLI、公式保留英文。网页里的图本地只留图注。

## 目录

```
{en,zh}/
  nvidia/
    benchmarking/         # 压测 vs 性能测试、NIM 手册、博客 1–2、GenAI-Perf 客户端文
    performance-tuning/   # Mastering、博客 3、TensorRT-LLM 全套
    cost/                 # 博客 4 TCO
    tools/                # AIPerf、GenAI-Perf、Perf Analyzer、Triton
  vllm/
    getting-started/      # 文档入口、quickstart、serve CLI
    optimization/
    benchmarking/         # bench CLI、auto-tune
    metrics/              # /metrics 表 + 设计
    features/             # prefix cache、spec decode、V1
    blog/                 # CATALOG 全表、必读列表、anatomy
```

## NVIDIA · benchmarking

| 本地 | 原文 | 完整度 |
|---|---|---|
| [nim-index](zh/nvidia/benchmarking/nim-index.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html | 目录 |
| [nim-01-overview](zh/nvidia/benchmarking/nim-01-overview.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html | 全译 |
| [nim-02-metrics](zh/nvidia/benchmarking/nim-02-metrics.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html | 全译 |
| [nim-03-parameters](zh/nvidia/benchmarking/nim-03-parameters.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html | 全译 |
| [nim-04-aiperf](zh/nvidia/benchmarking/nim-04-aiperf.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/quickstart.html | 全译 |
| [nim-05-lora](zh/nvidia/benchmarking/nim-05-lora.md) | https://docs.nvidia.com/nim/benchmarking/llm/latest/benchmarking-lora.html | 全译 |
| [nim-product-benchmarking](zh/nvidia/benchmarking/nim-product-benchmarking.md) | https://docs.nvidia.com/nim/large-language-models/latest/reference/benchmarking.html | 摘译 |
| [blog-01-fundamental-concepts](zh/nvidia/benchmarking/blog-01-fundamental-concepts.md) | https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/ | 摘译 |
| [blog-02-genai-perf-and-nim](zh/nvidia/benchmarking/blog-02-genai-perf-and-nim.md) | https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/ | 导读 |
| [blog-genai-perf-openai](zh/nvidia/benchmarking/blog-genai-perf-openai.md) | https://developer.nvidia.com/blog/measuring-generative-ai-model-performance-using-nvidia-genai-perf-and-an-openai-compatible-api/ | 导读 |

## NVIDIA · performance-tuning

| 本地 | 原文 | 完整度 |
|---|---|---|
| [mastering-llm-techniques](zh/nvidia/performance-tuning/mastering-llm-techniques.md) | https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ | 导读 |
| [blog-03-tensorrt-llm](zh/nvidia/performance-tuning/blog-03-tensorrt-llm.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/ | 导读 |
| [trtllm-product](zh/nvidia/performance-tuning/trtllm-product.md) | https://developer.nvidia.com/tensorrt-llm | 入口 |
| [trtllm-tuning-guide](zh/nvidia/performance-tuning/trtllm-tuning-guide.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/ | 摘译 |
| [trtllm-build-flags](zh/nvidia/performance-tuning/trtllm-build-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html | 摘译 |
| [trtllm-runtime-flags](zh/nvidia/performance-tuning/trtllm-runtime-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html | 摘译 |
| [trtllm-max-batch](zh/nvidia/performance-tuning/trtllm-max-batch.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html | 摘译 |
| [trtllm-fp8](zh/nvidia/performance-tuning/trtllm-fp8.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html | 摘译 |
| [trtllm-kvcache](zh/nvidia/performance-tuning/trtllm-kvcache.md) | https://nvidia.github.io/TensorRT-LLM/features/kvcache.html | 摘译 |
| [trtllm-paged-attention-ifb](zh/nvidia/performance-tuning/trtllm-paged-attention-ifb.md) | https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html | 摘译 |
| [trtllm-bench](zh/nvidia/performance-tuning/trtllm-bench.md) | https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html | 英文全文 + 中文导读 |

## NVIDIA · cost

| 本地 | 原文 | 完整度 |
|---|---|---|
| [blog-04-tco](zh/nvidia/cost/blog-04-tco.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ | 导读 |

## NVIDIA · tools

| 本地 | 原文 | 完整度 |
|---|---|---|
| [aiperf](zh/nvidia/tools/aiperf.md) | https://github.com/ai-dynamo/aiperf | 摘译 |
| [aiperf-load-generator](zh/nvidia/tools/aiperf-load-generator.md) | https://docs.nvidia.com/aiperf/benchmark-modes/load-generator-options-reference | 摘译 |
| [genai-perf](zh/nvidia/tools/genai-perf.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html | 英文全文 + 中文导读 |
| [perf-analyzer](zh/nvidia/tools/perf-analyzer.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf-analyzer-README.html | 摘译 |
| [triton-performance-tuning](zh/nvidia/tools/triton-performance-tuning.md) | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html | 摘译 |

## vLLM

| 主题 | 本地 | 原文 | 完整度 |
|---|---|---|---|
| getting-started | [index](zh/vllm/getting-started/index.md) | https://docs.vllm.ai/en/stable/ 与 https://docs.vllm.ai/en/latest/ | 入口 |
| getting-started | [quickstart](zh/vllm/getting-started/quickstart.md) | https://docs.vllm.ai/en/stable/getting_started/quickstart/ | 摘译 |
| getting-started | [serve](zh/vllm/getting-started/serve.md) | https://docs.vllm.ai/en/stable/cli/serve/ | 性能相关旗标（非整页 CLI） |
| optimization | [optimization](zh/vllm/optimization/optimization.md) | https://docs.vllm.ai/en/stable/configuration/optimization/ | 全译 |
| benchmarking | [cli](zh/vllm/benchmarking/cli.md) | https://docs.vllm.ai/en/stable/benchmarking/cli/ | 摘译 |
| benchmarking | [auto-tune](zh/vllm/benchmarking/auto-tune.md) | https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md | 摘译 |
| metrics | [production-metrics](zh/vllm/metrics/production-metrics.md) | https://docs.vllm.ai/en/stable/usage/metrics/ | 摘译 |
| metrics | [design-metrics](zh/vllm/metrics/design-metrics.md) | https://docs.vllm.ai/en/stable/design/metrics/ | 摘译 |
| features | [prefix-caching](zh/vllm/features/prefix-caching.md) | https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/ | 摘译 |
| features | [prefix-caching-design](zh/vllm/features/prefix-caching-design.md) | https://docs.vllm.ai/en/stable/design/prefix_caching/ | 摘译 |
| features | [speculative-decoding](zh/vllm/features/speculative-decoding.md) | https://docs.vllm.ai/en/stable/features/speculative_decoding/ | 摘译 |
| features | [v1-guide](zh/vllm/features/v1-guide.md) | https://docs.vllm.ai/en/stable/usage/v1_guide/ | 摘译 |
| blog | [MUST-READ](zh/vllm/blog/MUST-READ.md) | https://vllm.ai/blog · https://vllm.ai/blog/rss.xml | 必读 URL |
| blog | [CATALOG](en/vllm/blog/CATALOG.md) | https://vllm.ai/llms.txt | 全表（英文） |
| blog / architecture | [anatomy](zh/vllm/blog/architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm | 英文全文 + 中文导读 |

## 建议阅读顺序

1. `zh/nvidia/benchmarking/blog-01-fundamental-concepts.md`
2. `nim-02-metrics` → `nim-03-parameters` → `nim-04-aiperf`
3. `zh/nvidia/tools/aiperf-load-generator.md`
4. `zh/vllm/optimization/optimization.md` 与 `zh/vllm/getting-started/serve.md`
5. `zh/vllm/blog/architecture/anatomy.md`

## 刻意不做的

- vLLM 博客 100+ 篇正文（URL 全在 `CATALOG.md`，必读在 `MUST-READ.md`）
- `vllm serve` 生成页里每一个旗标

抓取：2026-08-30 / 2026-08-31。
