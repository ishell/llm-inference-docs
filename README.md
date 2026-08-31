# LLM 推理笔记（网站 × 主题）

个人学习库，**不是官方译本**。中文是学习笔记的重写：指标名、CLI、公式保留英文。

从「那条评论在指什么」读起：[zh/GUIDE.md](zh/GUIDE.md)。

**完整度请先看一句实话：** 评论里的官方 URL **几乎都有本地笔记**（NVIDIA 手册/博客/工具/TRT-LLM，以及 vLLM 文档页），但 **不是全文下载 + 完整翻译**。vLLM 博客全表仍在 `en/vllm/blog/CATALOG.md`；[必读列表](zh/vllm/blog/MUST-READ.md) 已全部写成学习译文，CATALOG 其余为 day-0 / 单模型 / 活动；`vllm serve` 只摘了性能相关旗标。即便标过「全译」的章节，也是压缩后的对照笔记，篇幅往往短于英文网页。版权也不允许把官方页面原样搬进来。

- `en/` 英文（抓取或摘录）
- `zh/` 中文（全译、摘译或导读；文件头 `source:` 是原文。最短路径几篇用了偏文学的科技笔记笔调：把「等待第一个 token」当成人的事情来写，但不改公式。）

网页里的图本地只留图注。NVIDIA Developer Blog 那一组（系列 1–4、GenAI-Perf OpenAI 文、Mastering LLM Techniques）已按同一笔调写成学习译文；TensorRT-LLM **Performance Tuning Guide** 六章及 KV / IFB / bench 邻居页也是；vLLM 必读博客同款。图仍在原网页。

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
| [blog-01-fundamental-concepts](zh/nvidia/benchmarking/blog-01-fundamental-concepts.md) | https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/ | 学习译文 |
| [blog-02-genai-perf-and-nim](zh/nvidia/benchmarking/blog-02-genai-perf-and-nim.md) | https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/ | 学习译文 |
| [blog-genai-perf-openai](zh/nvidia/benchmarking/blog-genai-perf-openai.md) | https://developer.nvidia.com/blog/measuring-generative-ai-model-performance-using-nvidia-genai-perf-and-an-openai-compatible-api/ | 学习译文 |

## NVIDIA · performance-tuning

| 本地 | 原文 | 完整度 |
|---|---|---|
| [mastering-llm-techniques](zh/nvidia/performance-tuning/mastering-llm-techniques.md) | https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ | 学习译文 |
| [blog-03-tensorrt-llm](zh/nvidia/performance-tuning/blog-03-tensorrt-llm.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/ | 学习译文 |
| [trtllm-product](zh/nvidia/performance-tuning/trtllm-product.md) | https://developer.nvidia.com/tensorrt-llm | 入口 |
| [trtllm-tuning-guide](zh/nvidia/performance-tuning/trtllm-tuning-guide.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/ | 学习译文 |
| [trtllm-baseline](zh/nvidia/performance-tuning/trtllm-baseline.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html | 学习译文 |
| [trtllm-build-flags](zh/nvidia/performance-tuning/trtllm-build-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html | 学习译文 |
| [trtllm-max-batch](zh/nvidia/performance-tuning/trtllm-max-batch.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html | 学习译文 |
| [trtllm-sharding](zh/nvidia/performance-tuning/trtllm-sharding.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/deciding-model-sharding-strategy.html | 学习译文 |
| [trtllm-fp8](zh/nvidia/performance-tuning/trtllm-fp8.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html | 学习译文 |
| [trtllm-runtime-flags](zh/nvidia/performance-tuning/trtllm-runtime-flags.md) | https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html | 学习译文 |
| [trtllm-kvcache](zh/nvidia/performance-tuning/trtllm-kvcache.md) | https://nvidia.github.io/TensorRT-LLM/features/kvcache.html | 学习译文 |
| [trtllm-paged-attention-ifb](zh/nvidia/performance-tuning/trtllm-paged-attention-ifb.md) | https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html | 学习译文 |
| [trtllm-bench](zh/nvidia/performance-tuning/trtllm-bench.md) | https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html | 英文全文 + 中文学习译文 |

## NVIDIA · cost

| 本地 | 原文 | 完整度 |
|---|---|---|
| [blog-04-tco](zh/nvidia/cost/blog-04-tco.md) | https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ | 学习译文 |

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
| blog | [MUST-READ](zh/vllm/blog/MUST-READ.md) | https://vllm.ai/blog · https://vllm.ai/blog/rss.xml | 必读 + 本地链接 |
| blog | [FLAG-MAP](zh/vllm/blog/FLAG-MAP.md) | `optimization.md` 旋钮 → 博客 | 对照表 |
| blog | [CATALOG](en/vllm/blog/CATALOG.md) | https://vllm.ai/llms.txt | 全表（英文） |
| blog / architecture | [paged-attention](zh/vllm/blog/architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm | 学习译文 |
| blog / architecture | [vs-deepspeed](zh/vllm/blog/architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed | 学习译文 |
| blog / architecture | [v1-alpha](zh/vllm/blog/architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release | 学习译文 |
| blog / architecture | [anatomy](zh/vllm/blog/architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm | 英文全文 + 中文导读 |
| blog / architecture | [mrv2](zh/vllm/blog/architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 | 学习译文 |
| blog / performance | [v0.6-throughput](zh/vllm/blog/performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update | 学习译文 |
| blog / performance | [spec-decode](zh/vllm/blog/performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode | 学习译文 |
| blog / performance | [fp8-kvcache](zh/vllm/blog/performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache | 学习译文 |
| blog / performance | [production-quality](zh/vllm/blog/performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality | 学习译文 |
| blog / serving | [distributed-inference](zh/vllm/blog/serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference | 学习译文 |
| blog / serving | [production-stack](zh/vllm/blog/serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release | 学习译文 |
| blog / serving | [aibrix](zh/vllm/blog/serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release | 学习译文 |
| blog / serving | [router](zh/vllm/blog/serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release | 学习译文 |
| blog / serving | [epd](zh/vllm/blog/serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd | 学习译文 |
| blog / serving | [large-scale](zh/vllm/blog/serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving | 学习译文 |
| blog / serving | [mooncake](zh/vllm/blog/serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store | 学习译文 |
| blog / serving | [elastic-ep](zh/vllm/blog/serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism | 学习译文 |

## 建议阅读顺序

1. [zh/GUIDE.md](zh/GUIDE.md)
2. `zh/nvidia/benchmarking/blog-01-fundamental-concepts.md`
3. `nim-01` → `nim-02` → `nim-03` → `nim-04-aiperf`
4. `zh/vllm/optimization/optimization.md` 与 `zh/vllm/getting-started/serve.md`
5. `zh/vllm/blog/MUST-READ.md`（立项 → Anatomy → V1 → MRV2 → 性能 → 切卡 → 集群 → Router → EPD → Wide-EP → Mooncake → Elastic EP）

## 刻意不做的

- vLLM 博客 CATALOG 里必读以外的 day-0 / 单模型 / 活动文（URL 仍在 `CATALOG.md`）
- `vllm serve` 生成页里每一个旗标

抓取：2026-08-30 / 2026-08-31。
