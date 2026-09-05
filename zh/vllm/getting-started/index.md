---
source: https://docs.vllm.ai/en/stable/
lang: zh
fetched: 2026-09-04
---

# vLLM 文档入口

稳定版：https://docs.vllm.ai/en/stable/  ·  滚动：https://docs.vllm.ai/en/latest/  
英文对照：[en/vllm/getting-started/index.md](../../../en/vllm/getting-started/index.md)

PagedAttention、continuous batching、chunked prefill、prefix cache、CUDA/HIP graph、speculative decoding、P/D 分离、多种量化、OpenAI / Anthropic / gRPC。不是官方译本。

本地顺序：

- 安装 / 离线 / `vllm serve`：[quickstart.md](quickstart.md)
- 服务端性能旗标（不是整页 CLI）：[serve.md](serve.md)
- 调优顺序：[optimization.md](../optimization/optimization.md)
- 客户端尺子：[cli.md](../benchmarking/cli.md) · 网格搜 batch：[auto-tune.md](../benchmarking/auto-tune.md)
- 屋里的钟：[/metrics](../metrics/production-metrics.md) · 怎么算：[design-metrics.md](../metrics/design-metrics.md)
- 功能： [APC](../features/prefix-caching.md) · [记账](../features/prefix-caching-design.md) · [投机解码](../features/speculative-decoding.md) · [V1](../features/v1-guide.md)
