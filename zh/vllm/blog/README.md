# vLLM 博客

- 必读短表（含最佳顺序）：[MUST-READ.md](MUST-READ.md)
- 旋钮对照：[FLAG-MAP.md](FLAG-MAP.md)
- 机器可读全表（英文）：[en/vllm/blog/CATALOG.md](../../../en/vllm/blog/CATALOG.md)  
  来源 https://vllm.ai/llms.txt · 索引 https://vllm.ai/blog · RSS https://vllm.ai/blog/rss.xml

必读线与第二–五波机制文已落学习译文。CATALOG 其余主要是 day-0 / 单模型 / 活动，仍在补。

| 主题 | 文件 | 原文 |
|---|---|---|
| architecture | [paged-attention.md](architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm |
| architecture | [vs-deepspeed.md](architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed |
| architecture | [v1-alpha.md](architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| architecture | [anatomy.md](architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |
| architecture | [mrv2.md](architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 |
| performance | [v0.6-throughput.md](performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update |
| performance | [spec-decode.md](performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode |
| performance | [fp8-kvcache.md](performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| performance | [production-quality.md](performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality |
| serving | [distributed-inference.md](serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference |
| serving | [production-stack.md](serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release |
| serving | [aibrix.md](serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release |
| serving | [router.md](serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| serving | [epd.md](serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd |
| serving | [large-scale.md](serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving |
| serving | [mooncake.md](serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store |
| serving | [elastic-ep.md](serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism |
| architecture | [torch-compile.md](architecture/torch-compile.md) | https://vllm.ai/blog/2025-08-20-torch-compile |
| architecture | [sleep-mode.md](architecture/sleep-mode.md) | https://vllm.ai/blog/2025-10-26-sleep-mode |
| performance | [struct-decode.md](performance/struct-decode.md) | https://vllm.ai/blog/2025-01-14-struct-decode-intro |
| performance | [dcp.md](performance/dcp.md) | https://vllm.ai/blog/2026-08-07-decode-context-parallelism |
| serving | [kv-offload.md](serving/kv-offload.md) | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| serving | [moriio.md](serving/moriio.md) | https://vllm.ai/blog/2026-04-07-moriio-kv-connector |
| serving | [hybrid-ssm.md](serving/hybrid-ssm.md) | https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg |
| serving | [afd.md](serving/afd.md) | https://vllm.ai/blog/2026-07-23-vllm-afd-plugin |
| architecture | [plugin-system.md](architecture/plugin-system.md) | https://vllm.ai/blog/2025-11-20-vllm-plugin-system |
| architecture | [hardware-plugin.md](architecture/hardware-plugin.md) | https://vllm.ai/blog/2025-05-12-hardware-plugin |
| architecture | [triton-attn.md](architecture/triton-attn.md) | https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive |
| serving | [shm-ipc.md](serving/shm-ipc.md) | https://vllm.ai/blog/2025-11-13-shm-ipc-cache |
| serving | [pegaflow.md](serving/pegaflow.md) | https://vllm.ai/blog/2026-05-18-pegaflow |
| performance | [turboquant.md](performance/turboquant.md) | https://vllm.ai/blog/2026-05-11-turboquant |
| serving | [native-rl.md](serving/native-rl.md) | https://vllm.ai/blog/2026-05-28-native-rl-apis |
| serving | [ray-symmetric.md](serving/ray-symmetric.md) | https://vllm.ai/blog/2025-11-22-ray-symmetric-run |
| architecture | [extract-hidden-states.md](architecture/extract-hidden-states.md) | https://vllm.ai/blog/2026-03-30-extract-hidden-states |
| performance | [p-eagle.md](performance/p-eagle.md) | https://vllm.ai/blog/2026-03-13-p-eagle |
| performance | [parallel-drafting.md](performance/parallel-drafting.md) | https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting |
| performance | [dspark-adaptive.md](performance/dspark-adaptive.md) | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification |
| serving | [streaming-realtime.md](serving/streaming-realtime.md) | https://vllm.ai/blog/2026-01-31-streaming-realtime |
| serving | [tilert.md](serving/tilert.md) | https://vllm.ai/blog/2026-07-14-vllm-tilert-pd |
| architecture | [deepseek-v4.md](architecture/deepseek-v4.md) | https://vllm.ai/blog/2026-04-24-deepseek-v4 |
| architecture | [rocm-attention.md](architecture/rocm-attention.md) | https://vllm.ai/blog/2026-02-27-rocm-attention-backend |
| serving | [gb200-wideep.md](serving/gb200-wideep.md) | https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1 |
| serving | [openrlhf.md](serving/openrlhf.md) | https://vllm.ai/blog/2025-04-23-openrlhf-vllm |
| architecture | [lfai-roadmap.md](architecture/lfai-roadmap.md) | https://vllm.ai/blog/2024-07-25-lfai-perf |
| performance | [spec-decode-amd.md](performance/spec-decode-amd.md) | https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus |
| serving | [rdt-weight-transfer.md](serving/rdt-weight-transfer.md) | https://vllm.ai/blog/2026-08-22-rdt-weight-transfer |
| serving | [isoexec.md](serving/isoexec.md) | https://vllm.ai/blog/2026-08-21-isoexec |
| serving | [bitwise-rl.md](serving/bitwise-rl.md) | https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference |
| serving | [agent-lightning.md](serving/agent-lightning.md) | https://vllm.ai/blog/2025-10-22-agent-lightning |
| performance | [speculators-v030.md](performance/speculators-v030.md) | https://vllm.ai/blog/2025-12-13-speculators-v030 |
| performance | [speculators-v050.md](performance/speculators-v050.md) | https://vllm.ai/blog/2026-05-28-speculators-v050 |
| performance | [eagle-3-1.md](performance/eagle-3-1.md) | https://vllm.ai/blog/2026-05-26-eagle-3-1 |
| architecture | [cuda-debugging.md](architecture/cuda-debugging.md) | https://vllm.ai/blog/2025-08-11-cuda-debugging |
| architecture | [cuda-debugging-source.md](architecture/cuda-debugging-source.md) | https://vllm.ai/blog/2025-12-03-improved-cuda-debugging |
| architecture | [vllm-tpu.md](architecture/vllm-tpu.md) | https://vllm.ai/blog/2025-10-16-vllm-tpu |
| performance | [ptpc-fp8.md](performance/ptpc-fp8.md) | https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm |
| architecture | [transformers-backend.md](architecture/transformers-backend.md) | https://vllm.ai/blog/2025-04-11-transformers-backend |
| serving | [semantic-router.md](serving/semantic-router.md) | https://vllm.ai/blog/2025-09-11-semantic-router |
| serving | [semantic-router-iris.md](serving/semantic-router-iris.md) | https://vllm.ai/blog/2026-01-05-vllm-sr-iris |
| serving | [vllm-omni.md](serving/vllm-omni.md) | https://vllm.ai/blog/2025-11-30-vllm-omni |
| map | [FLAG-MAP.md](FLAG-MAP.md) | `optimization.md` 旋钮 → 博客 |
