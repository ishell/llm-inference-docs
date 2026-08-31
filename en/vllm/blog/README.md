# vLLM blog notes

- Short list (best order): [MUST-READ.md](MUST-READ.md)
- Knob map: [FLAG-MAP.md](FLAG-MAP.md)
- Full machine table: [CATALOG.md](CATALOG.md) from https://vllm.ai/llms.txt  
  Hub: https://vllm.ai/blog · RSS: https://vllm.ai/blog/rss.xml

MUST-READ plus second- and third-wave mechanics posts have local notes. Remaining CATALOG entries are day-0 / single-model / events.

| Folder | File | Source |
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
| map | [FLAG-MAP.md](FLAG-MAP.md) | optimization knobs → blogs |
