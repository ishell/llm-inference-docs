---
source: https://vllm.ai/llms.txt
lang: en
fetched: 2026-08-31
---

# vLLM blogs to read first

Full machine list: [`CATALOG.md`](CATALOG.md). Hub: https://vllm.ai/blog · RSS: https://vllm.ai/blog/rss.xml

Knob map: [FLAG-MAP.md](FLAG-MAP.md) (`optimization.md` flags → these posts).

**Best order:** launch → Anatomy → V1 → MRV2 → (optional DeepSpeed notes) → v0.6 CPU → spec decode → FP8 KV → production CI → distributed sharding → production-stack → AIBrix → Router → encoder disagg → wide-EP → Mooncake → Elastic EP.

“Not yet supported” in V1 / spec-decode is historical. The EPD post is **encoder** disaggregation; text P/D lives in Router + large-scale. Mooncake is the cross-instance KV pool (agent prefixes). Elastic EP resizes DP at runtime.

## Architecture

| Post | Local | URL |
|---|---|---|
| PagedAttention launch | [paged-attention.md](architecture/paged-attention.md) | https://vllm.ai/blog/2023-06-20-vllm |
| Anatomy of vLLM | [anatomy.md](architecture/anatomy.md) | https://vllm.ai/blog/2025-09-05-anatomy-of-vllm |
| V1 alpha | [v1-alpha.md](architecture/v1-alpha.md) | https://vllm.ai/blog/2025-01-27-v1-alpha-release |
| Model Runner V2 | [mrv2.md](architecture/mrv2.md) | https://vllm.ai/blog/2026-03-24-mrv2 |
| vLLM vs DeepSpeed | [vs-deepspeed.md](architecture/vs-deepspeed.md) | https://vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed |

## Performance

| Post | Local | URL |
|---|---|---|
| v0.6 throughput | [v0.6-throughput.md](performance/v0.6-throughput.md) | https://vllm.ai/blog/2024-09-05-perf-update |
| Speculative decoding | [spec-decode.md](performance/spec-decode.md) | https://vllm.ai/blog/2024-10-17-spec-decode |
| FP8 KV cache | [fp8-kvcache.md](performance/fp8-kvcache.md) | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| Production-quality CI | [production-quality.md](performance/production-quality.md) | https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality |

## Serving

| Post | Local | URL |
|---|---|---|
| Distributed inference | [distributed-inference.md](serving/distributed-inference.md) | https://vllm.ai/blog/2025-02-17-distributed-inference |
| Production stack | [production-stack.md](serving/production-stack.md) | https://vllm.ai/blog/2025-01-21-stack-release |
| AIBrix | [aibrix.md](serving/aibrix.md) | https://vllm.ai/blog/2025-02-21-aibrix-release |
| Router | [router.md](serving/router.md) | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| Encoder disagg (EPD) | [epd.md](serving/epd.md) | https://vllm.ai/blog/2025-12-15-vllm-epd |
| Large-scale serving | [large-scale.md](serving/large-scale.md) | https://vllm.ai/blog/2025-12-17-large-scale-serving |
| Mooncake Store | [mooncake.md](serving/mooncake.md) | https://vllm.ai/blog/2026-05-06-mooncake-store |
| Elastic EP | [elastic-ep.md](serving/elastic-ep.md) | https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism |

## Second wave (mechanics, after the main line)

Not day-0 model posts. Order: torch.compile → Sleep Mode → structured decoding → DCP → KV offload → single-node P/D (MORI-IO) → Hybrid SSM → AFD.

| Post | Local | URL |
|---|---|---|
| torch.compile | [torch-compile.md](architecture/torch-compile.md) | https://vllm.ai/blog/2025-08-20-torch-compile |
| Sleep Mode | [sleep-mode.md](architecture/sleep-mode.md) | https://vllm.ai/blog/2025-10-26-sleep-mode |
| Structured decoding | [struct-decode.md](performance/struct-decode.md) | https://vllm.ai/blog/2025-01-14-struct-decode-intro |
| Decode Context Parallelism | [dcp.md](performance/dcp.md) | https://vllm.ai/blog/2026-08-07-decode-context-parallelism |
| KV offloading connector | [kv-offload.md](serving/kv-offload.md) | https://vllm.ai/blog/2026-01-08-kv-offloading-connector |
| Single-node P/D (MORI-IO) | [moriio.md](serving/moriio.md) | https://vllm.ai/blog/2026-04-07-moriio-kv-connector |
| Hybrid SSM disagg | [hybrid-ssm.md](serving/hybrid-ssm.md) | https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg |
| AFD Plugin | [afd.md](serving/afd.md) | https://vllm.ai/blog/2026-07-23-vllm-afd-plugin |

## Third wave (plugins / KV pool / quant / RL)

After the second wave. Still not day-0. Order: plugins → hardware plugin → Triton attention → SHM IPC → PegaFlow → TurboQuant → Native RL → Ray symmetric-run.

| Post | Local | URL |
|---|---|---|
| Plugin system | [plugin-system.md](architecture/plugin-system.md) | https://vllm.ai/blog/2025-11-20-vllm-plugin-system |
| Hardware plugin | [hardware-plugin.md](architecture/hardware-plugin.md) | https://vllm.ai/blog/2025-05-12-hardware-plugin |
| Triton attention | [triton-attn.md](architecture/triton-attn.md) | https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive |
| SHM IPC cache | [shm-ipc.md](serving/shm-ipc.md) | https://vllm.ai/blog/2025-11-13-shm-ipc-cache |
| PegaFlow | [pegaflow.md](serving/pegaflow.md) | https://vllm.ai/blog/2026-05-18-pegaflow |
| TurboQuant | [turboquant.md](performance/turboquant.md) | https://vllm.ai/blog/2026-05-11-turboquant |
| Native RL APIs | [native-rl.md](serving/native-rl.md) | https://vllm.ai/blog/2026-05-28-native-rl-apis |
| Ray symmetric-run | [ray-symmetric.md](serving/ray-symmetric.md) | https://vllm.ai/blog/2025-11-22-ray-symmetric-run |

## Fourth wave (spec follow-ons / heterogeneous serving / historical roadmap)

After wave 3. Order: hidden extract → P-EAGLE → parallel drafting → DSpark adaptive → streaming input → TileRT → DeepSeek V4 → ROCm attention → GB200 Wide-EP → OpenRLHF → LF AI roadmap.

| Post | Local | URL |
|---|---|---|
| Hidden-state extract | [extract-hidden-states.md](architecture/extract-hidden-states.md) | https://vllm.ai/blog/2026-03-30-extract-hidden-states |
| P-EAGLE | [p-eagle.md](performance/p-eagle.md) | https://vllm.ai/blog/2026-03-13-p-eagle |
| Parallel drafting | [parallel-drafting.md](performance/parallel-drafting.md) | https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting |
| DSpark adaptive verify | [dspark-adaptive.md](performance/dspark-adaptive.md) | https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification |
| Streaming / realtime | [streaming-realtime.md](serving/streaming-realtime.md) | https://vllm.ai/blog/2026-01-31-streaming-realtime |
| TileRT P/D | [tilert.md](serving/tilert.md) | https://vllm.ai/blog/2026-07-14-vllm-tilert-pd |
| DeepSeek V4 | [deepseek-v4.md](architecture/deepseek-v4.md) | https://vllm.ai/blog/2026-04-24-deepseek-v4 |
| ROCm attention | [rocm-attention.md](architecture/rocm-attention.md) | https://vllm.ai/blog/2026-02-27-rocm-attention-backend |
| GB200 Wide-EP | [gb200-wideep.md](serving/gb200-wideep.md) | https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1 |
| OpenRLHF | [openrlhf.md](serving/openrlhf.md) | https://vllm.ai/blog/2025-04-23-openrlhf-vllm |
| LF AI roadmap | [lfai-roadmap.md](architecture/lfai-roadmap.md) | https://vllm.ai/blog/2024-07-25-lfai-perf |

## Fifth wave (alignment / draft training / TPU / routing / Omni)

After wave 4. Order: AMD spec → RDT weights → IsoExec → bitwise RL → token IDs → Speculators 0.3/0.5 → EAGLE 3.1 → CUDA dump → TPU → PTPC-FP8 → Transformers backend → Semantic Router → Iris → Omni.

| Post | Local | URL |
|---|---|---|
| AMD spec decode | [spec-decode-amd.md](performance/spec-decode-amd.md) | https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus |
| RDT sharded weights | [rdt-weight-transfer.md](serving/rdt-weight-transfer.md) | https://vllm.ai/blog/2026-08-22-rdt-weight-transfer |
| IsoExec | [isoexec.md](serving/isoexec.md) | https://vllm.ai/blog/2026-08-21-isoexec |
| Bitwise RL | [bitwise-rl.md](serving/bitwise-rl.md) | https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference |
| return_token_ids | [agent-lightning.md](serving/agent-lightning.md) | https://vllm.ai/blog/2025-10-22-agent-lightning |
| Speculators 0.3 | [speculators-v030.md](performance/speculators-v030.md) | https://vllm.ai/blog/2025-12-13-speculators-v030 |
| Speculators 0.5 | [speculators-v050.md](performance/speculators-v050.md) | https://vllm.ai/blog/2026-05-28-speculators-v050 |
| EAGLE 3.1 | [eagle-3-1.md](performance/eagle-3-1.md) | https://vllm.ai/blog/2026-05-26-eagle-3-1 |
| CUDA core dump | [cuda-debugging.md](architecture/cuda-debugging.md) | https://vllm.ai/blog/2025-08-11-cuda-debugging |
| Hang → source line | [cuda-debugging-source.md](architecture/cuda-debugging-source.md) | https://vllm.ai/blog/2025-12-03-improved-cuda-debugging |
| vLLM TPU | [vllm-tpu.md](architecture/vllm-tpu.md) | https://vllm.ai/blog/2025-10-16-vllm-tpu |
| PTPC-FP8 | [ptpc-fp8.md](performance/ptpc-fp8.md) | https://vllm.ai/blog/2025-02-24-ptpc-fp8-rocm |
| Transformers backend | [transformers-backend.md](architecture/transformers-backend.md) | https://vllm.ai/blog/2025-04-11-transformers-backend |
| Semantic Router | [semantic-router.md](serving/semantic-router.md) | https://vllm.ai/blog/2025-09-11-semantic-router |
| Iris v0.1 | [semantic-router-iris.md](serving/semantic-router-iris.md) | https://vllm.ai/blog/2026-01-05-vllm-sr-iris |
| vLLM-Omni | [vllm-omni.md](serving/vllm-omni.md) | https://vllm.ai/blog/2025-11-30-vllm-omni |

## Rest of CATALOG (complete; optional)

Dated posts **129/129** have compressed study notes. Full table: [README.md](README.md). Do not read them as another must-read wave. High-signal clusters after the main line:

- Omni / TTS / diffusion RL: `minimax-h3`, `omni-tts`, `omni-diffusion-cache`, `omni-autoround`, `qwen3-omni`, `omni-layerwise-offload`, `verl-omni`, `verl-omni-v020`
- RL alignment: `vime`, `vime-rocm`
- Pareto / hardware: `qwen35-25k-tps`, `glm52-b300`, `hpc-ops`, `gb300-deepseek`, `gpt-oss-optimizations`, `blackwell-inferencemax`, `artificial-analysis`, `eagle3-amd`
- Day-0 / community: Nemotron 3 family, Gemma 4, Llama 3.1/4, gpt-oss getting-started, meetups, playground, vllm.ai website

Semantic Router follow-ons live under `serving/semantic-router-*.md`.
