---
source: https://docs.vllm.ai/en/stable/configuration/optimization/
lang: en
fetched: 2026-08-31
---

# Knob → blog map

The optimization page is the polite order of knobs. The blogs are how those knobs grew. Full reading order: [MUST-READ.md](MUST-READ.md). Docs: [optimization.md](../optimization/optimization.md), [serve.md](../getting-started/serve.md).

| Knob | Docs section | Blog first | One line |
|---|---|---|---|
| KV blocks / PagedAttention | preemption, cache size | [launch](architecture/paged-attention.md) → [Anatomy](architecture/anatomy.md) | The house is KV, not weights. |
| `-O0`…`-O3`, CUDA graphs, `torch.compile` | optimization level; `--enforce-eager` | [torch.compile](architecture/torch-compile.md), [v0.6](performance/v0.6-throughput.md), [V1](architecture/v1-alpha.md), [MRV2](architecture/mrv2.md) | Startup vs decode; fusion happens at compile time. |
| `max_num_batched_tokens`, chunked prefill | Chunked Prefill | [vs DeepSpeed](architecture/vs-deepspeed.md), Anatomy | 2023 name: SplitFuse. Small value guards ITL; large guards TTFT; throughput often wants >8192. |
| prefix cache | features | Anatomy, [production-stack](serving/production-stack.md), [Router](serving/router.md), [Mooncake](serving/mooncake.md), [KV offload](serving/kv-offload.md) | Local hit → sticky session; preempt → CPU; miss → pool. |
| speculative decoding | features | [spec-decode](performance/spec-decode.md) | Draft/verify. “Not yet supported” is historical. |
| FP8 KV / attention quant | memory / precision | [FP8 KV](performance/fp8-kvcache.md), [TurboQuant](performance/turboquant.md) | Default FP8; 3–4-bit storage pays dequant. |
| `gpu_memory_utilization`, preemption | Preemption | Anatomy, launch | V1 default RECOMPUTE. Frequent preempt → give KV rooms. |
| TP / PP | parallelism | [distributed](serving/distributed-inference.md); TRT-LLM sharding chapter | TP in-node, PP between; do not naive-TP MLA. |
| EP / DP / `--enable-expert-parallel` | Expert / Data Parallelism | [Wide-EP](serving/large-scale.md) | Dense → DP Attention; sparse → EP. |
| `--enable-dbo`, EPLB | (blog-first) | Wide-EP | Overlap microbatches when EP comm is fat; reshuffle hot experts. |
| `--enable-elastic-ep` | (blog) | [Elastic EP](serving/elastic-ep.md) | Resize DP at runtime. Then: TP=1, no DBO, Ray only. |
| Text P/D | deploy | [Router](serving/router.md), Wide-EP | One fat prefill can stall the EP combine. |
| `mm_encoder_tp_mode="data"`, MM caches | encoder DP; multimodal cache | [EPD](serving/epd.md) | Single-node batch-split the ViT; cluster moves it to another building. |
| KVConnector / external KV | (blog) | Mooncake, [KV offload](serving/kv-offload.md), [MORI-IO](serving/moriio.md), [PegaFlow](serving/pegaflow.md), production-stack | Same door: DRAM, cluster pool, in-node RDMA, standalone Rust daemon. |
| `--api-server-count`, CPU cores | API scale-out; CPU | v0.6, Anatomy | V1 is multiprocess; starved CPU looks like idle GPU. |
| Ship quality | CI | [production CI](performance/production-quality.md) | Nightly benches, many accelerators, two-week trains. |
| `--enable-sleep-mode` | (blog) | [Sleep Mode](architecture/sleep-mode.md) | Swap models without killing the process; L1→CPU, L2 drop weights. |
| guided / structured decoding | sampling | [structured decoding](performance/struct-decode.md) | Schema as logit masks; JSON / tool-call fence. |
| `-dcp` / `--decode-context-parallel-size` | parallelism; serve CLI | [DCP](performance/dcp.md) | Shard decode KV by sequence; MLA vs GQA constraints. |
| `--kv_offloading_*` | next layer after preemption | [KV offload](serving/kv-offload.md) | Async CPU offload instead of RECOMPUTE. |
| Hybrid SSM P/D | (blog) | [Hybrid SSM](serving/hybrid-ssm.md) | Two NIXL descriptor views on one tensor. |
| AFD / Attention-FFN split | (plugin) | [AFD](serving/afd.md) | Split MoE layer services; experimental; ratio decides. |
| Single-node P/D | (blog) | [MORI-IO](serving/moriio.md) | Split inside one 8-GPU box; write mode wins TTFT. |
| `mm_processor_cache_type="shm"` | multimodal cache | [SHM IPC](serving/shm-ipc.md) | Big images in shared memory, not recopied over IPC. |
| Hardware / platform plugins | (blog) | [plugins](architecture/plugin-system.md), [hardware plugin](architecture/hardware-plugin.md) | Custom scheduler or a new accelerator without a fork. |
| Attention backend | auto-select | [Triton attention](architecture/triton-attn.md) | ROCm default; portable fallback when FA is missing. |
| `turboquant_*` | `--kv-cache-dtype` | [TurboQuant](performance/turboquant.md) | Read the bake-off; production default stays FP8. |
| Weight sync / `pause keep` | (RL) | [Native RL](serving/native-rl.md) | Stop patching workers per framework; two-phase DPEP pause. |
| `ray symmetric-run` | multi-node launch | [Ray symmetric-run](serving/ray-symmetric.md) | Same command on every SLURM / mpssh node. |
| `method: extract_hidden_states` | spec config | [hidden extract](architecture/extract-hidden-states.md) | Dummy draft + KV Connector; prompt only. |
| `"parallel_drafting": true` / `dflash` / `dspark` | spec | [P-EAGLE](performance/p-eagle.md), [parallel drafting](performance/parallel-drafting.md), [AMD spec](performance/spec-decode-amd.md) | K tokens in one forward; larger N is not always faster. |
| `enable_adaptive_verification` | spec | [DSpark adaptive](performance/dspark-adaptive.md) | Budget = confidence × load; needs FULL varlen decode graphs. |
| `/v1/realtime`, StreamingInput | (blog) | [streaming](serving/streaming-realtime.md) | Model must be causal; chunked prefill is a different knob. |
| `--kv-cache-dtype fp8` + `--block-size 256` (V4) | memory | [DeepSeek V4](architecture/deepseek-v4.md) | Logical block = 256 native positions; compressor residual as SWA. |
| `--quantization ptpc_fp8` | ROCm quant | [PTPC-FP8](performance/ptpc-fp8.md) | Weight quant, not KV dtype. |
| `model_impl="transformers"` | model impl | [Transformers backend](architecture/transformers-backend.md) | Coverage, not the performance default. |
| `return_token_ids` | OpenAI API | [Agent Lightning](serving/agent-lightning.md) | Agent RL must not retokenize. |
| RDT / `sharded_rdt` | RL weight sync | [RDT](serving/rdt-weight-transfer.md) | Buffers outside `gpu_memory_utilization`; no EPLB then. |
| Routing / control plane | (not in optimization) | production-stack / [AIBrix](serving/aibrix.md) / Router | The rack on top of the engine can change; KV affinity cannot. |
| `--async-scheduling`, `--stream-interval` | API / scheduler | [gpt-oss opts](performance/gpt-oss-optimizations.md), [Qwen3.5 25K](serving/qwen35-25k-tps.md) | Hide CPU; stream-interval buffers later tokens and hurts ITL. Hybrid P/D needs the race fixes first. |
| `--gdn-prefill-backend`, `VLLM_SSM_CONV_STATE_LAYOUT=DS` | hybrid / P/D | [Qwen3.5 25K](serving/qwen35-25k-tps.md), [Hybrid SSM](serving/hybrid-ssm.md) | GDN/Mamba state is not the same transfer as KV. |
| `--language-model-only` | multimodal | Qwen3.5 25K | Text-only unlocks fused QK-norm+RoPE. |
| `--enable-distributed-layerwise-offload` | Omni | [DLO](serving/omni-layerwise-offload.md) | Stream DiT weights layerwise; two layers double-buffered on device. |
| `--omni`, `cache_backend` | Omni | [Omni](serving/vllm-omni.md), [diffusion cache](serving/omni-diffusion-cache.md), [TTS](serving/omni-tts.md) | Text TTFT ≠ audio TTFP; cache eats timestep redundancy. |
| `--attention-backend HPC_ATTN`, `--moe-backend hpc` | backend | [HPC-Ops](performance/hpc-ops.md) | Then Hy3 / FP8 / Hopper, not a universal default. |
| `VLLM_USE_V2_MODEL_RUNNER=1` | MoE runtime | [GLM-5.2 SLA](serving/glm52-b300.md), [MRV2](architecture/mrv2.md) | Dense already defaults V2; MoE must opt in. |
| `--block-size 128` (MSA) | long context | [MiniMax M3](serving/minimax-m3.md) | Matches sparse 128-token blocks; not an arbitrary cache size. |

NVIDIA sibling map: TensorRT-LLM sharding chapter. Official CLI once wrote `--tp_size` twice; PP is `--pp_size`.
