---
source: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
lang: en
fetched: 2026-09-04
---

# MiniMax M3 in vLLM: Day-0 Serving for 1M-Token Multimodal Reasoning

Chinese: [zh/vllm/blog/serving/minimax-m3.md](../../../../zh/vllm/blog/serving/minimax-m3.md)

2026-06-12. **vLLM Team**. Checkpoints: [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) (BF16), [`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8). Verified H200 / GB200 / B300; AMD MI350 / MI300. Earlier Lightning-Attention cousin: [minimax-m1.md](minimax-m1.md). Later Omni stack: [minimax-h3.md](minimax-h3.md). Spec path: [spec-decode.md](../performance/spec-decode.md), [p-eagle.md](../performance/p-eagle.md). Cache / P/D cousins: [large-scale.md](large-scale.md), [kv-offload.md](kv-offload.md), [mooncake.md](mooncake.md), [shm-ipc.md](shm-ipc.md). Study note; recipes live on [recipes.vllm.ai](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). **Not a new engine** — MSA backend, parsers, MXFP8 MoE, EAGLE3 recipe.

MiniMax M3 is built for workloads that are becoming normal: million-token context, native multimodal reasoning, coding and agentic workflows, tool use, controllable thinking. The hard part is not loading weights. It is making MiniMax Sparse Attention, multimodal preprocessing, MXFP8 MoE, EAGLE3, prefix caching, and deployment recipes work together in an engine people can actually run.

Local figures (copyright remains with the original site; study copies):

![hero minimax m3 vllm](../../../../assets/vllm/blog/serving/minimax-m3/01-hero-minimax-m3-vllm.svg)

![msa 1m context](../../../../assets/vllm/blog/serving/minimax-m3/02-msa-1m-context.svg)

![msa backend dispatch](../../../../assets/vllm/blog/serving/minimax-m3/03-msa-backend-dispatch.svg)

![multimodal request path](../../../../assets/vllm/blog/serving/minimax-m3/04-multimodal-request-path.svg)

![kv block major prefill](../../../../assets/vllm/blog/serving/minimax-m3/05-kv-block-major-prefill.svg)

![validation dashboard](../../../../assets/vllm/blog/serving/minimax-m3/06-validation-dashboard.svg)

**Figure 1.** Day-0 support: long-context, multimodal, sparse-attention serving in vLLM.

## TL;DR

- **Model family:** BF16 and MXFP8 MiniMax M3; 1M-token context subject to hardware and recipe.
- **Core architecture:** MiniMax Sparse Attention (MSA) — hybrid dense/sparse. Scores 128-token KV blocks, picks top per query and KV group, then GQA over the selected blocks.
- **Serving stack:** `minimax_m3` tool and reasoning parsers, thinking-mode control, text-only and multimodal paths, TP/EP, prefix caching, chunked Prefill, EAGLE3, Docker image.
- **Speculative decoding:** Day-0 EAGLE3 with [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3).
- **RL post-training:** Day-0 MiniMax M3 GRPO in [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL), vLLM as generation backend.
- **Performance work:** MSA Prefill/Decode kernels, indexer-score and top-k, fused QKNorm + RoPE + KV insert, GemmaNorm and quantization-path work, MXFP8 MoE backends.
- **Roadmap:** FP8 indexer/KV-cache, TRTLLM-Gen MoE, broader disaggregated recipes, context-parallel long-Prefill, multimodal gateway.

## MiniMax M3 Support Matrix

| Capability | What MiniMax M3 Adds | vLLM Support |
| --- | --- | --- |
| 1M-token context | Long-context text, code, agent traces, documents | `--max-model-len`, block-size 128 recipes, prefix caching, chunked Prefill, MSA kernels |
| MiniMax Sparse Attention | Block-sparse GQA over selected 128-token KV blocks | Hybrid attention backend, indexer-score kernels, top-k block selection, sparse GQA Prefill/Decode |
| MXFP8 model weights | Efficient MoE serving | DeepGEMM MXFP8 MoE on Blackwell-class; Marlin MXFP8 on Hopper-class |
| Native multimodality | Image and video with text | Model-specific multimodal preprocessing |
| Tool and reasoning outputs | Agentic workflows, controllable thinking | `minimax_m3` tool parser, `minimax_m3` reasoning parser, `thinking_mode` chat-template control |
| EAGLE3 speculative decoding | Draft-model acceleration | Day-0 EAGLE3 recipe with Inferact draft |

## Quickstart: Run MiniMax M3 with vLLM

On NVIDIA, MSA uses the **default** attention backend. Vision encoder: `--mm-encoder-attn-backend FLASHINFER`, shared-memory processor cache, data-parallel encoder.

MXFP8 on a Blackwell-class node:

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

BF16: same flags, `MiniMaxAI/MiniMax-M3`. Exact recipe depends on accelerator, dtype, context length, traffic shape, and whether you optimize for throughput, latency, or max context. Full NVIDIA and AMD recipes: [vLLM recipe for MiniMax M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3).

### AMD ROCm

MSA on Triton: `--attention-backend TRITON_ATTN`. Vision: `--mm-encoder-attn-backend ROCM_AITER_FA`, shm processor cache, data-parallel encoder. Verified on MI350 Series and MI300 Series.

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --attention-backend TRITON_ATTN \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend ROCM_AITER_FA \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

BF16: same flags, `MiniMaxAI/MiniMax-M3`.

### Deployment Knobs That Matter

`--block-size 128` must match MSA’s sparse grain. `--max-model-len` is advertised context and KV planning. `--tensor-parallel-size` and `--enable-expert-parallel` split attention, projections, and MoE experts. Enable the `minimax_m3` parsers for agent traffic. Long-context recipes should state whether prefix caching, chunked Prefill, EAGLE3, and multimodal preprocessing are on for that target.

### EAGLE3 Speculative Decoding

Draft: [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3). Add:

```bash
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

`num_speculative_tokens=3` is a conservative starting point. Production should tune against acceptance rate, TPOT, throughput, and target latency.

### Thinking Mode

Pass `thinking_mode` through `chat_template_kwargs`: `"enabled"`, `"disabled"`, or `"adaptive"`.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id
messages = [{"role": "user", "content": "Explain MiniMax Sparse Attention."}]

for mode in ["enabled", "disabled", "adaptive"]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"chat_template_kwargs": {"thinking_mode": mode}},
    )
    print(mode, response.choices[0].message.content)
```

## Model Key Features and New Capabilities

### 1M-Token Context with MiniMax Sparse Attention

Instead of every query attending densely over the full KV cache, MSA uses an index path to score KV blocks and select the most relevant ones. Default grain: **128-token** KV block. Selected blocks are shared across a GQA group.

Every query token:

1. Score candidate KV blocks with a small index head.
2. Select the top blocks, applying configured block rules.
3. Run online-softmax attention over only those selected KV blocks.

That is what makes 1M-token context practical for serving.

**Figure 2.** MSA keeps local and global context available while selecting sparse 128-token KV blocks from a 1M-token history.

### MSA Mechanics in More Detail

Two questions: which past blocks are worth reading, and how to run attention over them. Index path answers the first (fixed 128-token blocks). Sparse GQA answers the second.

The selected set is not only learned top-k. Config exposes `init_blocks` / `sparse_init_block` and `local_blocks` / `sparse_local_block`. Current recipe: **`init_blocks=0`**, **`local_blocks=1`**. Deterministic rule: the local-window block near the query; remaining slots from indexer-scored top-k. Correctness details: mask partial final blocks; respect causal boundaries inside a block; do not double-count a local block that also ranks in top-k; batched requests can have different valid block ranges.

### Native Multimodality

Not a text-only checkpoint with a sidecar. Image and video → patch tensors, grid metadata, handed to the model without stealing GPU time from generation. Same serving surface for text-only, tool-use, reasoning, and multimodal.

### MXFP8 MoE Weights

Validation: DeepGEMM MXFP8 MoE on Blackwell-class; Marlin MXFP8 on Hopper-class.

## vLLM Implementation

Hybrid model: some layers dense attention, sparse layers the MiniMax MSA backend. Distinction stays behind the model and attention backend — scheduler, cache allocation, batching, prefix caching, serving look familiar. Companion: [Anatomy of vLLM](../architecture/anatomy.md).

### MiniMax Sparse Attention Backend

Two jobs.

First, sparse metadata: indexer scores KV blocks, applies selection rules, emits top-k block IDs. For M3, the unit of sparsity is the same page-like 128-token block the cache manager already understands.

Second, attention over those blocks. Prefill and Decode have different shapes:

- **Prefill indexer-score:** Triton block scores and top-k.
- **Prefill sparse GQA:** Triton and the [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) CuTe/SM100 path. CuTe inverts query-to-block into a K-major CSR so KV blocks reuse efficiently.
- **Decode indexer-score:** Split-style kernels scan candidates, score, merge top-k.
- **Decode sparse GQA:** GQA Decode kernels consume selected pages and merge partials.

### Prefill Execution

Four conceptual stages: (1) build Q, K, V, and index projections; (2) score blocks (max or log-sum-exp per config); (3) select top-k plus configured rules; (4) sparse GQA over selected KV only.

Two schedules for the last step. Query-major: each query walks its selected blocks. KV-block-major: better for long prompts when many queries pick the same block — build a K-to-Q mapping, load one KV block, reuse across queries, then merge.

### Decode Execution

Usually one new token per active sequence; the batch can mix many context lengths. Update cache, score candidates, local-window handling, select top blocks, sparse GQA Decode, merge split work. Indexer-score and top-k sit on **TPOT**, not setup.

Config controls block size, top-k, optional init blocks, local-window blocks, index dim, sparse layer IDs, score type, and layers where index attention is selection-only. Every selected block ID must map back to the same logical request state the scheduler and cache manager know.

**Figure 3.** Dense layers through standard attention; sparse layers through the MiniMax MSA backend.

### KV Cache Layout: Standard Storage, Sparse Computation

KV can be ordinary paged KV; sparsity is in the compute path. Cache manager stays simple:

- Main attention KV and indexer K cache tracked explicitly.
- Prefix caching and chunked Prefill keep using stable cache blocks once cache-state interactions are validated.
- Disaggregated / NIXL-style transfer can treat cache as paged state; the attention backend handles sparse selection.

### Prefix Caching and Chunked Prefill

M3 workloads reuse long prompts: codebases, documents, multi-turn agent traces, multimodal context. A 1M-token request should not monopolize the engine as one giant Prefill. Release-readiness stress: index cache, main attention KV, dense attention state, prefix hits, preemption, batching, and long-context chunk boundaries must agree on the same block tables.

### Multimodal and Parser Integration

- `--tool-call-parser minimax_m3`
- `--reasoning-parser minimax_m3`
- Chat template `thinking_mode`
- Image and video preprocessing

Production: preprocess **before** GPU execution when possible. Target: a gateway that downloads media, decodes frames, samples video, resizes/normalizes, creates patch tensors, passes ready tensors to the worker. One video can look small at the API and large after sampling and patches. CPU-heavy media work upstream keeps GPU scheduling readable.

Parsers turn model-specific text into structured API responses. Wrong parser → useful text the application cannot consume.

**Figure 4.** CPU-side image/video preprocessing should hand ready tensors to the worker so GPU time is reserved for inference.

## Performance Optimizations

MSA cuts dense attention work, then adds indexer-score, block selection, sparse metadata, extra small kernels. Guiding principle: do not spend more time deciding which blocks to read than you save by not reading all of them. Three places: block-major Prefill, lean Decode indexer-score, fusing small elementwise / cache-write kernels around attention.

### KV-Block-Major Prefill

Many query tokens can select the same KV block. Query-major would move that block from HBM to on-chip over and over. CuTe/SM100 path: K-to-Q CSR, block-major sparse attention, log-sum-exp merge of partials. Better arithmetic intensity for long prompts and agentic traffic with long cached contexts.

**Figure 5.** KV-block-major Prefill reuses selected KV blocks across queries before the final LSE reduction.

### Decode Indexer-Score Kernels

Indexer is on the critical path every generated token: compare query-side index vectors to candidate key-side vectors, reduce each 128-token block to a score, local-window, keep top blocks. Specialized kernels instead of a padded dense GEMM. Selected KV is sparse in logical sequence space but still page-like in memory — avoid turning sparse pages into large temporary dense tensors unless reuse justifies it.

### Speculative Decoding in the Decode Kernels

EAGLE3 verification: one request can verify multiple draft tokens, so MSA Decode cannot assume exactly one query token per request.

Falling back to Prefill kernels for verification is expensive: Prefill kernels are tuned for much larger token counts and usually **not** compatible with full CUDA Graph mode.

Day-0 updates the MSA Decode indexer, top-k, and sparse GQA Decode kernels for a uniform `decode_query_len`. Flatten speculative verification tokens in request-major order; map each query token back to request metadata, sequence length, block table, causal position. EAGLE3 verification stays on the Decode-specialized split-K path. Same path supports full CUDA Graph coverage for uniform speculative Decode batches: shape-stable launch grids, fewer Triton specializations, explicit padded-row handling. Speculative decoding only improves TPOT when acceptance is not eaten by extra launches, recompiles, or cache-state overhead.

### Kernel Fusions

- **QKNorm + RoPE + KV insert** for the MSA path.
- **GemmaNorm and AllReduce + Norm** around tensor-parallel execution.
- **Quantization-path cleanup:** `silu_mul_quant_fp8` and related MXFP8/MoE input paths.
- **Router and MoE kernels** toward deeper TRTLLM-Gen integration.

Day-0 is conservative: correctness and stable cache beat enabling every graph or fusion knob.

### Quantization and KV Cache Dtype

MXFP8 changes weight and MoE execution, not the conceptual KV layout. “MXFP8 model” does **not** mean every cache and intermediate is MXFP8. Roadmap includes FP8 indexer and KV-cache paths because KV capacity controls how much long-context and batched traffic a box can serve.

### CUDA Graphs and Compile Behavior

CUDA Graphs help Decode because M3 adds several small ops per token. Capture only helps when the path is stable across batch shapes, cache states, and sparse metadata. Conservative graph settings first; expand coverage as validation matures.

## Validation

Daily loop before public release: accuracy, throughput, speculative decoding, container usability.

1. **Functional correctness:** load, serve, parse tool and reasoning, text-only plus multimodal.
2. **Accuracy parity:** benchmarks stay aligned after kernel, cache, parser, and recipe changes.
3. **Serving readiness:** containers run with intended TP/EP/speculative-decoding settings.

Short tasks catch parser/formatting/numerical issues. Long-context tasks catch MSA metadata, prefix caching, chunked Prefill, KV layout. Speculative tests catch acceptance regressions that ordinary accuracy runs miss.

Representative B300 snapshot (engineering validation, not a ranking; varies with image, weights, recipe, hardware):

| Dimension | Result |
| --- | ---: |
| GSM8K strict / flexible | 91.51% / 91.66% |
| ShareGPT @256 throughput | 8,530 tok/s |
| ShareGPT @256 TPOT | 56.0 ms |
| Speculative Sonnet TPOT, concurrency 1 / 16 / 64 | 4.51 / 9.04 / 14.36 ms |
| Speculative acceptance on Sonnet | ~67%, mean accept length ~3.0 |

**Figure 6.** Release-candidate validation: accuracy, throughput, speculative decoding.

## Beyond Serving: RL Post-Training with NeMo RL

The same MiniMax M3 work that powers serving in [vLLM PR #45381](https://github.com/vllm-project/vllm/pull/45381) also makes M3 post-training possible on day 0.

[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) runs MiniMax M3 with vLLM as a **non-colocated** generation backend. Short GRPO runs validated on the BF16 checkpoint: NeMo AutoModel with expert parallelism and BF16 vLLM generation. Long-run convergence and parallelism beyond EP still being validated. Reference: [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md).

## Roadmap: The Path Ahead

- **FP8 indexer and KV-cache** — KV memory pressure, batch capacity, sparse-attention accuracy.
- **TRTLLM-Gen MoE** — Blackwell MXFP8 expert execution.
- **Context parallelism** — very-long-context Prefill when one node is not enough.
- **Disaggregated serving** — NIXL and Prefill/Decode recipes; see [Large-Scale Serving](large-scale.md).
- **Kernel fusion** — indexer, top-k, quantization, normalization kernels MSA introduces.
- **Multimodal gateway** — keep image/video preprocessing off the GPU generation loop.

## MiniMax M3 vLLM FAQ

### Does vLLM support MiniMax M3?

Yes. Day-0 for BF16 and MXFP8: MSA, model-specific parsers, EAGLE3, multimodal preprocessing, TP/EP recipes, Docker image.

### What is MiniMax Sparse Attention?

Scores fixed 128-token KV blocks, selects the most relevant blocks per query and GQA group, applies the configured local-window rule, runs sparse GQA over that set. Current recipe: `init_blocks=0`, `local_blocks=1`.

### Does MXFP8 mean the KV cache is MXFP8?

No. MXFP8 is the weight and MoE path. KV-cache dtype is a separate serving decision. Native KV storage vs quantized KV-cache is roadmap work.

### What settings matter most for 1M-token context?

`--block-size 128`, enough GPU memory for the chosen batch and context, and a recipe that states whether prefix caching, chunked Prefill, and EAGLE3 are on. By default vLLM reads context length from the model config — you do not need `--max-model-len`. Cap it lower if you have limited GPU memory or do not need the full 1M window.

## Acknowledgments

MiniMax team for open-sourcing MiniMax-M3; MiniMax leadership for trust in vLLM. Model support led by Inferact Inc. NVIDIA and AMD contributed hardware support.

## Related vLLM Reading

- [Anatomy of vLLM](../architecture/anatomy.md) — scheduler, KV cache, prefix caching, distributed execution.
- [Speculative Decoding](../performance/spec-decode.md) and [P-EAGLE](../performance/p-eagle.md) — draft-model path.
- [Large-Scale Serving](large-scale.md), [KV Offloading Connector](kv-offload.md), [Moriio](moriio.md) — prefix reuse, KV movement, disaggregated serving.
- [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md) — GRPO with vLLM generation.
