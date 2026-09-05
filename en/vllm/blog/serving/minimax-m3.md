---
source: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
lang: en
fetched: 2026-09-05
---

# MiniMax M3 in vLLM: Day-0 Serving for 1M-Token Multimodal Reasoning

Chinese: [zh/vllm/blog/serving/minimax-m3.md](../../../../zh/vllm/blog/serving/minimax-m3.md)

2026-06-12. vLLM Team. Weights: [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3), [`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8). EAGLE3 draft: [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3). Recipes: [recipes.vllm.ai/MiniMaxAI/MiniMax-M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). NVIDIA verification: H200, GB200, B300. AMD: MI350 / MI300. MSA source: [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA). vLLM PR [#45381](https://github.com/vllm-project/vllm/pull/45381). NeMo RL: [minimax-m3.md](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md). H3 production serving: [minimax-h3.md](minimax-h3.md). Cousins: [anatomy.md](../core/anatomy.md), [spec-decode.md](../features/spec-decode.md), [kv-offload.md](../features/kv-offload.md). Local figures remain copyright of the original site.

The hard part is not loading the model. It is making MiniMax Sparse Attention, multimodal preprocessing, MXFP8 MoE, EAGLE3, prefix caching, and deployment recipes work together in a serving engine users can actually run. This post walks through the model features, the vLLM implementation, the kernel and cache work, and the optimizations still landing after day 0.

![Figure 1: MiniMax M3 day-0 long-context multimodal sparse-attention serving](../../../../assets/vllm/blog/serving/minimax-m3/01-hero-minimax-m3-vllm.svg)

## TL;DR

- **Model family:** BF16 and MXFP8. 1M-token context subject to hardware capacity and deployment configuration.
- **Core architecture:** MiniMax Sparse Attention (MSA). Scores 128-token KV blocks, selects top blocks per query and KV group, runs GQA over the selected blocks.
- **Serving stack:** `minimax_m3` tool and reasoning parsers, thinking-mode control, text-only and multimodal paths, TP/EP, prefix caching, chunked prefill, EAGLE3, a Docker image available to use.
- **Speculative decoding:** day-0 EAGLE3 with draft `Inferact/MiniMax-M3-EAGLE3`.
- **RL:** day-0 MiniMax M3 GRPO in [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL), vLLM as the generation backend.
- **Performance:** MSA prefill/decode, indexer-score and top-k, fused QKNorm + RoPE + KV insert, GemmaNorm and quantization-path work, MXFP8 MoE backend.
- **Roadmap:** FP8 indexer/KV, TRTLLM-Gen MoE, broader disaggregated serving recipes, context-parallel long-prefill, multimodal gateway.

## MiniMax M3 Support Matrix

| Capability | What MiniMax M3 Adds | vLLM Support |
| --- | --- | --- |
| 1M-token context | Long-context text, code, agent traces, documents | `--max-model-len`, block-size 128 recipes, prefix caching, chunked prefill, MSA kernels |
| MiniMax Sparse Attention | Block-sparse GQA over selected 128-token KV blocks | Hybrid attention backend, indexer-score, top-k, sparse GQA prefill/decode |
| MXFP8 model weights | Efficient MoE serving at scale | DeepGEMM MXFP8 on Blackwell; Marlin MXFP8 on Hopper |
| Native multimodality | Image and video alongside text | Model-specific multimodal preprocessing and serving integration |
| Tool and reasoning | Agentic workflows and controllable thinking | `minimax_m3` parsers, `thinking_mode` chat-template control |
| EAGLE3 | Draft-model acceleration | Day-0 recipe + `Inferact/MiniMax-M3-EAGLE3` |

## Quickstart: Run MiniMax M3 with vLLM

On NVIDIA, MSA uses the default attention backend; the vision encoder uses FlashInfer (`--mm-encoder-attn-backend FLASHINFER`) with a shared-memory processor cache and a data-parallel encoder.

MXFP8 starting point on a Blackwell-class node:

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

For BF16:

```bash
vllm serve MiniMaxAI/MiniMax-M3 \
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

The exact recipe depends on accelerator, dtype, context length, traffic shape, and whether the deployment prioritizes throughput, latency, or maximum context. Verification has been done on NVIDIA H200, GB200, and B300. Full NVIDIA and AMD launch recipes, strategies, and knobs: [vLLM recipe for MiniMax M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3).

### AMD ROCm

MiniMax M3 runs on AMD Instinct. MSA uses the Triton attention backend, so add `--attention-backend TRITON_ATTN`; the vision encoder uses AITER FlashAttention (`--mm-encoder-attn-backend ROCM_AITER_FA`) with a shared-memory processor cache and a data-parallel encoder.

MXFP8:

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

BF16:

```bash
vllm serve MiniMaxAI/MiniMax-M3 \
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

Verification: MI350 Series and MI300 Series.

### Deployment Knobs That Matter

A few knobs matter more than usual. `--block-size 128` aligns vLLM cache blocks with MSA’s sparse granularity. `--max-model-len` controls advertised context and KV capacity planning. `--tensor-parallel-size` and `--enable-expert-parallel` determine how attention, projections, and MoE experts are split. Enable the `minimax_m3` tool and reasoning parsers for agent workloads. Long-context recipes should state whether prefix caching, chunked prefill, EAGLE3, and multimodal preprocessing are on for that target.

### EAGLE3 Speculative Decoding

Day-0 EAGLE3. Draft: `Inferact/MiniMax-M3-EAGLE3`. When traffic and acceptance fit, use the draft-model path for lower generation latency.

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
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

The example uses `num_speculative_tokens=3`, a conservative starting point for validation. Production should tune against acceptance, TPOT, throughput, target latency, and traffic mix.

### Thinking Mode

Controllable thinking. In vLLM, pass the mode through `chat_template_kwargs`:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id

messages = [{"role": "user", "content": "Explain MiniMax Sparse Attention."}]

for mode in ["enabled", "disabled", "adaptive"]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "chat_template_kwargs": {
                "thinking_mode": mode,
            },
        },
    )
    print(mode, response.choices[0].message.content)
```

## Model Key Features and New Capabilities

MiniMax M3 matters for inference systems in three directions.

### 1M-Token Context with MiniMax Sparse Attention

The central architectural change is MSA. Instead of every query attending densely over the full KV cache, an index path scores KV blocks and selects the most relevant ones for the real attention computation. Default granularity is a 128-token KV block; selected blocks are shared across a GQA group.

Every query token follows three steps:

1. Score candidate KV blocks with a small index head.
2. Select the top blocks, applying the configured block rules.
3. Run online-softmax attention over only those selected KV blocks.

That preserves the long-context behavior users expect while bounding attention work per generated token. Practically, MSA is what makes M3’s 1M-token context practical for vLLM serving.

![Figure 2: MSA local + global context, sparse 128-token blocks from 1M history](../../../../assets/vllm/blog/serving/minimax-m3/02-msa-1m-context.svg)

### MSA Mechanics in More Detail

MSA separates two questions: which past blocks are worth reading, and how to run attention over those blocks. The index path answers the first by scoring fixed 128-token KV blocks. The sparse GQA path answers the second by running attention over the selected blocks.

The selected set is not only learned top-k. The M3 config exposes `init_blocks` / `sparse_init_block` and `local_blocks` / `sparse_local_block`; **the current recipe uses `init_blocks=0` and `local_blocks=1`**. In practice the deterministic rule is the local-window block near the query token; the remaining selected blocks come from indexer-scored top-k. Correctness depends on small details: partial final blocks must be masked; causal boundaries inside a block must be respected; local blocks that also rank in the top-k must not be counted twice; batched requests can have different valid block ranges.

### Native Multimodality

M3 is a multimodal model, not a text-only checkpoint with a sidecar. The serving path has to handle image and video, preprocess them into patch tensors, preserve grid metadata, and hand the result to the model without stealing GPU time from generation.

Release work includes model-specific multimodal preprocessing and parser support so users can run text-only, tool-use, reasoning, and multimodal workloads through the same serving surface.

### MXFP8 MoE Weights

The MXFP8 checkpoint is designed for efficient large-scale serving. Validation used DeepGEMM MXFP8 MoE on Blackwell-class systems and Marlin MXFP8 on Hopper-class systems.

## vLLM Implementation

M3 is a hybrid model: some layers route to dense attention, sparse layers to the MiniMax MSA backend. vLLM keeps that distinction behind the model and attention backend, so scheduler, cache allocation, batching, prefix caching, and serving still look familiar from the outside. Companion: [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm); note [anatomy.md](../core/anatomy.md).

### MiniMax Sparse Attention Backend

The MSA backend has two responsibilities.

First, it computes sparse metadata. The indexer scores KV blocks, applies block-selection rules, and emits top-k block IDs. For M3, selection is block-based: the unit of sparsity is the same page-like 128-token block the cache manager already understands.

Second, it computes attention over those blocks. Prefill and decode have different shapes, so specialized kernels:

- **Prefill indexer-score:** Triton computes block scores and top-k.
- **Prefill sparse GQA:** Triton and the [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) CuTe/SM100 path. The CuTe path inverts the query-to-block mapping into a K-major CSR so KV blocks can be reused.
- **Decode indexer-score:** Split-style decode kernels scan candidate blocks, score them, and merge top-k.
- **Decode sparse GQA:** GQA decode kernels consume selected block pages and merge partial attention outputs.

### Prefill Execution

Prefill processes the prompt and creates the KV cache. For M3, prompt length and sparse metadata both matter. Four conceptual stages:

1. **Build Q/K/V and index projections.** Dense projections produce the representations needed by the indexer and attention kernels.
2. **Score blocks.** The index path computes a score for each candidate KV block. The scoring reduction can use block-level rules such as max or log-sum-exp, depending on the model configuration.
3. **Select blocks.** Top-k combines learned scores with configured rules, then emits block IDs for each query and KV group.
4. **Run sparse GQA.** The kernel reads only selected KV blocks and computes the same online-softmax result as dense attention restricted to that selected set.

Two useful schedules for the final sparse GQA. Query-major is straightforward: each query walks its selected KV blocks. KV-block-major is better for long prompts when many queries select the same block. Then vLLM builds a K-to-Q mapping so one KV block can be loaded and reused across many queries before the output merge.

### Decode Execution

Decode has a different shape. Each step usually processes one new token per active sequence, but the batch can contain many sequences with different context lengths. The runtime updates cache state, scores candidate blocks, applies local-window handling, selects top blocks, runs sparse GQA decode, and merges partial outputs if the kernel uses split work. Because this happens **every** generated token, indexer-score and top-k are part of **TPOT**, not setup overhead.

M3’s sparse-attention config controls block size, top-k count, optional init blocks, local-window blocks, index dimension, sparse layer IDs, score type, and layers where index attention is used for selection only. The key implementation rule: every selected block ID must map back to the same logical request state the scheduler and cache manager know about.

![Figure 3: dense layers vs MiniMax MSA backend](../../../../assets/vllm/blog/serving/minimax-m3/03-msa-backend-dispatch.svg)

### KV Cache Layout: Standard Storage, Sparse Computation

M3 can store KV as ordinary paged KV and apply sparsity in the computation path. That keeps the cache manager simple while adding the flexibility the kernels need:

- The main attention KV cache and indexer K cache are tracked **explicitly**.
- Prefix caching and chunked prefill can keep using stable cache blocks once the recipe’s cache-state interactions are validated.
- Related disaggregated-serving and NIXL-style transfer paths can treat the cache as paged state while the attention backend handles sparse selection.

### Prefix Caching and Chunked Prefill

Prefix caching matters because M3 workloads often reuse long prompts: codebases, documents, multi-turn agent traces, multimodal context. Chunked prefill matters because a 1M-token request should not monopolize the engine as one giant prefill. Together they are release-readiness stress tests: index cache, main attention KV, dense attention state, prefix hits, preemption, batching, and long-context chunk boundaries all need to agree on the same block tables before a recipe is production-ready.

### Multimodal and Parser Integration

Model-specific parsing for tools, reasoning, and multimodal input. vLLM support:

- `--tool-call-parser minimax_m3`
- `--reasoning-parser minimax_m3`
- Chat-template support for `thinking_mode`
- Multimodal preprocessing for image and video

For production, preprocessing is best handled before GPU execution. Target architecture: a gateway that downloads media, decodes frames, samples video, resizes and normalizes images, creates patch tensors, and passes **ready-to-run tensors** to the worker.

This matters because multimodal requests can look small at the API boundary and large after preprocessing. One video can require frame sampling, per-frame resizing, patch generation, and metadata packing. Keeping CPU-heavy media work upstream makes GPU scheduling easier to reason about.

The parser side is equally important for agent traffic. Tool-call and reasoning parsers turn model-specific text conventions into structured API responses. Without the right parser, the model can generate useful text that is hard for an application to consume.

![Figure 4: CPU-side image/video preprocessing hands ready tensors to the worker](../../../../assets/vllm/blog/serving/minimax-m3/04-multimodal-request-path.svg)

## Performance Optimizations

M3 shifts the bottlenecks. MSA reduces dense attention work but introduces indexer-score, block selection, sparse metadata, and extra small kernels. The day-0 implementation focuses on keeping those new pieces cheap.

The guiding principle: **do not spend more time deciding which blocks to read than you save by not reading all blocks.** That shows up in three places: block-major prefill, lean decode indexer-score kernels, and fusing small elementwise or cache-write kernels around the attention path.

### KV-Block-Major Prefill

During prefill, many query tokens can select the same KV block. A naive query-major sparse attention kernel would repeatedly move the same KV block from HBM to on-chip memory. The [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) CuTe/SM100 path builds a K-to-Q CSR mapping, runs a block-major sparse attention kernel, and uses a log-sum-exp reduction to combine partial outputs. This improves arithmetic intensity for long prompts and agentic traffic where long cached contexts are common.

![Figure 5: KV-block-major prefill reuses selected KV blocks](../../../../assets/vllm/blog/serving/minimax-m3/05-kv-block-major-prefill.svg)

### Decode Indexer-Score Kernels

In decode, the indexer is on the critical path for every generated token. The engine must compare query-side index vectors against candidate key-side vectors, reduce each 128-token block into a score, apply local-window handling, and keep only the top blocks for sparse GQA.

The optimized decode path uses specialized indexer-score kernels instead of treating the problem as a padded dense GEMM. That avoids extra work around ragged per-request block ranges and keeps the top-k boundary close to the score computation.

The decode path also has to be careful about memory traffic. Selected KV blocks are sparse in logical sequence space but still page-like in memory, so the kernel should avoid turning sparse pages into large temporary dense tensors unless reuse justifies it.

### Speculative Decoding in the Decode Kernels

EAGLE3 also requires the decode kernels to handle speculative verification efficiently. One request can verify multiple draft tokens at once, so the MSA decode kernels **cannot assume** exactly one query token per request.

One fallback is to use prefill kernels for speculative verification, at high cost: prefill kernels are usually tuned for much larger token counts and perform poorly on small draft-token batches. They are also usually incompatible with full CUDA graph mode, an important optimization for low-latency decode.

The day-0 implementation updates the MSA decode indexer, top-k selection, and sparse GQA decode kernels to support a uniform `decode_query_len`. The kernels flatten speculative verification tokens in request-major order, then map each query token back to the correct request metadata, sequence length, block table, and causal position. EAGLE3 verification uses the decode-specialized split-K path instead of a less targeted prefill-style path, while keeping the speculative path close to the existing decode implementation.

The same path supports **full CUDA graph** coverage for uniform speculative decode batches. Launch grids stay shape-stable; selected arguments avoid unnecessary Triton specialization; padded request rows are handled explicitly so captured graphs can be replayed safely. These details matter: speculative decoding only improves TPOT when draft-token acceptance is not offset by extra kernel launches, recompiles, or cache-state overhead. The page expects to keep optimizing across draft lengths, concurrency, and traffic mixes.

### Kernel Fusions

Several smaller kernels were fused or routed through custom ops to reduce launch overhead and HBM round trips:

- **QKNorm + RoPE + KV insert:** normalization, position encoding, and cache write for the MSA path.
- **GemmaNorm and AllReduce + Norm:** reduces overhead around normalization in tensor-parallel execution.
- **Quantization-path cleanup:** improves `silu_mul_quant_fp8` and related MXFP8/MoE input paths.
- **Router and MoE kernels:** reduce overhead in the sparse expert path and prepare for deeper TRTLLM-Gen integration.

The release path is intentionally conservative: correctness and stable cache behavior win over enabling every graph or fusion knob on day 0. More aggressive fusions can land as public recipes mature.

### Quantization and KV Cache Dtype

The MXFP8 checkpoint primarily changes weight and MoE execution, not the conceptual structure of the KV cache. Public recipes should state model dtype, MoE backend, and KV-cache policy **separately**: “MXFP8 model” does not automatically mean every cache and intermediate tensor is MXFP8. The roadmap includes FP8 indexer and KV-cache paths because KV capacity directly controls how much long-context and batched traffic a deployment can serve.

### CUDA Graphs and Compile Behavior

CUDA graphs are valuable for decode because M3 introduces several small operations around each token step. Graph capture only helps when the captured path is **stable** across batch shapes, cache states, and sparse metadata. The day-0 path uses conservative graph settings where needed, then expands coverage as validation matures.

## Validation

Before the public release, the vLLM team ran daily validation across accuracy, throughput, speculative decoding, and container usability.

Three goals:

1. **Functional correctness:** the model loads, serves, parses tool and reasoning outputs, and handles text-only plus multimodal inputs.
2. **Accuracy parity:** benchmarks stay aligned with expected model behavior after kernel, cache, parser, and recipe changes.
3. **Serving readiness:** containers run with the intended TP/EP/speculative-decoding settings on target accelerators.

The most useful tests combine short correctness tasks with long-output and long-context workloads. Short tasks catch parser, formatting, and obvious numerical issues quickly. Long-context tasks catch MSA metadata, prefix caching, chunked prefill, and KV-cache layout problems. Speculative decoding tests catch acceptance regressions that may not show up in ordinary accuracy runs.

A representative snapshot on B300:

| Dimension | Result |
| --- | ---: |
| GSM8K strict / flexible | 91.51% / 91.66% |
| ShareGPT @256 throughput | 8,530 tok/s |
| ShareGPT @256 TPOT | 56.0 ms |
| Speculative Sonnet TPOT, concurrency 1 / 16 / 64 | 4.51 / 9.04 / 14.36 ms |
| Speculative acceptance on Sonnet | ~67%, mean accept length ~3.0 |

Engineering-validation measurements, not an official ranking. Exact results vary with image, weights, recipe, and hardware.

![Figure 6: release-candidate validation dashboard](../../../../assets/vllm/blog/serving/minimax-m3/06-validation-dashboard.svg)

## Beyond Serving: RL Post-Training with NeMo RL

Day-0 is not only inference serving. RL frameworks use vLLM as the generation engine that produces rollouts inside the training loop, so the same MiniMax M3 work that powers serving in [vLLM PR #45381](https://github.com/vllm-project/vllm/pull/45381) also makes M3 post-training possible on day 0.

[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) now runs MiniMax M3 with vLLM as a **non-colocated** generation backend. Short GRPO (Group Relative Policy Optimization) post-training runs have been validated on the BF16 checkpoint, using NeMo AutoModel with expert parallelism and BF16 vLLM generation. Long-run convergence and parallelism strategies **beyond expert parallel** are still being validated. Early results show what a solid serving path is worth: the engine that serves M3 is also the one that drives the rollout phase of RL training. Reference: [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md).

## Roadmap: The Path Ahead

Day-0 is the starting line. Next pieces already in flight:

- **FP8 indexer and KV-cache:** reduce KV memory pressure and increase batch capacity while preserving sparse-attention accuracy.
- **TRTLLM-Gen MoE:** improve Blackwell performance for MXFP8 expert execution.
- **Context parallelism:** improve very-long-context prefill scaling when one node is not enough.
- **Disaggregated serving:** expand NIXL and P/D recipes for M3 traffic, building on [Large-Scale Serving](https://vllm.ai/blog/2025-12-17-large-scale-serving).
- **Kernel fusion:** reduce the many small indexer, top-k, quantization, and normalization kernels MSA introduces.
- **Multimodal gateway:** keep image and video preprocessing out of the critical GPU generation loop.

## MiniMax M3 vLLM FAQ

### Does vLLM support MiniMax M3?

Yes. This post covers day-0 support for the BF16 and MXFP8 checkpoints, including MSA, model-specific parsers, EAGLE3, multimodal preprocessing, TP/EP recipes, and a Docker image available to use.

### What is MiniMax Sparse Attention?

MSA scores fixed 128-token KV blocks, selects the most relevant blocks for each query and GQA group, applies the configured local-window rule, and runs sparse GQA over that selected set. In the current M3 recipe: `init_blocks=0` and `local_blocks=1`.

### Does MXFP8 mean the KV cache is MXFP8?

**No.** MXFP8 describes the model weight and MoE execution path. KV-cache dtype is a separate serving decision; current sparse-attention validation treats native KV storage and quantized KV-cache support as separate roadmap work.

### What settings matter most for 1M-token context?

Starting points: `--block-size 128`, enough GPU memory for the chosen batch and context shape, and a recipe that states whether prefix caching, chunked prefill, and EAGLE3 are enabled. By default vLLM reads context length from the model config, so you **do not** need `--max-model-len`. If GPU memory is limited or you do not need the full 1M window, pass `--max-model-len` to cap it lower and reduce KV pressure.

## Acknowledgments

Thanks to the MiniMax team for open-sourcing MiniMax-M3, and to MiniMax leadership for their trust and support in vLLM. Model support is led by **Inferact Inc.**, aiming to grow vLLM as the world’s AI inference engine and make inference cheaper and faster. NVIDIA and AMD contributed hardware support.

## Related vLLM Reading

- [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) for scheduler, KV cache, prefix caching, distributed execution. Note: [anatomy.md](../core/anatomy.md).
- [Speculative Decoding](https://vllm.ai/blog/2024-10-17-spec-decode) and [P-EAGLE](https://vllm.ai/blog/2026-03-13-p-eagle). Note: [spec-decode.md](../features/spec-decode.md).
- [Large-Scale Serving](https://vllm.ai/blog/2025-12-17-large-scale-serving), [KV Offloading Connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector), [Moriio KV Connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector). Note: [kv-offload.md](../features/kv-offload.md).
- [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md).
