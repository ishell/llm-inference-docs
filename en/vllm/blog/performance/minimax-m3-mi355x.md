---
source: https://vllm.ai/blog/2026-09-10-minimax-m3-mi355x
lang: en
fetched: 2026-09-11
---

# Following the Bottleneck: Optimizing MiniMax M3 on AMD Instinct MI355X

Chinese: [zh/vllm/blog/performance/minimax-m3-mi355x.md](../../../../zh/vllm/blog/performance/minimax-m3-mi355x.md)

2026-09-10. **AMD and Embedded LLM Teams**. Study extract, not an official reprint. Follow-up to day-0 [minimax-m3.md](../serving/minimax-m3.md). Public numbers: [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/inference). AgentX sibling: [agentx.md](../serving/agentx.md). Spec / MSA: [spec-decode-amd.md](spec-decode-amd.md), [spec-decode.md](spec-decode.md). P/D KV: [moriio.md](../serving/moriio.md). Page logos not copied.

Day-0 described the first working path: MiniMax Sparse Attention (MSA), multimodal inputs, reasoning and tool outputs, MXFP8 weights, and EAGLE3 on AMD Instinct MI355X. This post is what happened after the model ran. The useful result is not only a higher throughput number. It is a way to decide what to optimize next when the bottleneck keeps moving.

## Results in one minute

Public InferenceX numbers for MiniMax-M3 on MI355X:

- Concurrency 32, fixed-topology MXFP8 standard serving: **109.1 → 342.4** output tokens/s/GPU, **3.14×** the day-0 result. Median TTFT **1.46 → 0.67** s. Mean TPOT **69.1 → 22.1** ms.
- Concurrency 128, same TP4/EP1 four-GPU path: **297.8 → 623.7** output tokens/s/GPU, **2.09×**. Median TTFT **3.53 → 1.54** s. Mean TPOT **100.7 → 48.8** ms.
- MXFP4 first rose from **212.1 → 716.8** output tokens/s/GPU at concurrency 128 under the same TP4/EP1 four-GPU contract. A later TP2/EP1 result reached **943.5** output tokens/s/GPU, **31.6%** above that TP4 checkpoint and **4.45×** the initial per-GPU result.
- EAGLE3 speculative decoding: **682.4** output tokens/s/GPU at concurrency 128 on TP4/EP1.
- P/D disaggregation, retuned prefill/decode topology: **6,370.5** total tokens/s/GPU at concurrency 512 with **1.32** s median TTFT.

![hero](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/01-hero-fixed-workload-progress.svg)

**Figure 1.** Standard decoding at 8K input and 1K output. Compare points within a row. MXFP8 stays on TP4/EP1; the last MXFP4 point moves from TP4 to TP2, so it shows higher deployment density, not a fixed-topology speedup.

Each checkpoint is cumulative and can bundle several changes. Isolated PR measurements below explain individual optimizations.

## One decode step, five questions

Estimate dominant costs, measure them, batch repeated work, precompute invariants, then move up the stack when leaf profiles flatten.

MiniMax M3 has 60 decoder layers; 57 use sparse MoE and sparse attention. For a 1K-token response, a small per-layer cost can appear tens of thousands of times in one request. Five questions:

1. What local shape reached this rank?
2. What work repeats at every layer or token?
3. Which bytes move, and can metadata move instead?
4. Did the intended fast path run, with the intended math?
5. When kernels are no longer dominant, which queue grows?

![optimization map](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/02-optimization-map.svg)

**Figure 2.** Top lane: one sparse decoder layer in execution order; AR marks tensor-parallel collectives after attention and MoE. Lower lane: same reasoning on P/D — validate the KV handoff, then add capacity where requests wait.

## 1. What shape reached this rank?

Kernels run on local M, N, and K after tensor parallelism, head replication, padding, and token routing. At TP8, MiniMax M3's 64 query heads shard to eight per rank, while its four KV and four index heads replicate to one per rank. The fused QKV projection therefore sees local **N=1536** — not global N divided by eight.

Prefill and decode also arrive with different M. Prefill processes many tokens at once; decode often has only a few rows. [vLLM #45725](https://github.com/vllm-project/vllm/pull/45725) split the launcher into large-M and small-M regimes, improving TP8 8K/1K output throughput by **7.8%–9.4%**. [vLLM #46117](https://github.com/vllm-project/vllm/pull/46117) then selected tiles from the full local shape: narrower N tiles exposed more independent work in decode; a larger K step reduced loop iterations. Prefill used wider tiles when M already supplied enough parallelism.

![local shape](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/03-local-shape-tile-selection.svg)

**Figure 3.** TP sharding and head replication determine local N; the serving phase determines M. The launcher selects tiles for the shape each rank actually runs.

The same PR reordered grouped-MoE programs so neighboring programs could reuse activation rows and expert-weight tiles from GPU cache instead of fetching them again from HBM. Across its TP4 end-to-end tests, combined changes produced **1.08×–1.46×**. The benefit was largest at low concurrency, where the original launch had the least parallel work.

Local shape also decides which backend is legal. The AITER sparse-attention path in the final MXFP8 recipe requires **one KV head per TP rank**. TP4 satisfies that. TP2 uses vLLM's Triton fallback. Changing TP can change the operator graph, not only collective size.

There was no universally fastest backend. [InferenceX #2003](https://github.com/SemiAnalysisAI/InferenceX/pull/2003) initially selected an emulated linear backend for the whole sweep. Later [InferenceX #2187](https://github.com/SemiAnalysisAI/InferenceX/pull/2187) showed native MXFP8 linear was faster at low and middle concurrency; emulation won only for long-input, high-concurrency runs. Sparse paged attention had a similar crossover. The final recipe enables both only for **8K input at concurrency 64 or higher**.

Lesson: tune and dispatch on the shape distribution that actually runs. “Prefill,” “decode,” or “TP4” is only a label.

## 2. What work repeats?

The first easy-to-miss cost was launch overhead. The day-0 recipe ran eagerly. [InferenceX #1754](https://github.com/SemiAnalysisAI/InferenceX/pull/1754) and [#1755](https://github.com/SemiAnalysisAI/InferenceX/pull/1755) enabled graph execution for standard and EAGLE3 serving. Published results do not isolate that gain.

The larger structural win was the shared expert. Originally every sparse-MoE layer ran it as a separate dense MLP: gate/up, activation, down, intermediate storage, and addition. The math was required. The separate path was not.

[vLLM #46545](https://github.com/vllm-project/vllm/pull/46545) appended the shared expert to the routed expert table and selected it for every token. Grouped GEMMs then handled routed and shared experts together. Output throughput improved **30.2%** at concurrency 1 and **5.6%** at concurrency 128 — launch amortization.

![shared expert](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/04-shared-expert-fusion.svg)

**Figure 4.** Fusion appends the shared expert as a slot selected by every token. Routed and shared experts then use the same grouped GEMMs.

The AITER path applied the same idea in [vLLM #46474](https://github.com/vllm-project/vllm/pull/46474). [vLLM #46184](https://github.com/vllm-project/vllm/pull/46184), backed by [AITER #3811](https://github.com/ROCm/aiter/pull/3811), moved MXFP8 weight and scale reshuffling to model load. AITER carries tuned MoE configurations from 1 to 32,768 tokens and for the local intermediate widths produced by TP4 and TP8. Layout conversion happens once; the serving loop consumes the prepared form.

Speculative decoding: the original MSA indexer launched one workgroup per speculative token. [vLLM #45743](https://github.com/vllm-project/vllm/pull/45743) launched one workgroup per request and processed all draft positions together, reusing key loads. It also removed a positive score scale because only top-k order matters. The index kernel improved by as much as **48.9%**; end-to-end serving improved by about **3.3%** in the PR tests (Amdahl).

## 3. Which bytes move?

Sparse attention reduces attention math but adds a control plane: score blocks, select top-k, map logical blocks to physical pages, pass metadata to the attention kernel.

[vLLM #47269](https://github.com/vllm-project/vllm/pull/47269): adjacent sparse layers often selected nearly the same blocks. With index sharing, one layer computes top-k and later layers reuse it. Mean TPOT fell by about **10%** at concurrency 1 and about **4%** at high concurrency.

Skipping the selector was only half. The fused projection still produced index Q/K, normalized them, applied RoPE, and wrote the index cache. [vLLM #47287](https://github.com/vllm-project/vllm/pull/47287) made reuse visible to that fused kernel so the unused producer branch is compiled away.

The same PR integrated AITER sparse paged attention across a layout mismatch. MiniMax M3 selects logical **128-token** blocks; AITER consumes **16-token** pages. Instead of copying KV, vLLM turns each selected block ID into eight page IDs and builds a compact page table. The KV cache stays where it is.

![sparse page adapter](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/05-sparse-page-adapter.svg)

**Figure 5.** Each selected 128-token block resolves to a physical block, then expands into eight 16-token page entries. AITER reads them through a view of the existing KV allocation; only the table is rebuilt.

At TP4, concurrency 256, the PR improved output throughput by **6.93%** for MXFP4 and **5.56%** for MXFP8 in isolated A/B.

**Benchmark boundary:** InferenceX's fixed 8K/1K review policy **excluded** cross-layer index reuse because it reduces architecture work. The fixed-shape recipe uses the page adapter but **not** top-k reuse. AgentX enables reuse under its workload rules. Do not credit the fixed-contract curve with work it did not run.

Quantization: three independent byte planes:

| Byte plane | MiniMax M3 example | What must be proved |
|---|---|---|
| Weights and activations | MXFP8 or MXFP4 GEMM/MoE | Packing, scales, activation math, backend layout |
| Persistent state | FP8 KV and sparse index cache | Platform dtype, page layout, read/write geometry |
| Communication | Quantized all-reduce or KV transfer | Eligibility, selected codec, ownership, completion |

“The model is MXFP4” does not tell us the KV dtype or the collective path.

## 4. Did the fast path run — and was it correct?

This audit changed one claim. They initially believed a roughly **1.5 MB** decode collective used INT4 QuickReduce. The available evidence does not prove it.

[InferenceX #2104](https://github.com/SemiAnalysisAI/InferenceX/pull/2104) configured INT4 and a **256 KB codec** threshold, but not QuickReduce's separate eligibility threshold. For BF16 at TP4, the [pinned built-in table](https://github.com/vllm-project/vllm/blob/69715823df89b11ee684b84066390cbb9092d5c1/vllm/distributed/device_communicators/quick_all_reduce.py#L49-L61) requires **16 MB** for INT4. The 1.5 MB collective therefore does not reach codec selection; the 256 KB threshold is consulted only after QuickReduce is eligible.

![QuickReduce gates](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/06-quickreduce-dispatch-gates.svg)

**Figure 6.** QuickReduce checks eligibility before choosing FP or INT4. Here the collective falls below the built-in eligibility gate, so configuration alone cannot establish execution.

Logs prove INT4 was **configured**, not that the QuickReduce kernel **ran**. Treat #2104 as a cumulative image and recipe checkpoint; do not attribute its curve to INT4 all-reduce.

```text
configured  !=  eligible  !=  executed
```

Use a dispatch trace or profiler before assigning a gain to a backend.

Correctness examples (different broken contracts):

- [vLLM #45794](https://github.com/vllm-project/vllm/pull/45794) mapped packed MXFP4 Q/K/V and gate/up checkpoint tensors into the correct fused-parameter slices and passed MiniMax M3's SwiGLU-OAI parameters into MoE.
- [vLLM #45720](https://github.com/vllm-project/vllm/pull/45720) fixed the FP8 KV view on FNUZ ROCm devices. On MI300X, unpatched GSM8K strict match **0.0099**; patched **0.9575**. Correctness fix, **not** a claimed MI355X speedup.
- [vLLM #47158](https://github.com/vllm-project/vllm/pull/47158) fixed the expert-parallel mask passed to AITER. Buggy cosine similarity **0.527**; corrected **1.0**, restoring GSM8K accuracy.

The last two do not explain the TP4/EP1 hero curve. For performance work, “passed” should mean: the output is correct, the intended path executed, and the end-to-end metric improved under the same contract.

## EAGLE3 adds a second decode loop

EAGLE3 adds a draft model, multi-token verification, acceptance behavior, and a second set of attention metadata. It is not a flag on the standard curve.

[vLLM #45546](https://github.com/vllm-project/vllm/pull/45546) connected the AMD model to the EAGLE3 interface. [vLLM #45564](https://github.com/vllm-project/vllm/pull/45564) fixed a cache-key bug: target and draft use different query-head counts, so they must not share an attention-group builder merely because backend and KV type match. General cache rule: the key must contain every invariant that changes the cached object.

After request-level index batching, [InferenceX #2107](https://github.com/SemiAnalysisAI/InferenceX/pull/2107) found that the target's attention-backend setting did not configure the draft. Pinning `TRITON_ATTN` inside the speculative config avoided the draft's slower fallback.

[vLLM #47984](https://github.com/vllm-project/vllm/pull/47984) extended AITER sparse paged attention from one-token decode to multi-token verification. It maps each flattened query row back to its request and local speculative position, reuses the existing page-table builder, and preserves the one-token fast path. TP4 tests improved output throughput by **8.32%** (MXFP4) and **7.90%** (MXFP8) without materially changing acceptance.

Together: the separate **682.4** output tok/s/GPU EAGLE3 result at concurrency 128.

## 5. Which queue grows?

Prefill/decode disaggregation moved the bottleneck above one process. Before tuning worker counts, the KV boundary had to be correct.

The initial MORI-IO path assumed the first layer's KV layout represented every layer. MiniMax M3 has separated K/V tensors, interleaved K/V tensors, and a key-only index cache. Transfer completed and throughput looked healthy, but GSM8K fell to roughly **0.0008** — token salad.

Repair in three PRs:

- [vLLM #46039](https://github.com/vllm-project/vllm/pull/46039) derived transfer geometry and byte offsets per layer.
- [vLLM #46290](https://github.com/vllm-project/vllm/pull/46290) counted the writes actually scheduled for each request, sealed that count after forward, and released buffers only after those writes completed.
- [vLLM #46332](https://github.com/vllm-project/vllm/pull/46332) added heterogeneous-TP rank mapping and acknowledgment fan-in. With prefill TP4 and decode TP8, two decode ranks can consume one producer rank, so both must acknowledge before its blocks are reused.

Only then was worker allocation worth tuning.

![P/D profiles](../../../../assets/vllm/blog/performance/minimax-m3-mi355x/07-pd-system-profiles.svg)

**Figure 7.** Two 8K/1K P/D operating points. Endpoints use different GPU counts and concurrency, so this is system evolution, not a controlled speedup.

First public profile: one TP8 prefill worker and one TP8 decode worker. At concurrency 1024: **2,084.6** total tok/s/GPU, but median TTFT **223.20** s ([InferenceX #1762](https://github.com/SemiAnalysisAI/InferenceX/pull/1762)). The prompt queue was the signal.

[InferenceX #2144](https://github.com/SemiAnalysisAI/InferenceX/pull/2144) moved every worker to TP4, synchronized them with the faster single-node recipe, and searched the prefill/decode ratio. For 8K/1K: two TP4 prefill workers fed one TP4 decode worker. At concurrency 512: **6,370.5** total tok/s/GPU and **1.32** s median TTFT.

Mean TPOT moved from **31.26** to **54.60** ms. Not a contradiction: added prefill capacity cleared the admission queue, while the selected decode point produced each active sequence more slowly. P/D has at least two latency objectives. Publish both.

One high-concurrency run also exhausted the container's file-descriptor limit. Raising `nofile` fixed the TCP failures. An OS limit can be as real as a GEMM tile.

## AgentX shows the next bottleneck

Fixed 8K/1K is for controlled comparisons. Agentic coding is not fixed-shape: long multi-turn traces, reusable prefixes, irregular outputs, a KV-capacity knee.

[InferenceX #2487](https://github.com/SemiAnalysisAI/InferenceX/pull/2487) is the first MI355X MiniMax M3 AgentX point with MXFP4, EAGLE3-GQA, prefix caching, optional TP-sharded LMCache, and cross-layer index reuse. Throughput replay uses a committed synthetic acceptance length so compared systems do the same speculative work; eval uses real target verification.

[Successful run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31558297538): TP4 at concurrency 28 delivered **127.4** output tok/s/GPU, **509.5** total output tok/s, **0.582** mean QPS, **645** ms p50 TTFT, **41.3** ms p50 TPOT.

Service metrics:

- Theoretical prefix-cache hit rate: **96.7%**
- Realized GPU cache hit rate: **92.1%**
- GPU KV-cache use: **88.5%**
- GPU KV capacity: **6,264,960** tokens

At this point another GEMM is not automatically the next project. The **4.6**-point cache-realization gap and the near-capacity operating point point at prefix alignment, admission and eviction, scheduling, and offload. Observations from one run, not yet an optimization claim. Baseline for the next agentic round.

## A checklist for the next model

When a new serving path works but is not yet fast:

1. Record local shape histograms after sharding. Include replicated heads and routed-token counts.
2. Estimate repetition. Multiply per-layer work by layers, output tokens, and active requests.
3. Separate the byte planes: compute tensors, persistent state, and communication.
4. For every fast path, record its eligibility condition and verify its execution.
5. Put a correctness gate beside every performance gate.
6. After each win, profile again. If leaf kernels flatten, inspect queues, ownership, cache capacity, and OS limits.

MiniMax M3 became faster because the team kept changing the level of the question — from tiles, to repeated paths, to sparse metadata, to distributed state, and finally to workload queues.

## Reproduce the fixed-shape result

Final MXFP8 TP4 checkpoint image:

```text
vllm/vllm-openai-rocm:nightly-9e57de7197f234f9d9187715d96e07e007048c0f
```

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --language-model-only \
  --moe-backend aiter \
  --max-model-len 10240 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice
```

Figure 1 high-concurrency dispatch: gated recipe in [InferenceX #2187](https://github.com/SemiAnalysisAI/InferenceX/pull/2187). MXFP4 TP2: [#2446](https://github.com/SemiAnalysisAI/InferenceX/pull/2446). P/D: [#2144](https://github.com/SemiAnalysisAI/InferenceX/pull/2144). Copying only the flags above does not reproduce a different image, topology, or workload.

Concurrency-128 bench:

```bash
vllm bench serve \
  --backend vllm \
  --model MiniMaxAI/MiniMax-M3-MXFP8 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --num-prompts 1280 \
  --max-concurrency 128 \
  --request-rate inf \
  --ignore-eos \
  --num-warmups 256 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result
```

## Acknowledgements

MiniMax for MiniMax M3. Named on the page: Aakif Nawaz, Ajith Sirra, Bryan Shan, Bugen Zhao, Cameron Quilici, Chun Fang, Duyi Wang, Ethan Yang, Fangzhou Ai, Felix Marty, functionstackx, Hongxia Yang, Isotr0py, Jun Kang Chow, Pin Siang Tan, Qiang Li, Seung Rok Jung, Sun Peng, Tian Di, Tun Jian Tan, Uma Kannikanti, wangjiaxin99, Ye Hur Cheong, youkaichao, Yue Liu, Zheng Gong. Broader vLLM, AMD, Embedded LLM, Inferact, and SemiAnalysis InferenceX reviewer communities.
