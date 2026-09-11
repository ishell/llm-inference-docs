---
source: https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading
lang: en
fetched: 2026-09-11
---

# GLM 5.3 Optimizations, Part 1: Hybrid HiSparse Offloading in vLLM

Chinese: [zh/vllm/blog/serving/glm53-hisparse.md](../../../../zh/vllm/blog/serving/glm53-hisparse.md)

2026-09-08. **vLLM Team**. Study extract, not an official reprint. Two-part series; this is Part 1 (aggregated 8× H200). Part 2 (named, not yet in CATALOG) will combine PCP, [DCP](../performance/dcp.md), [adaptive verification](../performance/dspark-adaptive.md), and Hybrid HiSparse at large P/D scale. GLM-5.2 SLA sibling: [glm52-b300.md](glm52-b300.md). CPU/tiered offload: [kv-offload.md](kv-offload.md), [tiered-kv-offload.md](tiered-kv-offload.md). Papers: [HiSparse](https://arxiv.org/abs/2608.07009), [IndexShare](https://arxiv.org/abs/2603.12201). Page occupancy calculator is an HTML widget — not mirrored. Repro client scripts stay on the site.

**TL;DR (page):** Hybrid HiSparse is a pressure-driven memory tier that composes with the Hybrid Memory Allocator and KV offloading. On a single 8× H200 node that is tight for GLM 5.3, it enables the full **1 million** context length (previously impossible on this hardware) and higher concurrency across context lengths.

## Exploiting sparsity when we need to

Agentic workloads: many concurrent requests, each with a long context that keeps growing. The GPU block pool is fixed, so the KV cache eventually runs out of room for new blocks.

Two existing options, each with a tradeoff:

- **Preemption** picks a request, drops its KV, and re-prefills later. The request pays its full TTFT again on every eviction.
- **Offloading** moves blocks to host memory, but dense attention needs every token GPU-resident, so concurrency stays bounded by GPU memory.

For sparse-MLA KV, the indexer selects top-K tokens and attends only to those. [HiSparse](https://arxiv.org/abs/2608.07009) offloads all KV except those selected tokens to the CPU, which bounds the GPU memory each request needs. Indexer KV stays GPU-resident and still grows with context, but it is much smaller. GLM 5.3's [IndexShare](https://arxiv.org/abs/2603.12201) means **one indexer layer per four sparse-MLA layers**.

**Hybrid HiSparse** keeps KV on the GPU while there is capacity. Only under KV pressure does it apply HiSparse offloading. Hot buffer pages are indexed by tokens; a page can hold tokens from many different CPU blocks, so reduction spans a wide context. CPU–GPU transfers are paid only when the system is under KV pressure (higher concurrency).

![two requests](../../../../assets/vllm/blog/serving/glm53-hisparse/01-hisparse-two-requests.svg)

**Figure (page two-request cartoon).** Preemption: B's slots are freed and its KV is gone. Conventional offload: B's KV survives on host, so we do not re-prefill, but B still cannot run until all of it fits on GPU again, so A decodes alone. Hybrid sparse offload: each request releases its coldest pages in place, the same slots are re-leased as new tails and hot pages, and both keep decoding.

Only Hybrid HiSparse keeps both requests decoding. Hot pages are leased from the **same** block pool as KV pages, and they live in the **same** KV-cache tensor, so they look like ordinary pages to the sparse MLA kernel. Uniquely for hybrid sparse, some tokens can sit in hot buffers while others remain in GPU-resident pages, which reduces CPU reloading.

## How it works

![residency](../../../../assets/vllm/blog/serving/glm53-hisparse/02-hisparse-residency.svg)

**Figure (page residency panels).** The same six top-K tokens are ringed in every panel; only residency changes. Solid arrows: misses copying one row into a hot page. Dashed arrows: hot hits reused without a copy.

Residency is tracked per page. A request moves among three states as pressure rises and falls:

- **Full residency:** all sparse-MLA KV remains GPU-resident while completed prefix pages are proactively materialized in host memory.
- **Mixed residency:** the tail stays on the GPU, older pages live only in CPU memory, and the rows the indexer wants from those pages sit in hot buffers. The block table holds real blocks and null placeholders side by side; the tail is never evicted. One fused kernel resolves top-K: resident tokens are read in place, hot tokens are read and their LRU entry refreshed, and a miss copies a single row from pinned host memory into an LRU slot. Nothing on the decode path waits on a CPU decision, so it stays CUDA-graph-capturable.
- **No residency:** a new request reusing a prefix that only exists in CPU memory starts with placeholders and a hot page. Rows arrive as the indexer selects them, so we pay for what the model attends to rather than the whole history.

All three states work because hot buffers are **not** a separate allocation. A hot buffer page is an ordinary KV-cache block, leased from the same pool as resident pages through the Hybrid Memory Allocator, taken when a request first needs one and returned when it does not. For rows in a resident page or in the hot buffer, the resolver hands HMA row IDs and HMA gathers them with one stride. A block freed by one request can become hot-buffer capacity for another.

HiSparse prepares for pressure before it arrives. When a cacheable prefix page is complete, HiSparse queues a copy to CPU while continuing to serve it from the GPU. If the GPU cache later fills, that page can release its GPU slot without another copy. Even if pressure reaches a newer page first, its GPU slot becomes reusable as soon as the copy is queued, and the CPU copy becomes available for prefix reuse when the transfer completes.

The `hisparse-glm` branch keeps this path lightweight by copying all sparse-MLA layers together in one launch after the forward pass. The copy is ordered on the model's GPU stream.

## Composing with the rest of vLLM

Hybrid HiSparse is a residency policy over the shared HMA pool and a connector alongside vLLM's other KV machinery. Other cache groups still use normal prefix caching, transfer, and offloading. Indexer KV in particular is **untouched** by HiSparse: the standard OffloadingConnector can offload it independently at block granularity. Imports from P/D disaggregation can land host-side when a prefix does not fit resident. Speculative decoding works through per-step replayable resolver plans that share the request's hot state.

Hot buffers default to **2× top-K** rows per request. Because MLA KV is identical across TP ranks, the pinned host pool is allocated **per DP replica** and shared across its local TP ranks. TP rank 0 writes the shared copy; every rank can read it; a CUDA event preserves stream ordering.

## The numbers

GLM 5.3 on **8× H200**, OpenHands multi-turn agentic workload ([source](https://www.lmsys.org/blog/2026-07-13-glm52-optimization)): 13-turn conversations, **74,160**-token first turn, **753**-token later turns, fixed **220**-token outputs. Both TP8 deployments used MTP3, FP8 KV cache, **142K** admission limit, `max_num_batched_tokens=32768`, `max_num_seqs=256`, `gpu_memory_utilization=0.92`. Offloading baseline: **512 GiB** offload pool. Hybrid HiSparse split the same host budget into **384 GiB** HiSparse pool + **128 GiB** offloading.

![OpenHands Pareto](../../../../assets/vllm/blog/serving/glm53-hisparse/03-openhands-pareto-occupancy.svg)

**Figure (page).** Top: interactivity–throughput sweep. Interactivity is 1000 / mean TPOT; logical total-token throughput includes prefix-cached prompt tokens and is divided by eight GPUs. Bottom: mean non-zero `vllm:num_requests_running` samples during each point. Hybrid HiSparse: `e8ef1e07bd`. Offloading baseline: `80cb71c9ff`.

They plan to make Hybrid HiSparse widely available in **vLLM v0.30**. Exact launch commands are in the appendix below.

## Offloading only where we need it

KV starts on the GPU and stays there while there is room, then gives up residency page by page as the pool runs short. Hot buffers and resident pages share pool and tensor, so a request under pressure keeps decoding at partial residency instead of waiting for a slot or paying to prefill again.

## Estimate the benefit (page calculator — not mirrored)

The live page embeds a concurrency calculator (`/assets/interactive_pages/hisparse_concurrency_calculator.html`). Planning estimates, **not** guaranteed serving limits: runtime workspaces, request-length skew, and scheduling can lower concurrency in practice.

MTP can further limit concurrency because hot buffers must hold all verification tokens at once. At publication, size each hot buffer to `(num_speculative_tokens + 2) × top-K`. Subject to change as they shrink buffers. The calculator did **not** take this into account yet.

## Part 2

Hybrid HiSparse matters most on the **decode** side of a P/D deployment, where contexts are longest and KV pressure is highest. Part 2 (announced) puts the pieces together at large scale: Prefill Context Parallelism (PCP), [DCP](../performance/dcp.md), [adaptive verification](../performance/dspark-adaptive.md), and Hybrid HiSparse.

## Acknowledgements

Implementation: Matthew Bonanni (Red Hat), Lucas Wilkinson (Red Hat), Fares Obeid (Prime Intellect). Design collaboration: Chao Lei (Ant Group), Nicolò Lucchesi (Mistral). Performance evaluation and blog: Simon Veitner (Red Hat). Thanks to the [HiSparse](https://arxiv.org/abs/2608.07009) authors for the sparse offloading concept.

## Appendix: reproducing the results

Results use [vLLM `e8ef1e07bd`](https://github.com/neuralmagic/vllm/commit/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1) (neuralmagic fork pin). Until v0.30, build that checkout. Hybrid HiSparse on one 8× H200 node:

```bash
vllm serve zai-org/GLM-5.3 \
  --served-model-name glm-agentx \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 142000 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --attention-config '{"hisparse_config":{"host_pool_gib":384}}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":137438953472}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

`host_pool_gib` is per DP replica and is rounded to whole host blocks. The **128 GiB** offloading pool (`cpu_bytes_to_use` **137438953472**) stores cache groups HiSparse does not manage, including indexer KV. HiSparse without MTP: omit `--speculative-config`. No-HiSparse MTP3 baseline: keep `--speculative-config`, omit `--attention-config`, set `cpu_bytes_to_use` to **549755813888** (512 GiB). Omit both HiSparse and `--speculative-config` for the no-MTP baseline. HiSparse is currently **NVIDIA-only**.

### Padded OpenHands sweep

Client assets on the blog (not copied here): `build_openhands_padded_dataset.py`, `install_evalscope_deps.sh`, `evalscope-all-nodeps.txt`. EvalScope pin: `acd09b44384d53174768bb1063f675420f76fae9`. Builds a deterministic 128-conversation dataset, then runs c1/c8/c16/c24/c32 with fresh conversations at every point:

```bash
python3.12 -m venv client-venv
source client-venv/bin/activate
bash install_evalscope_deps.sh
pip install 'modelscope[datasets]==1.34.0' 'lxml==6.0.2'
pip install 'evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9'

python build_openhands_padded_dataset.py \
  --model zai-org/GLM-5.3 \
  --pad-source openscience \
  --first-turn-length 74160 \
  --subsequent-turn-length 753 \
  --num-turns 13 \
  --number 128 \
  --output-path openhand-zai-org-GLM-5.3.json

evalscope perf \
  --model glm-agentx \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path openhand-zai-org-GLM-5.3.json \
  --dataset-offset 52 \
  --max-tokens 220 \
  --multi-turn \
  --number 4 16 32 48 64 \
  --parallel 1 8 16 24 32 \
  --extra-args '{"ignore_eos":true}' \
  --name tp8-hisparse384-native128 \
  --outputs-dir results \
  --no-timestamp
```

For the figure: interactivity is `1000 / mean_TPOT_ms`; logical total-token throughput per GPU is EvalScope's total token throughput divided by eight. They scraped `/metrics` every **30** seconds. Request occupancy is the mean of non-zero `vllm:num_requests_running` samples. MTP acceptance length is `1 + Δ(vllm:spec_decode_num_accepted_tokens_total) / Δ(vllm:spec_decode_num_drafts_total)`.
