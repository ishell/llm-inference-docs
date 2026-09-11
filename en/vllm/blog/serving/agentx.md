---
source: https://vllm.ai/blog/2026-09-08-vllm-agentx
lang: en
fetched: 2026-09-11
---

# vLLM x AgentX: Optimizing for Real-World Agentic Serving

Chinese: [zh/vllm/blog/serving/agentx.md](../../../../zh/vllm/blog/serving/agentx.md)

2026-09-08. **vLLM Team and Inferact**. Study extract, not an official reprint. Benchmark: [SemiAnalysis AgentX](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat). Dashboard: [InferenceX](https://inferencex.semianalysis.com/inference). Harness: [SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness). May 2026 sibling: [mooncake.md](mooncake.md). DCP: [dcp.md](../performance/dcp.md). Kimi K3: [kimi-k3.md](kimi-k3.md). DeepSeek V4: [deepseek-v4.md](../architecture/deepseek-v4.md). MiniMax M3: [minimax-m3.md](minimax-m3.md), AMD follow-up [minimax-m3-mi355x.md](../performance/minimax-m3-mi355x.md). Page interactive embeds (session slider, packed-KV toggle, hover Pareto) not mirrored; static figures kept. Site JS not copied.

**TL;DR (page):** Agentic traffic is a major vLLM load: multi-turn sessions, long contexts, heavy prefix reuse. Coordinated work across KV cache, parallelism/kernels/scheduling, and P/D ratio. On AgentX: up to **130K** total tokens per GPU-second on DeepSeek V4 Pro; interactivity up to **376** tokens/s on MiniMax M3. Across DeepSeek V4 Pro, MiniMax M3, and Kimi K3: **14.6×–106×** serving-cost advantage versus Opus 5 API pricing (tables below). Comparison is serving cost, **not** model quality.

![hero](../../../../assets/vllm/blog/serving/agentx/01-hero-vllm-agentx.png)

![Figure 1 Pareto](../../../../assets/vllm/blog/serving/agentx/10-agentx-pareto-summary.png)

**Figure 1 (static copy).** vLLM on SemiAnalysis AgentX. Total tokens per $1 of TCO against P90 interactivity for the best vLLM configuration of DeepSeek V4 Pro, MiniMax M3, and Kimi K3, with DeepSeek V4 Pro on GB300 NVL72 as a case study. Source: SemiAnalysis AgentX. Live hover plot on the original page.

## Characterizing agentic workloads

Since the [May Mooncake post](https://vllm.ai/blog/2026-05-06-mooncake-store), agentic share has grown. As of June 2026, [OpenAI reported](https://openai.com/signals/enterprise-data/) that Codex generated <strong>64%</strong> of combined Codex and ChatGPT output tokens among enterprise customers.

That stresses two axes: **cost** (how many concurrent agents fit a fixed hardware budget) and **latency** (how fast each agent moves through reasoning and tool cycles). Optimize the latency–cost frontier as a whole.

AgentX is built from real agentic coding traces:

- **Long-running, multi-turn sessions.** Median **43** turns per session.
- **Long contexts with short outputs.** Median input **142K** tokens, median output **444** tokens.
- **Extensive prefix reuse.** Prefix-cache hit rate above <strong>96%</strong>.
- **Subagent-heavy traffic.** <strong>44%</strong> of sessions contain at least one subagent; median **four** subagent rollouts among those sessions.

Each turn appends the latest tool result to the accumulated context and sends the whole thing back, so input keeps growing while each turn adds only a short new prefill, and almost all of the request is a prefix the engine has already seen. Subagents fork from that context or start fresh; results join back into the parent. The original page has an interactive session explorer (Figure 2) — not mirrored.

## Three serving challenges

1. **Prefix cache pressure.** Every turn replays the full conversation so far. To keep many sessions running, the engine offloads KV between turns. Harder at scale across GPUs, P/D instances, and replicas.
2. **Execution efficiency.** Long contexts and tight latency: more tokens, more work per token, less time. Parallelism, kernels, scheduling, speculative decoding must match the new request shape.
3. **Finding the right P/D ratio.** Context lengths and cache hit rates vary wildly across sessions and subagents. Routing must balance cache affinity and load. The throughput-optimal P/D ratio also shifts with concurrency.

## The vLLM approach: three planes

![full stack](../../../../assets/vllm/blog/serving/agentx/02-full-stack-overview.png)

**Figure 3.** Full-stack optimization. Data plane: distributed shared KV. Execution plane: parallelism and kernels per model. Control plane: P/D ratio and request scheduling.

### Data plane: keep KV warm and close to compute

#### Hybrid KV cache management

Agentic long contexts pressure KV capacity. Hybrid models mix sliding-window and linear attention with full attention; cached blocks differ in size and lifetime.

vLLM's hybrid KV cache manager uses a **uniform memory page** as the allocation unit, in **one shared block pool**.

![hybrid KV](../../../../assets/vllm/blog/serving/agentx/03-hybrid-kv-cache-manager.png)

**Figure 4.** One page size for all attention types; a single block pool is shared. Reallocate on demand instead of statically partitioning by attention type. Full-attention KV grows with sequence length; sliding-window and recurrent state follow different lifetimes. The best partition therefore changes with concurrency, context length, and prefix reuse.

The abstraction keeps evolving. [DeepSeek V4](../architecture/deepseek-v4.md)'s initial layout fragmented cache types into three size buckets and allocated **92** separate tensors — wasted padding, inefficient P/D transfer and offload.

The [packed KV cache layout](https://github.com/vllm-project/vllm/pull/44577) stores all cache groups and layers in **one contiguous backing allocation per block**. That reduces descriptor and P/D transfer overhead, and permits a smaller allocation unit when the FP4 indexer is enabled, saving [roughly 10% of KV cache memory](https://github.com/vllm-project/vllm/pull/48993). Page Figure 5 is an interactive MXFP4/FP8 toggle — not mirrored.

#### Hierarchical KV cache offloading

[Mooncake Store](https://github.com/kvcache-ai/Mooncake) as a distributed KV pool ([May post](https://vllm.ai/blog/2026-05-06-mooncake-store)).

- **Model architecture parity.** Sparse / compressed / linear attention stay first-class, with async scheduling, P/D, speculative decoding, and parallelism still working.
- **Hierarchical tiers.** Disks and extra CPU-only nodes via Mooncake `standalone-store`: an external Mooncake client owns the CPU pool and disk tier; vLLM workers are requesters. Launch a standalone client per node to expand the pool. Integrated with [Dynamo](https://github.com/ai-dynamo/dynamo) and [llm-d](https://github.com/llm-d/llm-d) so requests can hit cache on any instance.
- **Performance.** Hybrid models must key and look up per attention type (CPU overhead). Reduced via better data structures, async lookups, work off the scheduler critical path, and parallel send/receive: [PR#46188](https://github.com/vllm-project/vllm/pull/46188/changes), [PR#45444](https://github.com/vllm-project/vllm/pull/45444/changes), [PR#45659](https://github.com/vllm-project/vllm/pull/45659/changes), [PR#47317](https://github.com/vllm-project/vllm/pull/47317/changes).
- **Session-aware prefix-cache retention** for hybrid models (linear / sliding-window + full attention). Reuse needs the linear state or sliding-window cache at the reuse boundary. Keeping snapshots at every token is expensive, so two policies:
  1. [**Interval-based retention**](https://github.com/vllm-project/vllm/pull/43447) preserves prompt-end caches / linear states at each turn. Later turns and forked subagents can reuse.
  2. Shared prefixes often end *within* a turn, so [**Marconi-style selective retention**](https://github.com/vllm-project/vllm/pull/47782) retains a checkpoint when a prefix is observed a second time. First miss recomputes missing state and saves a checkpoint at that boundary.

Together they keep a high hit rate without excessive storage. Depth: [kimi-k3.md](kimi-k3.md).

### Execution plane: generate tokens fast

#### Model-specific parallelism

Axes: TP, DP, EP, PP, CP. Optimum depends on architecture, topology, workload, latency SLOs. Two representatives on NVIDIA GB/B-series (and AMD counterparts).

**Kimi K3.** MLA + Kimi Delta Attention (KDA). MLA compresses KV into one latent head, so plain TP that replicates that latent cache is inefficient.

[DCP](../performance/dcp.md) shards the cache along the sequence dimension (each rank keeps 1/N of KV):

- **Lower decode latency.** MLA attention is memory-bound and grows with context. As agentic prefixes grow, attention is a larger share of each decode step; sharding shortens that step.
- **Higher throughput and KV capacity.** Avoiding KV replication keeps more sequences in flight without stalling on KV admission.

![K3 TP8 vs DCP8](../../../../assets/vllm/blog/serving/agentx/04-k3-tp8-vs-dcp8.png)

**Figure 6.** For Kimi K3, DCP8 achieves lower decode latency than TP8 and scales to higher concurrency.

DCP cost: extra communication. Sequence-sharded KV means every MLA decode layer needs a query gather before attention and a partial-output reduction after.

They bypass NCCL with **symmetric-memory buffers** that peer GPUs load from and store to directly. Queries are multicast into buffers consumed by attention kernels. Each GPU writes partial attention outputs and log-sum-exp (LSE) statistics into peers' receive slots; each rank merges locally with online softmax. GPU-to-GPU writes fuse with compute in the same kernels, cutting latency by about <strong>13%</strong> per layer versus default DCP8.

![DCP symmetric memory](../../../../assets/vllm/blog/serving/agentx/05-k3-dcp-symmem.gif)

**Figure 7.** MLA decode path under DCP4 using symmetric memory. Each step fuses into a single kernel, replacing NCCL all-gather, staging copy, all-to-all, and unpack.

A larger scale-up domain can change the best strategy. On NVL72-class systems, wide EP with data parallelism (**DEP**) can scale better than DCP at the same decode latency SLO. At larger multi-node DCP sizes, sharded-attention communication outweighs the compute it saves. DEP assigns requests and their KV to different DP ranks, avoiding DCP's attention collectives while sharding MoE experts.

![DCP8 vs DEP16](../../../../assets/vllm/blog/serving/agentx/06-k3-dcp8-vs-dep16.png)

**Figure 8.** For Kimi K3, wide EP (DEP16) scales better than DCP8 once the per-rank batch size exceeds **3**.

**DeepSeek V4.** MLA-style KV also replicates under TP. Compressed sparse attention makes TP head sharding compute-inefficient for three reasons:

- Compressor paths produce one shared KV representation per compressed position, not independent per-head states. TP cannot shard along the KV-head dimension; every rank repeats compressor work.
- The indexer has 64 heads but one global top-k per token. The current TP path replicates the full indexer on every rank (avoids a dense score reduction before top-k, duplicates work).
- Sparse MLA is dominated by scanning and gathering top-k KV, not attention arithmetic. TP repeats much of this memory-bound work while dividing only cheaper head-wise compute.

In practice **PCP** is best for long prefills; **DEP** works across a broader range. PCP shards the prompt sequence (query tensor), distributing compressor and indexer work, and giving sparse MLA a wider head-local shape. For a **32K** prompt, PCP8 is a **2.65×** prefill speedup over TP8 (TTFT). It still replicates decode-side state, so it fits dedicated prefill workers.

DCP is less effective for V4 than for Kimi K3 (see bitter lessons). DEP keeps attention fully local; it is the default for most DeepSeek V4 configurations.

#### Scheduling mixed agentic traffic at two levels

Mix: frequent append-only requests (long prefixes, short prefills) and occasional fresh prefills of tens of thousands of tokens. Two problems: inside an instance, a long prefill can block short interactive turns; across DEP ranks, uneven prefill placement imbalances the group.

##### Breaking head-of-line blocking

Default chunked-prefill scheduler is FIFO. One long prefill can claim the entire token budget step after step; short turns on the same rank wait until it finishes.

![HOL blocking](../../../../assets/vllm/blog/serving/agentx/07-hol-blocking.gif)

**Figure 9.** Head-of-line blocking, session view of one rank. Left: no chunk cap, a long prefill claims the whole budget. Right: 512-token cap, short turns join every step and begin decoding sooner.

`--long-prefill-token-threshold` caps how many tokens one request may schedule per step. With a **512**-token threshold, a long prefill leaves room for short turns. DeepSeek V4 Pro on B300s: total tokens per GPU-second (TPGS) up to <strong>+93%</strong>, P90 interactivity roughly **2.3×**. Trade-off: higher TTFT for the long request itself. TTFT-sensitive deployments should use a larger threshold.

##### Align DEP prefill schedule cadence

MoE all-to-all forces ranks to advance in lockstep, so a rank doing prefill slows the whole group. Prefills arriving on different steps pay that penalty repeatedly.

`--prefill-schedule-interval` admits prefill work only every Nth engine step, with a counter aligned across DP ranks. Prefill concentrates on the same steps; remaining steps are decode-only.

![prefill cadence](../../../../assets/vllm/blog/serving/agentx/08-prefill-schedule-interval.gif)

**Figure 10.** Prefill schedule cadence across a DEP8 group. Left: prefills arrive on different steps and stall the lockstep group. Right: interval of 4 coalesces prefills onto cadence steps; in-between steps are decode-only.

### Scaling with optimal P/D

More GPUs or disaggregation will not automatically improve the frontier. Prefill and decode must be **rate-matched**. Two-phase methodology (can be automated):

**Phase 1: saturation profiling.** Prefill-only and decode-only separately. Sweep parallelism (TP vs wide EP) and size (8, 16, or 32 GPUs) with rising concurrency until throughput saturates. Output: max prefill/decode req/s for each (parallelism, size).

**Phase 2: P/D sweep.** Derive the P/D ratio from Phase 1 saturation points, then sweep concurrency on the combined disaggregated deployment.

### Kernels and community contributions

Bottlenecks shift toward long-context attention, speculative decoding, and communication. All kernels named here are open source; some already adopted by other engines.

- MiniMax M3: [CuteDSL long-context indexer](https://github.com/vllm-project/vllm/pull/48582) improves reported GB300 indexer latency by roughly <strong>3%–31%</strong> depending on shape. Upstreamed MSA top-k: up to **4×** worst-case kernel, about <strong>7%</strong> AgentX end-to-end throughput. Speculative-verification path: about <strong>20%</strong> medium-batch decode in reported tests.
- Kimi K3: [GEMM and reduce-scatter fusion](https://github.com/vllm-project/vllm/pull/52079) for sequence-parallel communication; [latent-tail MoE fusion](https://github.com/vllm-project/vllm/pull/53152) reduces end-to-end latency by roughly <strong>5%</strong>.
- DeepSeek V4: MXFP4 MoE and HCA compression ([#43584](https://github.com/vllm-project/vllm/pull/43584), [#44230](https://github.com/vllm-project/vllm/pull/44230)); [multi-stream C4A](https://github.com/vllm-project/vllm/pull/42925); [cluster-based top-k](https://github.com/vllm-project/vllm/pull/43008).

## Performance: agentic-first, openly verifiable

Independent validation on AgentX: open dataset from **$3M** of real agentic coding traces with **1M** context, public infrastructure of more than **1,000** chips and roughly **2 MW**.

![K3 dashboard](../../../../assets/vllm/blog/serving/agentx/09-k3-agentx-dashboard.png)

**Figure 11.** Total tokens per $1 under varying P90 interactivity, Kimi K3 on various hardware. Source: Kimi K3 SemiAnalysis AgentX Dashboard. Live: [AgentX Dashboard](https://inferencex.semianalysis.com/inference?i_seq=agentic-traces&i_xmode=interactivity&g_runid=33418433573&i_best=0&i_active=b200_vllm%2Cb300_vllm%2Cgb200_dynamo-vllm%2Cgb300_dynamo-vllm&i_hc=1&i_advlabel=0&i_label=0).

Highest-throughput vLLM configuration that keeps **P90 interactivity > 50 tok/s/user**:

| Model | GPUs / concurrency | Total tokens per GPU-second (TPGS) @ P90 > 50 tok/s | P90 interactivity |
|---|---:|---:|---:|
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | 12 GB300s / 256 | **83K TPGS** | 58.3 tok/s |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | 2 B300s / 24 | 70K TPGS | **74.2 tok/s** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | 16 GB300s / 48 | 11.8K TPGS | 62.7 tok/s |

TPGS counts input, output, **and** cached tokens. Breakdown via each model's link.

DeepSeek V4 Pro: 12-chip GB300 P/D, 256 concurrent sessions, 58.3 tok/s/user at P90, 83K TPGS. MiniMax M3: 2 B300s, 74.2 tok/s/user at P90, 70K TPGS. Kimi K3 (2.8T): too large for a conventional single server; 16 GB300s, 62.7 tok/s/user at P90, 11.8K TPGS.

Serving cost versus Opus 5:

| Model | GPU TCO/hour | Equivalent Opus 5 cost/hour | Cost advantage |
|---|---:|---:|---:|
| [DeepSeek V4 Pro 1.6T](https://inferencex.semianalysis.com/inference/agentic/439873) | $27.72 | $2,926 | **106×** |
| [MiniMax M3 428B](https://inferencex.semianalysis.com/inference/agentic/439907) | $4.52 | $384 | **85×** |
| [Kimi K3 2.8T](https://inferencex.semianalysis.com/inference/agentic/441066) | $36.96 | $538 | **14.6×** |

Opus 5 calc: cached input × **$0.50/M** + uncached input × **$5/M** + output × **$25/M**. Assumes a **perfect theoretical** cache hit rate; excludes cache-write charges and long-context premiums — conservative and favorable to Opus. Serving cost, not quality.

Advantage comes from agentic traffic's defining property: theoretical cache hit rate <strong>>96%</strong>. vLLM turns that reuse into serving efficiency under the same settings as the table.

DeepSeek V4 Pro: about **$28**/hour GB300 TCO versus about **$2,926** on Opus 5 even after applying the cache-read price to every theoretically reusable token.

Numbers as of publication; dashboard is live. Every result links to its InferenceX run.

## Bitter lessons

#### Pipeline parallelism does not fit warm agentic turns

PP, including [chunked pipeline parallelism (CPP)](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dynamic_chunk_pipeline_parallel.html), works on long, fresh prompts. Large prefills keep pipeline stages occupied; throughput can scale nearly linearly with little communication.

Most agentic turns already have system prompts and previous turns cached; each new request may add only a few hundred or a few thousand tokens. Not enough fresh compute to fill the pipeline; bubbles eat the gain.

PP is effective for **cold, compute-heavy prefills**. It should not be the default for **warm, prefix-heavy** turns that dominate agentic sessions.

#### DCP does not transfer cleanly to DeepSeek V4

DCP works well for pure MLA (DeepSeek R1, Kimi K2.5, K2.7) and hybrid MLA (Kimi K3). V4's compressed sparse attention plus highly compressed attention includes an indexer, an extra compressor, and main attention. Context parallelism must partition and coordinate all of those sublayers.

They invested in overlapping communication with compute and in kernels. Even then DCP only **matched** DEP rather than beating it. Parallelism must follow model architecture.

#### Load balance does not guarantee better performance

Aggregated DEP showed substantial KV-usage imbalance across ranks. Balancing by queue depth, running tokens, or current KV utilization **underperformed** simple session-aware sticky routing on AgentX.

Many agentic sessions have short inter-turn delays, so the next turn often arrives while its prefix is still resident. Moving the session to a less-loaded rank forces a KV retrieve even though the prefix lives in the distributed pool. Transfer is async and overlaps compute, but is not free. Prefetched blocks temporarily occupy GPU KV capacity and reduce how many sequences the destination can admit. The queue can look more balanced while processing **fewer** concurrent requests.

For short inter-turn delays, **session locality beats instantaneous load balance**. Routing must account for state already resident, not only queued work.

## Path ahead

Make agentic structure explicit throughout the stack.

Control plane: route first-turn requests (long fresh prefills to fill the prefix cache) separately from turn 2+ (high reuse, short append prefill). Avoid HOL blocking; configure engines and parallelism differently (PCP, CPP) on each side.

Execution / data plane, with the community:

- **Agent hints.** Frameworks could send session structure, branch points, cache positions, tool-call latencies, lifecycle. First step: consume hints through standardized APIs, then guide scheduling and eviction.
- **Programmable KV cache.** Placement, retention, replication, eviction as a user-facing interface (prefetch, eviction, soft-pin).
- **Session-based KV management.** Inter-turn gaps: move retained KV toward the worker likely to serve the next turn. Prefetch during idle to hide transfer latency.

## Acknowledgements

Led by [Inferact](https://inferact.ai/) with the vLLM community. SemiAnalysis for AgentX methodology and reproducibility. NVIDIA and AMD for collaboration.
