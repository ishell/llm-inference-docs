---
source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism
lang: en
fetched: 2026-09-05
---

# Efficient Decode Context Parallelism with vLLM for Long Context Workloads

Chinese: [zh/vllm/blog/performance/dcp.md](../../../../zh/vllm/blog/performance/dcp.md)  
Source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism

2026-08-07. **Seonghee Lee, Sungsoo Ha, Omri Almog (NVIDIA), Lucas Wilkinson (Red Hat AI).** vLLM had supported DCP for almost a year; the post writes it down because agents pushed context to **64K–1M**. CLI: `--decode-context-parallel-size` (also `-dcp` in serve help). Sibling idea in TensorRT-LLM: [Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md). Docs: [Decode Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel). Summary claim: **~3×** throughput vs standard TP on long-context agentic workloads.

Local figures (copyright remains with the original site; study copies). Section 2.2 says “see table below”; the live-page table is a JS/Plotly widget and is omitted. Prose numbers are kept.

## 1. Introduction

Long-context inference is becoming essential for agentic AI: assistants reason over large repos and long chats. Agent-trace benchmarks now run **64K–1M** tokens; KV caches grow with them. Baseline **tensor-parallel (TP)** shards KV **by attention head** — a hard floor.

Both modern attention schemes hit that floor.

- **Grouped-query attention (GQA):** few KV heads. TP can split only down to **one KV head per GPU**. Once `tensor_parallel_size` exceeds the KV-head count, the cache **duplicates**.
- **Multi-head latent attention (MLA):** Key/Value compressed into one low-rank **latent** shared by all query heads — effectively **one** KV head. Under ordinary TP there is nothing to split; the latent is **fully replicated on every TP rank**.

Duplicated KV eats HBM, concurrency dies, throughput and cost/token get worse. Cousin note: [distributed-inference.md](../serving/distributed-inference.md).

Decode Context Parallelism shards KV **by sequence** so each GPU stores and reads only a slice. That frees memory for a larger batch. Needs a fast GPU interconnect to keep interactivity while many long agents share the box.

vLLM has supported DCP for almost a year. This post is written now because long-context agents made the benefit obvious.

![kv parallelism overview](../../../../assets/vllm/blog/performance/dcp/01-kv-parallelism-overview.svg)

**Caption.** Under plain TP both schemes waste memory on duplicated KV: GQA can only split down to one KV head per GPU before replicating; MLA behaves like a single KV head, so its latent is replicated on every rank. DCP shards along the sequence dimension — each GPU holds a unique slice.

## 2. Performance Results

Same GPUs, model, workload; only Decode-time KV sharding changes.

![figure 1](../../../../assets/vllm/blog/performance/dcp/02-figure-1.png)

![figure 2](../../../../assets/vllm/blog/performance/dcp/03-figure-2.png)

### 2.1 Dataset

Public agentic long-context trace in **Mooncake-trace** format: [JSONL](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl) ([dataset notes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/README.md#dataset)). Each line: `input_length`, `output_length`, `hash_ids`. Replay with a Mooncake-compatible harness (e.g. `aiperf --custom-dataset-type mooncake_trace`). `hash_ids` encode shared prefix blocks — useful for prefix-cache / KV-reuse benches.

Shape: long in, short out. Median input ~**67K**, output ~**400**. **Bimodal**, not uniformly huge:

- ~**53%** at **64K+** (heavy tail ~**1M**)
- ~**47%** under 64K; ~**18%** under 8K
- ~**8%** exceed 128K; ~**3–4%** exceed 256K

### 2.2 Benefits of Decode Context Parallelism

Single **8×B200** node, **Kimi K2.6 NVFP4**, vLLM. Concurrency sweep **16 → 512**. The original says “see table below”; that widget is omitted. Across the throughput–interactivity Pareto, DCP sustains far higher concurrency and higher tok/s/GPU.

![figure 3](../../../../assets/vllm/blog/performance/dcp/04-figure-3.png)

Baseline TP replicates KV on every GPU. KV hits **100%** at concurrency **64**; throughput plateaus near **1,863 tok/s/GPU** — no more requests fit. DCP sequence-shards KV (1/N per GPU). At concurrency **512**, still ~**82%** KV, **6,091 tok/s/GPU**.

**Core value:** DCP keeps scaling concurrency on long-context runs, the regime where replicated-KV TP OOMs first.

### 2.3 Comparison by Sequence Length

![figure 4](../../../../assets/vllm/blog/performance/dcp/05-figure-4.png)

One throughput–interactivity Pareto vs full sequence length (input + output). Five bands: **&lt;32K**, **32–64K**, **64–128K**, **128–200K**, **200K+**. DCP stays on a high, stable frontier even in **200K+**; short and long buckets nearly overlap. Replicated TP cannot scale there.

## 3. Challenges of Serving Long Contexts

Under TP, KV is partitioned **by the attention head**. Each KV head owns its K/V tensors; the head is the smallest unit TP can hand to a GPU. Standard TP cannot slice **one head’s** KV. With K KV heads you can give each GPU a distinct subset **until every GPU holds one head**. Past K, two GPUs hold a **copy** of the same head.

## 4. What is DCP?

DCP splits KV by **token positions** of the same sequence. Example: one **200K** request, four GPUs → 0–50K / 50K–100K / 100K–150K / 150K–200K. Per-GPU footprint shrinks as you add GPUs; batch size can rise.

![figure 5](../../../../assets/vllm/blog/performance/dcp/06-figure-5.png)

### 4.1 Decode Context Parallelism Process

Rhythm: **AllGather Q → Compute → AllGather + ReduceScatter**.

- **AllGather Q.** Each GPU has only a fragment of Q; attention needs the full query against any key. All-gather across the DCP group. Cheap in Decode: Q is **one token**. MLA opt-in: [PR #45964](https://github.com/vllm-project/vllm/pull/45964) replicates the small query projection inside the DCP group at **load** time so Decode **skips** this all-gather (`VLLM_DCP_Q_REPLICATE=1`).
- **Compute.** Attention between gathered Q and the **local** KV slice. vLLM: `k_up` for MLA, `tensor_broadcast` for GQA.
- **AllGather + ReduceScatter (`cp_lse_ag_out_rs`).** Share partial output + LSE; LSE reweights/merges (online softmax); ReduceScatter sums and returns each GPU only its own head-slice.

## 5. vLLM Usage

One extra argument: `decode_context_parallel_size`, beside existing TP.

### 5.1 Offline

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
)
outputs = llm.generate(prompts, sampling_params)
```

### 5.2 Online

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

### 5.3 MLA Backend

**Models:** DeepSeek-V2 / V3 / R1, Kimi K2.6 (MLA).

**Why it's different.** MLA compresses Key/Value into one low-rank latent shared across query heads — effectively one KV head. Under pure TP there is nothing to split; the latent is replicated on every TP rank. TP does not shrink it, so MLA is the ideal DCP candidate: the whole cache is redundant and can be sequence-split.

**What they do.** DCP splits the latent along the sequence dimension; at attention each rank **up-projects** its latent slice (`k_up`) to reconstruct K/V. Effective KV-head count is 1 → split up to full TP:

- `tensor_parallel_size >= decode_context_parallel_size`
- `tensor_parallel_size % decode_context_parallel_size == 0`

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8
```

### 5.4 GQA Backend

**Example models:** Qwen3-235B, Llama-family, other GQA.

**Why it's different.** GQA stores `num_key_value_heads` KV heads; TP splits by those heads first. Clean only up to that count; beyond it, `tp // num_key_value_heads` identical copies.

**What they do.** DCP fills those copies with **different sequence chunks**; shared KV heads broadcast across query heads (`tensor_broadcast` for GQA). Sequence-split degree is capped by the duplication factor:

- `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
- `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`

```python
# Qwen3-235B has num_key_value_heads = 4; tp=8 gives 8//4 = 2 redundant copies,
# so dcp can be up to 2.
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2
```

## 6. Future Work

Finer TP/DCP sizes; better DCP **A2A** kernels (multi- and single-node); **MTP / speculative decoding** without giving up spec latency; harden **P/D** disaggregation; hybrid models and Dynamic Chunked Pipeline Parallelism; more backends. Community: **GLM-5.2**, **Kimi K3**. Longer roadmap: **Prefill Context Parallelism (PCP)**. Kimi K3 DCP benches were still in progress. Docs: [Decode Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel).

## 7. Conclusion

DCP rethinks how GPUs are organized for long-context inference. Shard the sequence during attention, then reconfigure the same GPUs to amortize FFN weight loading — scale with context length instead of degrading under it.

Native in vLLM. Same industry direction as NVIDIA [Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md) in TensorRT-LLM. Kimi K3 DCP benches were still in progress at posting.

## About Us

NVIDIA reviews and benches: Anahita Bhiwandiwalla, Xin Li, Pavani Majety, Nidhi Bhatia, Roman Ageev, Pen Chung Li, Chris Hoge. Initial DCP upstream: [Moonshot AI](https://www.moonshot.cn/), [vLLM #23734](https://github.com/vllm-project/vllm/pull/23734). Follow-up: [Lucas Wilkinson](https://github.com/LucasWilkinson). Measured on **NVIDIA B200**, Kimi K2.6 **NVFP4**; recipes on vLLM builds that support `--decode-context-parallel-size`.

Long-context map: TP shards heads, DCP shards sequence, Mooncake pools prefixes, P/D splits Prefill from Decode. They are not mutually exclusive.
