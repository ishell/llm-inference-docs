---
source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism
lang: en
fetched: 2026-09-04
---

# Efficient Decode Context Parallelism with vLLM for Long Context Workloads

Chinese: [zh/vllm/blog/performance/dcp.md](../../../../zh/vllm/blog/performance/dcp.md)

2026-08-07. vLLM had supported DCP for almost a year; the post writes it down because agents pushed context to **64K–1M**. CLI: `--decode-context-parallel-size` (also `-dcp` in serve help). Sibling idea in TensorRT-LLM: [Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md). Docs: [Decode Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel).

Local figures (copyright remains with the original site; study copies):

![kv parallelism overview](../../../../assets/vllm/blog/performance/dcp/01-kv-parallelism-overview.svg)

![figure 1](../../../../assets/vllm/blog/performance/dcp/02-figure-1.png)

![figure 2](../../../../assets/vllm/blog/performance/dcp/03-figure-2.png)

![figure 3](../../../../assets/vllm/blog/performance/dcp/04-figure-3.png)

![figure 4](../../../../assets/vllm/blog/performance/dcp/05-figure-4.png)

![figure 5](../../../../assets/vllm/blog/performance/dcp/06-figure-5.png)

## 1. Why TP runs out of room

Long-context agents (repos, long chats) make KV huge. Baseline **tensor parallel** shards KV **by attention head**. That floor is real:

- **GQA:** few KV heads. TP can split only down to **one KV head per GPU**. Beyond that, the cache **duplicates**.
- **MLA:** Key/Value compressed into one low-rank **latent** shared by all query heads — effectively **one** KV head. Under ordinary TP there is nothing to split; the latent is **fully replicated on every TP rank**.

Duplicated KV eats HBM, concurrency dies, throughput and cost/token get worse. DCP shards KV **by sequence** so each GPU stores and reads only a slice. Needs a fast GPU interconnect to keep interactivity while many long agents share the box. Figure 1: overview.

## 2. Results (demo)

Same GPUs, model, workload; only Decode-time KV sharding changes. Figures 1–2 on the original are the comparison plots (local `02-figure-1.png`, `03-figure-2.png`).

### 2.1 Dataset

Public agentic long-context trace in **Mooncake-trace** format: [JSONL](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl) ([dataset notes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/README.md#dataset)). Each line: `input_length`, `output_length`, `hash_ids`. Replay with a Mooncake-compatible harness (e.g. `aiperf --custom-dataset-type mooncake_trace`). `hash_ids` encode shared prefix blocks — useful for prefix-cache / KV-reuse benches.

Shape: long in, short out. Median input ~**67K**, output ~**400**. **Bimodal**, not uniformly huge:

- ~**53%** at **64K+** (heavy tail ~**1M**)
- ~**47%** under 64K; ~**18%** under 8K
- ~**8%** exceed 128K; ~**3–4%** exceed 256K

### 2.2 Benefit

Single **8×B200** node, **Kimi K2.6 NVFP4**, vLLM. Concurrency sweep **16 → 512**. The original says “see table below”; the live page’s table is a JS widget and did not appear in the fetched markdown. Prose numbers:

- Baseline TP: KV **100%** at concurrency **64**, throughput plateaus near **1,863 tok/s/GPU** — no more requests fit.
- DCP: sequence-shards KV (1/N per GPU). At concurrency **512**, still ~**82%** KV, **6,091 tok/s/GPU**.

Figure 3. Core claim: DCP keeps scaling concurrency on long-context runs, the regime where replicated-KV TP OOMs first.

### 2.3 By sequence length

Figure 4: one throughput–interactivity Pareto, requests in five bands: **&lt;32K**, **32–64K**, **64–128K**, **128–200K**, **200K+**. DCP stays on a high, stable frontier even in **200K+**; short and long buckets nearly overlap. Replicated TP cannot scale there.

## 3. The TP head floor (again, with the mechanism)

Each KV head owns its K/V tensors; the head is the smallest unit TP can hand to a GPU. Standard TP cannot slice **one head’s** KV. With K KV heads you can give each GPU a distinct subset **until every GPU holds one head**. Past K, two GPUs hold a **copy** of the same head.

## 4. What DCP is

Split KV by **token positions** of the same sequence. Example: one **200K** request, four GPUs → 0–50K / 50K–100K / 100K–150K / 150K–200K. Per-GPU footprint shrinks as you add GPUs; batch size can rise. Figure 5.

### 4.1 Rhythm

**AllGather Q → Compute → AllGather + ReduceScatter.**

- **AllGather Q.** Each GPU has only a fragment of Q; attention needs the full query against any key. All-gather across the DCP group. Cheap in Decode: Q is **one token**. MLA opt-in: [PR #45964](https://github.com/vllm-project/vllm/pull/45964) replicates the small query projection inside the DCP group at **load** time so Decode **skips** this all-gather (`VLLM_DCP_Q_REPLICATE=1`).
- **Compute.** Attention between gathered Q and the **local** KV slice. vLLM: `k_up` for MLA, `tensor_broadcast` for GQA.
- **AllGather + ReduceScatter (`cp_lse_ag_out_rs`).** Share partial output + LSE; LSE reweights/merges (online softmax); ReduceScatter sums and returns each GPU only its own head-slice.

## 5. Usage

`decode_context_parallel_size` beside existing TP.

### 5.1 Offline

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
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

### 5.3 MLA backend

**Models:** DeepSeek-V2 / V3 / R1, Kimi K2.6 (MLA). The whole latent is redundant under TP, so the whole cache can be sequence-split. At attention each rank **up-projects** its latent slice (`k_up`) to reconstruct K/V. Effective KV-head count is 1 → split up to full TP:

- `tensor_parallel_size >= decode_context_parallel_size`
- `tensor_parallel_size % decode_context_parallel_size == 0`

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8
```

### 5.4 GQA backend

**Examples:** Qwen3-235B, Llama-family, other GQA. TP splits by `num_key_value_heads` first. Clean only up to that count; beyond it, `tp // num_key_value_heads` identical copies. DCP fills those copies with **different sequence chunks**; shared KV heads broadcast across query heads (`tensor_broadcast` for GQA). Sequence-split degree is capped by the duplication factor:

- `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
- `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`

```bash
# Qwen3-235B num_key_value_heads = 4; tp=8 → 8//4 = 2 copies → DCP ≤ 2
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2
```

## 6. Future work (then)

Finer TP/DCP sizes; better DCP **A2A** kernels (multi- and single-node); **MTP / speculative decoding** without giving up spec latency; harden **P/D** disaggregation; hybrid models and Dynamic Chunked Pipeline Parallelism; more backends. Community: **GLM-5.2**, **Kimi K3**. Longer roadmap: **Prefill Context Parallelism (PCP)**. Kimi K3 DCP benches were still in progress at posting.

Conclusion language: shard the sequence during attention, then reconfigure the same GPUs to amortize FFN weight loading across the pool — scale with context length instead of degrading under it.

## About / repro

NVIDIA reviews and benches: Anahita Bhiwandiwalla, Xin Li, Pavani Majety, Nidhi Bhatia, Roman Ageev, Pen Chung Li, Chris Hoge. Initial DCP upstream: Moonshot AI, [vLLM #23734](https://github.com/vllm-project/vllm/pull/23734). Follow-up: [Lucas Wilkinson](https://github.com/LucasWilkinson). Measured on **NVIDIA B200**, Kimi K2.6 **NVFP4**; recipes on vLLM builds that support `--decode-context-parallel-size`.

Long-context map: TP shards heads, DCP shards sequence, Mooncake pools prefixes, P/D splits Prefill from Decode. They are not mutually exclusive.
