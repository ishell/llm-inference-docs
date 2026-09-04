---
source: https://vllm.ai/blog/2026-03-24-mrv2
lang: en
fetched: 2026-09-04
---

# Model Runner V2

Chinese: [zh/vllm/blog/architecture/mrv2.md](../../../../zh/vllm/blog/architecture/mrv2.md)

2026-03-24. A ground-up re-implementation of the **model runner**, not the whole engine. **No user-facing API change.** At posting it was opt-in, not default:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

They planned to make it default soon after. Feature gaps below are **as of v0.18.0** and will age. Install the latest build, set the env var, keep using the Python API or `vllm serve`.

Like V1, this is an architectural upgrade from production lessons. They revisited persistent batching, async scheduling, input preparation, and sampling, then rebuilt the runner around three principles:

- **Be modular.** Isolate model-specific logic from the common execution path.
- **Be GPU-native.** Move bookkeeping off the CPU and onto the GPU.
- **Be async-first.** Treat overlapped CPU/GPU execution as a design constraint, not a retrofit.

Local figures (copyright remains with the original site; study copies):

![persistent batch v1](../../../../assets/vllm/blog/architecture/mrv2/01-persistent_batch_v1.png)

![persistent batch mrv2](../../../../assets/vllm/blog/architecture/mrv2/02-persistent_batch_mrv2.png)

![async scheduling](../../../../assets/vllm/blog/architecture/mrv2/03-async_scheduling.png)

![async spec decoding](../../../../assets/vllm/blog/architecture/mrv2/04-async_spec_decoding.png)

![throughput comparison](../../../../assets/vllm/blog/architecture/mrv2/05-throughput_comparison.png)

![tpot mtp](../../../../assets/vllm/blog/architecture/mrv2/06-tpot_mtp.png)

## Why Model Runner V2?

After V1 shipped, the runner accumulated debt as features landed incrementally. Useful in isolation; harder to reason about once **async scheduling** and **speculative decoding** sat in the middle of the execution model.

Recurring pain:

- **Tangled persistent batch state.** Persistent state was tightly coupled to per-step model inputs. Insertions, removals, and reordering were more complex than they needed to be.
- **Fragile async execution.** Async was retrofitted onto the V1 runner. Many features needed unnatural logic to coexist with it.
- **CPU-bound bookkeeping.** Input preparation and sampling were many small CPU ops. GPUs kept getting faster; those ops started to show.
- **Difficult extensibility.** New models and features were hard to extend cleanly.

## What’s new

### 1. Persistent batch + GPU-native input prep

vLLM does a lot of bookkeeping for batching, paged attention, sampling parameters. Historically that was CPU-side.

V1 already had **persistent batching**: consecutive batches are usually similar, so incremental updates beat rebuilding large tensors every step. But V1 used persistent state *directly* as model and sampler inputs — awkward layout constraints.

Figure 1: V1 persistent batch. Request order is glued to block-table layout; add/remove means a complex reorder.

**MRV2 decouples persistent request state from per-step input tensors.** Each live request owns a **stable row** in a fixed-size state table for its active lifetime. Each step, the runner **gathers** step-specific inputs from that table in current request order. Incremental updates stay; a large class of state-management complexity goes. Redundant backup state such as `CachedRequestState` goes away — active requests no longer depend on fragile tensor-wide reordering.

Figure 2: MRV2. Stable state table independent of per-step layout; a gather produces the ordered input block table.

Input preparation moves to the **GPU** with **Triton** kernels. Request state stays largely on device. `input_ids`, `positions`, `query_start_loc`, `seq_lens` are built on GPU. Three concrete wins:

- **Lower CPU overhead** — less Python and CPU tensor wrangling.
- **Lower code complexity** — no more constraints imposed by CPU-side tensor ops.
- **Better async + spec decode** — GPU-resident prep can consume device-side rejection-sampling results **without synchronization**.

### 2. Async-first

Async scheduling is now fundamental: scheduler and worker prepare step **N+1** while the GPU runs step **N**. V1 already supported this; it was a retrofit.

Figure 3: V1 async — CPU schedules/prepares the next step while the GPU executes the current one.

MRV2 treats async as a core assumption and aims for **zero CPU–GPU synchronization** across all *supported* model and feature combinations.

The combination that was ugly in V1 — async scheduling **together with** speculative decoding — falls out: prep kernels consume rejection-sampling results on device. Step outputs go to the CPU on a **side CUDA stream**, decoupled from the main compute stream. The same path covers spec decode + **structured outputs**.

Figure 4: MRV2 async + spec decode. GPU-side prep eats rejection results directly; no CPU–GPU sync points.

### 3. Triton-native sampler

Reworked sampling, Triton kernels, more control over memory and numerics:

- **Gumbel-Max** kernel — no explicit softmax materialization; **stateless in-kernel RNG**.
- **Top-k logprobs** — find top-k logits first, compute logprobs only for those candidates.
- **Prompt logprobs** — finer chunking, including **within a single prompt**.
- **Spec decode** — indirection via `idx_mapping` inside kernels instead of expanding request state to match every logits vector.

Peak memory drops; rich combinations of sampling parameters get easier.

### 4. `ModelState` (modularization)

vLLM supports many architectures; the old runner absorbed that complexity. MRV2 introduces **`ModelState`**:

```python
class ModelState(ABC):
    def add_request(self, ...): ...
    def remove_request(self, ...): ...
    def get_mm_embeddings(self, ...): ...
    def prepare_inputs(self, ...): ...
    def prepare_attn(self, ...): ...
    def prepare_dummy_inputs(self, ...): ...
    # multimodal embeddings, extra model inputs,
    # attention metadata, CUDA graph capture, ...
```

Model-specific logic lives here so the main runner stays on the common path. The complaint this answers: contributors who only care about DeepSeek, Qwen, Kimi, or a private internal model should not have to read the whole maze.

Files split too. Old `gpu_model_runner.py` **>6,700** lines; largest MRV2 file **<1,300**.

## Performance (host-bound stress)

Not just cleanup. They picked a **tiny model on a fast GPU** so host overhead is a large fraction of the step.

**Qwen3-0.6B on 1×GB200:** **16K → 25K** output tok/s (**+56.2%**), from offloading input prep to GPU. Figure 5.

**Speculative decoding:** `GLM-4.7-FP8`, **MTP=1**, **4×GB200**: mean TPOT **−6.3%** across request rates. Comes from **zero sync points when spec decode is on**. Figure 6.

They expect the foundation to matter more as stacks combine async scheduling, spec decode, multimodal preprocess, and heterogeneous model state.

## Not in v0.18.0 (experimental)

MRV2 was still experimental. Not supported then:

- Linear attention (Qwen3.5, Nemotron 3 Super)
- Spec decoding methods other than **Eagle / Eagle3 / MTP**
- **EPLB** and **DBO**
- Logits processors
- **LoRA**

Full list: page 2 of the [design doc](https://docs.google.com/document/d/1gFqtDkcoqhy9j-X0ndshzbhapX1uNey1-wBENwGPI80/edit?usp=sharing).

Quality bar: when a V1 feature moves into MRV2, they want it **rethought from first principles**, not copied. Landings that touch MRV2 may be slower than usual — that is the point of tearing it down.

## Getting started

1. Latest vLLM build.
2. `export VLLM_USE_V2_MODEL_RUNNER=1`.
3. Existing APIs — Python or `vllm serve`. No API changes.

## Acknowledgments

Woosuk Kwon, Nick Hill, Giancarlo Delfin, Santino Ramos (Inferact); Wentao Ye, Zhanqiu Hu, Lucas Wilkinson (Red Hat); Haoran Zhu (Alibaba).

Read after V1 / [anatomy](../architecture/anatomy.md): once the engine skeleton stands, the next bottleneck is how each step lays the batch onto the GPU. Anatomy’s model runner is the board this rewrite replaces.
