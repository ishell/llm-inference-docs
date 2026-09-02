---
source: https://vllm.ai/blog/2026-03-24-mrv2
lang: en
fetched: 2026-08-31
---

# Model Runner V2

2026-03-24. Rewrite of the **model runner**, not the whole engine. No API change. Then: `export VLLM_USE_V2_MODEL_RUNNER=1`. Feature gaps below are **v0.18.0**. Figures on the official page.


Local figures (copyright remains with the original site; study copies):

![persistent batch v1](../../../../assets/vllm/blog/architecture/mrv2/01-persistent_batch_v1.png)

![persistent batch mrv2](../../../../assets/vllm/blog/architecture/mrv2/02-persistent_batch_mrv2.png)

![async scheduling](../../../../assets/vllm/blog/architecture/mrv2/03-async_scheduling.png)

![async spec decoding](../../../../assets/vllm/blog/architecture/mrv2/04-async_spec_decoding.png)

![throughput comparison](../../../../assets/vllm/blog/architecture/mrv2/05-throughput_comparison.png)

![tpot mtp](../../../../assets/vllm/blog/architecture/mrv2/06-tpot_mtp.png)

## Why

After V1, async scheduling and spec decode piled onto the runner. Recurring pain: persistent batch coupled to per-step inputs; async retrofitted; CPU bookkeeping; hard to extend.

Principles: **modular** (isolate model-specific logic), **GPU-native** (bookkeeping on device), **async-first** (overlap as a constraint).

## What changed

**Persistent batch.** V1 used persistent state *as* model/sampler inputs. MRV2: each live request owns a **stable row** in a fixed state table; gather per-step tensors in current order. Drops `CachedRequestState`. Triton kernels build `input_ids`, `positions`, `query_start_loc`, `seq_lens` on GPU. Rejection-sampling results can stay on device.

**Async.** Prepare step N+1 while GPU runs N. Goal: **zero CPU–GPU sync** for supported combos. Outputs on a side CUDA stream. Same path for spec decode + structured outputs.

**Triton sampler.** Gumbel-Max without materializing softmax; top-k logits then logprobs; chunked prompt logprobs; `idx_mapping` instead of expanding state per logits vector.

**`ModelState`.** Model-specific add/remove/MM embeddings/prepare inputs/attention/dummy inputs. Old `gpu_model_runner.py` >6700 lines; largest MRV2 file <1300.

## Numbers (host-bound stress)

Qwen3-0.6B on 1×GB200: **16K → 25K** output tok/s (**+56%**).

GLM-4.7-FP8, MTP=1, 4×GB200: mean TPOT **−6.3%** (no sync points with spec decode on).

## Not in v0.18.0

Linear attention (Qwen3.5, Nemotron 3 Super); spec decode beyond Eagle/Eagle3/MTP; EPLB, DBO; logits processors; LoRA. Design-doc page 2 for the rest. Features are re-thought, not copied, when they land.
