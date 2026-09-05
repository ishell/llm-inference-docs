---
source: https://vllm.ai/blog/2025-11-13-shm-ipc-cache
lang: en
fetched: 2026-09-04
---

# Shared Memory IPC Caching: Accelerating Data Transfer in LLM Inference Systems

Chinese: [zh/vllm/blog/serving/shm-ipc.md](../../../../zh/vllm/blog/serving/shm-ipc.md)

2025-11-13. **Donglu Wang (Cohere)**. First posted on the [Cohere blog](https://cohere.com/blog/making-data-transfer-in-llm-systems-faster-leaner-and-more-scalable). Landed via [PR #20452](https://github.com/vllm-project/vllm/pull/20452). Enable with `mm_processor_cache_type = "shm"`; docs: [IPC caching in the User Guide](https://docs.vllm.ai/en/latest/configuration/optimization/#ipc-caching) / local [optimization.md](../../optimization/optimization.md). Study note. Encoder-side cousin that moves the ViT out of the engine: [epd.md](epd.md).

Headline numbers from the page (Command-A Vision, 4×A100, VisionArena-Chat): first-time Prefill **+11.5%**, TTFT **−10.5%**; cached KV+image Prefill **+69.9%**, TTFT **−40.5%**. Gains grow with input size and TP width.

Local figures (copyright remains with the original site; study copies):

![processes1](../../../../assets/vllm/blog/serving/shm-ipc/01-processes1.png)

![shared memory object store](../../../../assets/vllm/blog/serving/shm-ipc/02-shared_memory_object_store.png)

![processes2](../../../../assets/vllm/blog/serving/shm-ipc/03-processes2.png)

## Inter-process communication in LLM inference

A typical multi-process stack has three parts: the **front-end** (handle and preprocess requests), the **coordinator** (scheduling and orchestration), and inference **workers** (the model).

**Figure 1.** Four-GPU example: front-end sends input to the coordinator; the coordinator routes to four workers, one per GPU.

Each stage usually runs in its **own process** so the stack can scale and run asynchronously. Data therefore moves by IPC. For small inputs the tax is noise; as inputs grow, IPC time becomes a bottleneck.

## The problem: repeated large data transfers

Multimodal inputs — images, audio, long context — are large. In [`CohereLabs/command-a-vision-07-2025`](https://huggingface.co/CohereLabs/command-a-vision-07-2025), a single max-size image of **1024×3072** is about **9 MB** as an **int8** array. The model accepts **multiple images**, so a request can reach **tens of megabytes**.

IPC of that size is not free. In multi-turn chat or batching, the **same** inputs may be sent **again**, and the tax compounds.

## The existing solution: mirrored caching

vLLM already used **mirrored caching** to skip redundant IPC. Sender and receiver keep **replicated** caches with the **same insertion order and eviction policy**. On a sender-side hit, it **assumes** the receiver is in the same state and **skips** the transfer.

The catch: **strict input ordering**. Sender and receiver must process inputs in the **exact same sequence**. If mirrored caches sit on the workers, the coordinator may **reorder** for scheduling; the caches desync; behavior can go **wrong**.

So in vLLM, mirrored caching is used only on **front-end ↔ coordinator**. On the coordinator ↔ worker path:

- **Single worker:** vLLM puts it in the **same process** as the coordinator — no extra IPC.
- **Multiple workers:** fallback to **socket** IPC: serialize, transmit, deserialize.

## A new approach: Shared Memory IPC Caching

One shared cache, directly visible to sender and receivers. No ordering assumption. No extra copies of the payload.

### Shared Memory Object Store

A data structure with **one writer** and **many readers** over the **same** memory buffer.

**Design**

- **Writer:** insert the object into a shared **ring buffer**, update an address index, **broadcast the address** to interested readers.
- **Reader:** use that address to read the object **directly** from shared memory.

**Figure 2.** Sender process holds the writer; each receiver holds a reader.

Send path for a key–object pair (serialization/deserialization omitted on the page):

1. `is_cached(key)` — already in the store?
2. Hit: `get_cached(key)` → buffer address.
3. Miss: `put(key, object)` into shared memory → address.
4. Broadcast the **address** over default IPC (small).
5. Receiver: `get(address)` from shared memory.

**Eviction and safety**

When space is low, the writer evicts from the **ring-buffer head**. **Reader counters (shared)** and **writer counters (local)** stop eviction of in-use data. An entry is dropped only when:

```
writer_counter × n_readers == reader_counter
```

**Benefits listed**

- **No ordering assumptions** — processes may consume inputs in any order.
- **Single shared cache** — memory does **not** grow with the number of readers.
- **Efficient concurrent access** — many readers, same input, minimal sync, no extra copies.

Applied to the front-end–coordinator–worker picture: **writer in the front-end**, **one reader in each worker**. Large inputs no longer bounce through the coordinator as a second copy.

**Figure 3.** Same four-GPU coordination, now backed by the Shared-Memory Object Store.

## vLLM benchmark results

Implementation: multimodal inputs, the PR above. Recipe on the page:

- Model: [`CohereLabs/command-a-vision-07-2025`](https://huggingface.co/CohereLabs/command-a-vision-07-2025)
- Hardware: **4× A100 (80GB), TP=4**
- Dataset: [VisionArena-Chat](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat?ref=cohere-ai.ghost.io)

**First-time requests**

| Metric | Baseline | Shared Memory IPC Cache | Difference |
| --- | ---: | ---: | --- |
| Prefill throughput | 581.34 tok/s | 648.22 tok/s | **+11.5%** |
| Mean TTFT | 3898.98 ms | 3491.15 ms | **−10.5%** |

Speedup from writing once in the front-end and letting workers read concurrently — fewer redundant transfers **and** less IPC queueing.

**Cached requests** (both KV and image inputs reused)

| Metric | Baseline | Shared Memory IPC Cache | Difference |
| --- | ---: | ---: | --- |
| Prefill throughput | 2894.03 tok/s | 4917.57 tok/s | **+69.9%** |
| Mean TTFT | 790.18 ms | 470.60 ms | **−40.5%** |

IPC tax is especially visible on this path. Larger inputs and wider TP move more bytes over IPC, so the same cache helps more.

## Get started (then)

Then on vLLM **main**. For multimodal caching:

```
mm_processor_cache_type = "shm"
```

User Guide section linked above. The post also claims the store is useful **beyond** LLM inference wherever IPC caching would skip redundant transfers.

`optimization.md` adds an operational caveat not spelled out in the blog body: API-server scale-out turns this **IPC cache** off (it wants a 1:1 API ↔ engine pairing); the processor cache itself stays.

## Acknowledgments

**Bharat Venkitesh** at Cohere. From the vLLM community: [Cyrus Leung](https://github.com/DarkLight1337) (review and integration); [Nick Hill](https://github.com/njhill) and [Roger Wang](https://github.com/ywang96) (early concept verification); [Kero Liang](https://github.com/imkero) (bug report and fix).
