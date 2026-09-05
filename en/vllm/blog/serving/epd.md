---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: en
fetched: 2026-09-05
---

# Encoder Disaggregation for Scalable Multimodal Model Serving

Chinese: [zh/vllm/blog/serving/epd.md](../../../../zh/vllm/blog/serving/epd.md)  
Source: https://vllm.ai/blog/2025-12-15-vllm-epd

2025-12-15. **Multimodality Workstream @ vLLM.** EPD here is **Encoder vs Prefill/Decode**, not the text P/D in [router.md](router.md). Two different “disaggregations.” Native [PR #25233](https://github.com/vllm-project/vllm/pull/25233), merged early November 2025, in **v0.11.1**. NVIDIA Dynamo had an EPD-shaped split with vLLM first (docs were thin). Single-node cousin: `mm_encoder_tp_mode="data"` (ViT DP + LM TP).

Local figures (copyright remains with the original site; study copies).

## Motivation: Why Disaggregate the Encoder in LMM Serving?

Modern LMMs add a serving bottleneck: **before any text generation, all images must go through a visual encoder (e.g. ViT)**. That stage has a different compute profile from text Prefill and Decode. Colocating encoder + Prefill + Decode on the *same* GPU — today’s common approach — creates structural waste.

### Problems With Colocating Encoder and Text Generation

**1. Encoder–Prefill–Decode Interference**

Current pipeline (E+PD on the same GPU):

```
[E PD] -> [E PD] -> [E PD]
```

All requests must finish *both* stages before the next can proceed. Encoder work cannot overlap Prefill/Decode for other requests.

Effects:

- Encoder is slow and variable (resolution, image count, complexity).
- Mixed with text-only requests, a single LMM input can stall the entire batch.
- Prefill and streaming Decode become jittery.
- Compute-bound encoder and memory-bound Decode share hardware and one parallelism plan.

**2. Coupled and Inefficient Resource Allocation**

Three phases, three optimal profiles:

- **Encoder:** one-shot, compute-bound, high parallelism.
- **Prefill:** high memory bandwidth, large GEMMs.
- **Decode:** heavily memory-bound, long-lived, sequential.

Colocation welds one plan and one resource ratio:

- You cannot scale encoder throughput without overprovisioning text-generation GPUs.
- Occasional multimodal requests create outsized cost.

## Solutions: Encoder Disaggregation

A separate, scalable encoder service.

### 1. Pipelined Execution and Elimination of Interference

```
E → P D   (Request 1)
......E → P D   (Request 2)
..........E → P D   (Request 3)
```

- Encoder for request N can run while N–1 is already in Prefill or Decode.
- Text-only requests **bypass** the encoder and never wait behind image jobs.
- Encoder-induced queueing goes away.
- The system is pipeline-parallel: more throughput, smoother latency.

### 2. Independent, Fine-Grained Scaling

Each stage follows its own demand curve:

- **Encoder GPUs** follow multimodal image volume.
- **Prefill/Decode GPUs** follow request rate and output length.

No more buying a fat Decode cluster for rare image spikes. Each pool uses the right hardware and parallelism.

### 3. Encoder Output Caching and Reuse

A centralized encoder service naturally caches embeddings across requests:

- Frequent images (logos, diagrams, product shots) computed once, reused across users.
- Hits have **zero encoder cost**, which cuts TTFT.
- Encoder load falls as hit rate grows.

## Design

![EPD Architecture](../../../../assets/vllm/blog/serving/epd/01-image.png)

**Caption.** EPD Architecture.

### Components

**Proxy & Router**

- Orchestrates request flow.
- Sends multimodal (MM) inputs to encoder instances.
- Waits for encoder completion, then forwards the original request (embeddings now in remote storage) to Prefill/Decode (PD) instances.

**Data Transfer Layer**

- Remote storage for encoder-produced multimodal embeddings (Encoder Cache, or EC).
- Shared transport between encoder workers and PD workers.

**EC Connectors**

- Bridge workers/schedulers to that layer.
- Store and retrieve encoder caches.

Roles:

- **Scheduler-side connector:** which multimedia embeddings to load or save this scheduling iteration; metadata for downstream workers.
- **Worker-side connector:** actual read/write to remote storage; per-worker embedding transfers.

## Workflow

### Dataflow Graph

![EPD Dataflow Graph](../../../../assets/vllm/blog/serving/epd/02-workflow.png)

**Caption.** EPD Dataflow Graph.

### Request Lifecycle

1. **Proxy receives request.** Extracts multimodal inputs. Creates **N encoder jobs** (one per MM input), dispatches to encoder instances.
2. **Encoder scheduling.** Encoder scheduler runs the jobs, writes embeddings to remote storage via EC connectors.
3. **Encoder completion.** Encoder workers notify the proxy when all embeddings are stored.
4. **Proxy forwards request to PD instance.** Original request with **image hashes, no pixel data**.
5. **PD execution.** PD loads MM embeddings from remote storage via EC connectors, injects them into the model runner cache, runs Prefill and Decode as usual.

## Implementation

### Core Components

#### 1. `ECConnectorRole`

Where the connector instance runs:

```python
class ECConnectorRole(enum.Enum):
    SCHEDULER = 0   # in scheduler process
    WORKER = 1      # in worker process
```

#### 2. `ECConnectorMetadata`

Abstract sync/state object shared between scheduler-side and worker-side connectors:

```python
class ECConnectorMetadata(ABC):
    pass
```

#### 3. `ECConnectorBase`

Abstract interface for all connectors.

Fields: `role`, `config`, `metadata`.

Methods:

- `has_caches(request)`: remote embeddings already exist?
- `build_connector_meta(sched_output)`: which caches workers must load
- `update_state_after_alloc(request, item)`: update allocation on hit/miss
- `save_caches(encoder_cache)`: push encoder outputs to remote storage
- `start_load_caches(metadata)`: load on the PD side before Prefill/Decode

Cousin of the text **KVConnector**: do not recompute intermediate state across machines.

## Scheduler-Side Behavior

### 1. Connector Initialization

Scheduler:

```python
if self.vllm_config.ec_transfer_config is not None:
    self.ec_connector = ECConnectorFactory.create_connector(
        config=self.vllm_config,
        role=ECConnectorRole.SCHEDULER,
    )
```

Worker:

```python
def ensure_ec_transfer_initialized(vllm_config):
    global _EC_CONNECTOR_AGENT
    if vllm_config.ec_transfer_config is None:
        return
    if vllm_config.ec_transfer_config.is_ec_transfer_instance and _EC_CONNECTOR_AGENT is None:
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
            config=vllm_config,
            role=ECConnectorRole.WORKER,
        )
```

### 2. Remote Cache Check

When scheduling media items:

```python
remote_cache_has_item = self.ec_connector.has_caches(request)
```

### 3. Cache State Updates

After scheduling:

```python
for i in external_load_encoder_input:
    self.encoder_cache_manager.allocate(request, i)
    if self.ec_connector:
        self.ec_connector.update_state_after_alloc(request, i)
```

### 4. Metadata Construction

End of a scheduler iteration:

```python
ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
scheduler_output.ec_connector_metadata = ec_meta
```

## Worker-Side Behavior

Workers use `ECConnectorModelRunnerMixin` to fold connector operations into GPU model runners.

## Execution Integration

### Encoder Side (Saving to Remote Storage)

After computing embeddings:

```python
for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
    self.encoder_cache[mm_hash] = scatter_mm_placeholders(...)
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

### Prefill/Decode Side (Loading Remote Embeddings)

Wrap the media encoder path with a loader that injects cached embeddings before the local encoder runs:

```python
with self.maybe_get_ec_connector_output(
        scheduler_output,
        encoder_cache=self.encoder_cache,
    ) as ec_connector_output:

    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

## Performance Results

**Environment:** 4×A100 80G  
**Dataset:** `vllm bench serve --dataset-name random-mm`  
**Inputs:** 400 / 2000 text tokens; 1–4 images per request (640×640 → ~**400** visual tokens each)  
**Outputs:** 150 tokens  
**QPS range:** 4–24  
**Model:** Qwen3-VL-4B-Instruct  
**Baseline:** 1 Encoder + 3 PD (**1E3PD**) vs Data Parallel (`--data-parallel-size 4`)

Production LMM serving wants tail guarantees — typically **P99 TTFT** and **P99 TPOT**. **Goodput** = max sustainable request rate at which both SLOs hold (**20000 ms** TTFT, **100 ms** TPOT in this evaluation).

## Short-Text Workloads (~400 tokens)

![Short-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/03-plot_len400_epd_vs_non_epd.png)

**Caption.** Short-Text Workloads Performance.

Benefits grow with images per request.

- **Single-image:** modest goodput (23 → 24 QPS).
- **Four-image:** goodput **doubles** (6 → 12 QPS).

Tail latency: P99 TTFT/TPOT often **20–50%** lower than non-EPD.

Throughput-versus-rate:

- Without EPD, multi-image destabilizes around **12–14 QPS**; P99 TPOT spikes **30–50%**, SLO broken.
- EPD pushes that cliff out; latency curves grow slower — encoder/Decode no longer share a queue; text-only bypasses vision.

## Long-Text Workloads (~2000 tokens)

![Long-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/04-plot_len2000_epd_vs_non_epd.png)

**Caption.** Long-Text Workloads Performance.

Longer inputs: image encode is a small fraction; Decode-dominated. Still substantial gains.

Baseline sustainable QPS before P99 violations:

- 1 image: **8 QPS**
- 3–4 images: **4 QPS**

EPD holds:

- **18 / 11 / 9 / 8 QPS** — **2× to 2.5×** goodput.

Also:

- Effective decoding throughput **+10–30%** across multimodal settings.
- P99 TTFT **−30–50%**.
- P99 TPOT **−20–40%** inside stable regions.

Decoupled Encode/Text pipeline removes modal contention: higher concurrency, more throughput, tighter SLOs.

## Hardware Portability: Ascend NPU

Same experiments on Ascend NPUs, minimal changes:

- **Environment:** 4×Ascend 910B 32G
- **Model:** Qwen2.5-VL-7B-Instruct
- **QPS:** 1–10

![NPU Short-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/05-npu_plot_len400_epd_vs_non_epd.png)

**Caption.** NPU Short-Text Workloads Performance.

![NPU Long-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/06-npu_plot_len2000_epd_vs_non_epd.png)

**Caption.** NPU Long-Text Workloads Performance.

Across Ascend runs, the **same hardware-agnostic benefits**:

- Consistently higher throughput (**5–20%** in stable regions).
- Significant P99 TTFT and P99 TPOT reductions.
- Delayed congestion, tighter tails.

Gains from architectural decoupling, not a vendor GPU’s temperament — portable across GPU and NPU.

## Conclusion

A **decoupled, pipeline-parallel multimodal serving architecture** that:

- reduces TTFT and TPOT,
- improves throughput and stability,
- eliminates cross-modal interference, and
- enables efficient, scalable multimodal serving.

Follow-ons named then: [encoder parameter loading](https://github.com/vllm-project/vllm/pull/30242), [more EC connectors](https://github.com/vllm-project/vllm/pull/30468).

## Related Work

### ViT DP + LM TP

Before cluster EPD, vLLM shipped [ViT Data Parallel + LLM Tensor Parallel](https://github.com/vllm-project/vllm/issues/22743) on one node: vision encoder DP across GPUs, language model TP. Cuts TTFT, raises throughput. Adopted elsewhere, e.g. [SGLang](https://github.com/sgl-project/sglang/pull/13126).

### Prior Art and Industry Adoption

NVIDIA Dynamo first supported [EPD-style disaggregation](https://github.com/ai-dynamo/dynamo/blob/44a2cba976d12a79b2164ed11612c1bc7491a3d8/examples/backends/vllm/launch/agg_multimodal_epd.sh#L5) with vLLM; docs were limited. Native vLLM EPD ([PR #25233](https://github.com/vllm-project/vllm/pull/25233)) merged early November 2025, available since **0.11.1**.

## Reference

- Qiu, Haoran, et al. *ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving*. 2025.
- Singh, G., et al. *Efficiently Serving Large Multimodal Models Using Encoder-Decoder Disaggregation*. 2025.

## Acknowledgments

Main contributors: ZHENG Chenguang, Nguyen Kha Nhat Long, Tai Ho Chiu Hero, Le Manh Khuong, Wu Hang, Wu Haiyan. Maintainers: Roger Wang, Nicolò Lucchesi, Cyrus Leung.

Router owns text P/D; EPD owns “the image goes to another building first.” [large-scale.md](large-scale.md) welds text P/D to Wide-EP.
