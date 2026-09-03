---
source: https://vllm.ai/blog/2026-08-22-rdt-weight-transfer
lang: en
fetched: 2026-09-01
---

# Sharded weight transfer with RDT: Kimi K2 in 7.53s

Chinese: [zh/vllm/blog/serving/rdt-weight-transfer.md](../../../../zh/vllm/blog/serving/rdt-weight-transfer.md)  
Kimi K2 BF16: 48×(8×H100), 32 trainer + 16 inference nodes, **7.53 s** for ~**7.9 TB**, ~**1049 GB/s** aggregate.

NCCL broadcast ships full HF weights to every inference rank: TP8 keeps ⅛; wide EP wastes more; collectives stall on stragglers. RDT (NIXL) lets each inference rank **pull only its shard**.


Local figures (copyright remains with the original site; study copies):

![rdt blog overview](../../../../assets/vllm/blog/serving/rdt-weight-transfer/01-rdt_blog_overview.png)

![layerwise reloading](../../../../assets/vllm/blog/serving/rdt-weight-transfer/02-layerwise_reloading.webp)

![rdt blog init flow](../../../../assets/vllm/blog/serving/rdt-weight-transfer/03-rdt_blog_init_flow.png)

![AllScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/04-AllScenes.gif)

![ExpertScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/05-ExpertScenes.gif)

![rdt pipelined execution 2x](../../../../assets/vllm/blog/serving/rdt-weight-transfer/06-rdt_pipelined_execution-2x.png)

![rdt qwen weight sync latencies](../../../../assets/vllm/blog/serving/rdt-weight-transfer/07-rdt_qwen_weight_sync_latencies.png)

![rdt fault tolerance](../../../../assets/vllm/blog/serving/rdt-weight-transfer/08-rdt_fault_tolerance.png)

## Recording tensor

vLLM loaders fuse, transpose, split GQA Q vs KV differently, unpack Llama-4 experts — trainers cannot hand-code that. At init, a **recording tensor** (shape/dtype, no data) logs view/narrow/transpose into a sharding plan. Trainers slice BF16 shards from the plan; inference runs process + copy into live weights so CUDA graphs survive.

SkyRL Qwen3-235B: NCCL 64.72 s → sharded V1 25 s → PP/EP-local V2 5.61 s → pipelined V3 **3.49 s**. A Kimi MoE layer is ~30 GB BF16 — do not gather all experts per GPU.

SkyRL: `generator.inference_engine.weight_sync_backend=sharded_rdt` and `colocate_all=false`. Other frameworks implement a `WeightSource` iterator.

Limits then: loaders must be recordable; RDT buffers sit **outside** `gpu_memory_utilization`; **not** with EPLB; serial across trainer PP to avoid layerwise OOM. Dead inference replicas: NIXL still syncs the live set — a broadcast collective cannot. Read with [Native RL](native-rl.md): that post is pause/keep; this one is **how weights move**.
