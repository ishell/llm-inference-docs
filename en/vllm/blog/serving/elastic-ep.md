---
source: https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism
lang: en
fetched: 2026-08-31
---

# Elastic Expert Parallelism

2026-05-14. https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism  
RFC #20323, PR #34861; NIXL EP PR #35627; fault-tolerance RFC #30112. Study note. Scope of the implementation in the post is narrow — flags as written then.

Wide-EP ([large-scale.md](large-scale.md)) made expert layout the main course. EP itself stayed **static**: start with N workers, live with N, or restart. Elastic EP resizes **data-parallel workers at runtime**. Attention stays per DP worker (own KV + scheduler); expert layers share one EP group of size `DP × TP`. Changing DP changes the EP group and forces expert redistribution.

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

## What has to change

A new EP size invalidates: EP/DP/world groups (fixed rank sets); expert→rank maps; weights on new and reshuffled ranks; CUDA graphs and `torch.compile` specializations. Scaling is a coordinated state machine that must coexist with in-flight forwards.

## Scale-up `DP=N` → `DP=M`

1. `POST /scale_elastic_ep`. If `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1`, drain in-flight work up to `drain_timeout` (default **120 s**).
2. **Ray DP backend** launches extra engine-cores on free GPUs. New ranks get the current expert map and placeholder weights. Two-phase ready: standby groups first, then weight transfer.
3. Existing ranks build **standby groups** with `StatelessGroupCoordinator` (independent of PyTorch `WORLD`) so the old topology can keep forwarding. `nixl_ep` can add/remove ranks via `connect_ranks()` / `disconnect_ranks()` without tearing everything down.
4. Broadcast expert map; GPU-to-GPU send/recv (EPLB path, NVLink/RDMA) moves **non-expert** weights (attention, norms, embeddings). Expert weights wait for the post-switch EPLB reshuffle. Ordinary EPLB is paused.
5. **Switch:** drop CUDA graphs and compile state; promote standby groups; destroy old groups; reconfigure MoE; re-warm. Sync running flag / wave / step counters. New ranks can run attention but **do not own experts yet**.
6. **EPLB reshuffle** across all M ranks, then resume normal EPLB.

## Scale-down

Reshuffle **first**. Departing ranks may still own experts; all M cores consolidate onto the surviving N, then leave.

## Two-stage barrier

DP cores are async. A naive barrier splits the group between reconfiguration and one extra forward → deadlock. Stage 1 has a timeout: if peers did not arrive, go back for one more engine step. Stage 2 (no timeout path) enters the next phase together.

## Fault tolerance

Same reconfiguration path: detect failure → scale-down (drop the dead rank, redistribute experts) → scale-up when spare GPUs exist. NIXL EP adds EP-side failure detect/report/recover and reconnect.

## Not yet (as of the post)

`tensor_parallel_size>1`; `api_server_count` capped at 1; **no DBO**; no MoE draft/drafter; Ray DP only; autoscaling *policy* is left to Dynamo / llm-d.

## Launch sketch

Ray DP, TP=1, one API server, no DBO:

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-backend ray \
    --api-server-count 1 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --eplb-config.num_redundant_experts 0 \
    --all2all-backend allgather_reducescatter \
    --gpu-memory-utilization 0.8
```

Join GPUs with `ray start --address="${HEAD_NODE_IP}:6379"`, then POST a new `new_data_parallel_size`. For NIXL: `--all2all-backend nixl_ep`.
