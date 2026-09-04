---
source: https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism
lang: en
fetched: 2026-09-04
---

# Elastic Expert Parallelism in vLLM

Chinese: [zh/vllm/blog/serving/elastic-ep.md](../../../../zh/vllm/blog/serving/elastic-ep.md)

2026-05-14. [RFC #20323](https://github.com/vllm-project/vllm/issues/20323), landing [PR #34861](https://github.com/vllm-project/vllm/pull/34861); NIXL EP [PR #35627](https://github.com/vllm-project/vllm/pull/35627). Fault-tolerance direction: [RFC #30112](https://github.com/vllm-project/vllm/issues/30112). DP Attention + EP background: [RFC #16037](https://github.com/vllm-project/vllm/issues/16037). Scope in the post is **narrow** — flags as written then.

Wide-EP ([large-scale.md](large-scale.md)) made expert layout the main course: more KV room, high concurrency or long context (RL + agents). EP itself stayed **static**: start with N workers, live with N, or **full restart** (slow, drops in-flight traffic). Elastic EP resizes **data-parallel workers at runtime**. Attention stays per DP worker (own KV + scheduler); expert layers share one EP group of size `DP × TP`. Changing DP changes the EP group and forces expert redistribution.

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 8}'
```

Local figures (copyright remains with the original site; study copies):

![elastic ep](../../../../assets/vllm/blog/serving/elastic-ep/01-elastic-ep.png)

## Background

MoE: dense attention, sparse expert FFN; tokens go only to chosen experts.

- **DP Attention:** request-level parallelism. Each engine-core has its own KV and scheduler. MLA hates naive TP (duplicated KV).
- **EP:** whole experts on different GPUs; dispatch only to owners.

Elastic EP moves **DP count**, therefore EP group size.

## State that a resize invalidates

- EP / DP / world groups (fixed rank sets)
- Expert→rank maps
- Weights on new ranks and reshuffled experts
- CUDA graphs and `torch.compile` specializations

Scaling is a coordinated state machine that must coexist with in-flight forwards.

## Scale-up `DP=N` → `DP=M`

1. `POST /scale_elastic_ep`. If `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1`, drain in-flight work up to `drain_timeout` (default **120 s**). Otherwise start immediately.
2. **Ray DP backend** launches extra engine-cores on free GPUs. New ranks get the current expert map and **placeholder** weights. Two-phase ready: standby groups first, then weight transfer.
3. Existing ranks build **standby groups** with `StatelessGroupCoordinator` (independent of PyTorch `WORLD`) so the old topology can keep forwarding. `nixl_ep` can add/remove ranks via `connect_ranks()` / `disconnect_ranks()` without tearing everything down.
4. Broadcast expert map; GPU-to-GPU send/recv (EPLB path, NVLink/RDMA) moves **non-expert** weights (attention, norms, embeddings), load-spread across old ranks. **Expert weights wait** for the post-switch EPLB reshuffle. Ordinary EPLB is paused.
5. **Switch:** drop CUDA graphs and compile state; promote standby groups to active EP/DP/world; destroy old groups; reconfigure MoE; re-warm. Sync running flag / wave / step counters. New ranks can run attention but **do not own experts yet**.
6. **EPLB reshuffle** across all M ranks, then resume normal EPLB.

## Scale-down

Reshuffle **first**. Departing ranks may still own experts; all M cores consolidate onto the surviving N, then leave.

## Two-stage barrier

DP cores are async. A naive barrier splits the group between reconfiguration and one extra forward → deadlock. Stage 1 has a timeout: if peers did not arrive, go back for one more engine step. Stage 2 (no timeout path) enters the next phase together.

## Fault tolerance

Same reconfiguration path after a death: detect (health checks or backend signals) → scale-down (drop the dead rank, redistribute experts) → scale-up when spare GPUs exist. NIXL EP adds EP-side failure detect/report/recover and reconnect.

## Not yet (as of the post)

- `tensor_parallel_size > 1` and richer parallel mixes
- `api_server_count` capped at **1**; **no DBO**; no MoE draft / drafter
- Reconfiguration window (warmup, CUDA graph recapture) still expensive
- Autoscaling **policy** is left to Dynamo / llm-d — this post is the control knob
- Scale operations depend on the **Ray DP backend**

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

Join GPUs: `ray start --address="${HEAD_NODE_IP}:6379"`, then POST a new `new_data_parallel_size` (example: 16 up, 8 down).

NIXL path:

```bash
uv pip install nixl

vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-backend ray \
    --api-server-count 1 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --enable-eplb \
    --all2all-backend nixl_ep
```

Install / transport: [NIXL repo](https://github.com/ai-dynamo/nixl).

## Acknowledgements

Sky Computing: Yongji Wu. NVIDIA: Itay Alroy, Moein Khazraee, Omri Kahalon, Tzu-Ling Kan, Ron Tourgeman. Red Hat: Tyler Michael Smith. Anyscale: Rui Qiao. Plus the broader vLLM community.
