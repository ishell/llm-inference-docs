---
source: https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload
lang: en
fetched: 2026-09-01
---

# Distributed Layerwise Offload: 124 GB model, 64 GB HBM

Chinese: `../../zh/vllm/blog/serving/omni-layerwise-offload.md`  
vLLM 0.27.0 + Omni `v0.27.0rc1`. Cosmos3-Super 64B / 124 GB.

`--enable-distributed-layerwise-offload`. mmap weights through the OS page cache; Cosmos3-Nano DP4 cold-start cgroup peak **178 GB → 47 GB** (~−73%). Each rank keeps 1/dp of weights; AllGather overlaps compute. Exactly **two layers** double-buffered on device, independent of depth. DP multi-concurrency ~**3.3×** vs single-request HSDP (83% of 4×). On 8×B300 AllGather vs rank-local depends on topology. v0.26.0 Cosmos3 DLO+DP rejected requests (`supports_request_batch`); #5864 lets each rank run a single-request forward. `--dlo-no-use-allgather` still had #5911 open.

Local figures (copyright remains with the original site; study copies):

![dlo problem overview](../../../../assets/vllm/blog/serving/omni-layerwise-offload/01-dlo-problem-overview.svg)

![mmap loading memory](../../../../assets/vllm/blog/serving/omni-layerwise-offload/02-mmap-loading-memory.svg)

![weight sharding allgather](../../../../assets/vllm/blog/serving/omni-layerwise-offload/03-weight-sharding-allgather.svg)

![dlo pipeline last frame](../../../../assets/vllm/blog/serving/omni-layerwise-offload/04-dlo_pipeline_last_frame.png)

![dlo pipeline](../../../../assets/vllm/blog/serving/omni-layerwise-offload/05-dlo_pipeline.gif)

![hbm nano vs super](../../../../assets/vllm/blog/serving/omni-layerwise-offload/06-hbm-nano-vs-super.svg)

![dp multi concurrency](../../../../assets/vllm/blog/serving/omni-layerwise-offload/07-dp-multi-concurrency.svg)

![ascend memory accounting](../../../../assets/vllm/blog/serving/omni-layerwise-offload/08-ascend-memory-accounting.svg)

![minimax h3 topology policy](../../../../assets/vllm/blog/serving/omni-layerwise-offload/09-minimax-h3-topology-policy.svg)

![minimax h3 multimodal frontiers](../../../../assets/vllm/blog/serving/omni-layerwise-offload/10-minimax-h3-multimodal-frontiers.png)
