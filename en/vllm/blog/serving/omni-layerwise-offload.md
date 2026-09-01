---
source: https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload
lang: en
fetched: 2026-09-01
---

# Distributed Layerwise Offload: 124 GB model, 64 GB HBM

Chinese: `../../zh/vllm/blog/serving/omni-layerwise-offload.md`  
vLLM 0.27.0 + Omni `v0.27.0rc1`. Cosmos3-Super 64B / 124 GB.

`--enable-distributed-layerwise-offload`. mmap weights through the OS page cache; Cosmos3-Nano DP4 cold-start cgroup peak **178 GB → 47 GB** (~−73%). Each rank keeps 1/dp of weights; AllGather overlaps compute. Exactly **two layers** double-buffered on device, independent of depth. DP multi-concurrency ~**3.3×** vs single-request HSDP (83% of 4×). On 8×B300 AllGather vs rank-local depends on topology. v0.26.0 Cosmos3 DLO+DP rejected requests (`supports_request_batch`); #5864 lets each rank run a single-request forward. `--dlo-no-use-allgather` still had #5911 open.
