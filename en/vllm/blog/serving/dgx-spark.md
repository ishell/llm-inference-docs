---
source: https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
lang: en
fetched: 2026-09-01
---

# DGX Spark: small-batch NVFP4 on 128GB unified memory, not a datacenter GPU

Chinese: `../../zh/vllm/blog/serving/dgx-spark.md`  
GB10, `sm_121`. Nemotron-3-Super-120B-A12B-NVFP4. Demo numbers, not a leaderboard.

CPU/GPU/OS/container/weights/KV share one pool. `--gpu-memory-utilization` must leave headroom. `--max-num-seqs 4`: higher and per-token bandwidth tax beats continuous batch. Fit ~10–15B-active NVFP4 MoE, not high-concurrency dense. Official image for `sm_121`; `cu130-nightly` is a track, not a pin.

Decode held **22.7–23.7 tok/s** (five-scenario median after warmup). TTFT scales roughly with prompt; prefill ~140 tok/s on tiny prompts to ~1900 on long. First request Inductor/FlashInfer JIT ~**25s** — ping it yourself. Safetensor load 10–15 min. `--kv-cache-dtype fp8` can hurt on Spark; not a default. MTP / async need recipe re-measure. Do not paste datacenter TPS onto a desk box.
