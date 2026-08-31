---
source: https://vllm.ai/blog/2025-11-22-ray-symmetric-run
lang: en
fetched: 2026-09-01
---

# `ray symmetric-run`

2025-11-22. For SLURM / `mpssh`: one command on every node. [elastic-ep.md](elastic-ep.md) needs Ray DP; this is how the cluster comes up. Study note.

Old dance: head `ray start --block`; workers `--address`; another head terminal for `vllm serve`; `ray stop` everywhere. Missing `VLLM_HOST_IP` means tear down and retry.

```bash
ray symmetric-run \
  --address <head>:6379 \
  --min-nodes 2 \
  --num-gpus 8 \
  -- vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

Same argv everywhere. Workers only join and wait; head `--head`, wait for `--min-nodes`, run the command, shut down. Prefix env vars propagate: `ENV=VAR ray symmetric-run ...`.
