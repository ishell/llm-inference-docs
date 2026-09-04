---
source: https://vllm.ai/blog/2025-11-22-ray-symmetric-run
lang: en
fetched: 2026-09-04
---

# Streamlined multi-node serving with Ray `symmetric-run`

Chinese: [zh/vllm/blog/serving/ray-symmetric.md](../../../../zh/vllm/blog/serving/ray-symmetric.md)

2025-11-22. Richard Liaw (Anyscale/Ray), Kaichao You (vLLM). For SLURM / `mpssh` / `pssh`: **one command on every node**. [elastic-ep.md](elastic-ep.md) needs the Ray DP backend; this is how the cluster comes up. Study note.

Ray added `ray symmetric-run`: launch the **same entrypoint** on every node in a Ray cluster. That matches HPC and parallel-SSH habits when you spawn multi-node vLLM. The post walks the old dance, a two-machine example, then the new API.

Ray had recently joined the PyTorch Foundation. The vLLM and Ray teams were aligning on the next layer of AI infrastructure.

Local figures (copyright remains with the original site; study copies):

![symmetric run](../../../../assets/vllm/blog/serving/ray-symmetric/01-symmetric-run.png)

Figure 1: Overview of Ray’s `symmetric-run`.

## Context

People coming to Ray already have a launch ritual.

- Interactive work on bare clusters: `mpssh` / `pssh`, one command parameterized by “rank”.
- HPC (SLURM, PBS): **symmetric execution** — one program entrypoint on every node at once, like MPI.

Ray’s recommended job pattern is a different philosophy. The entrypoint runs on the **head**; the head orchestrates and delegates to **workers**. Runtime lifecycle is separate from job execution, so you keep two command sets: one to stand up head/worker roles, another to run the work.

## Motivating Example

Two bare machines. No Ray cluster launcher, no KubeRay.

Head, start Ray:

```bash
ray start --block
```

Each worker, connect back (worker terminal 1):

```bash
ray start --block --address='ip:6379'
```

Then a **second** terminal on the head for the job:

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

Teardown, on **every** node:

```bash
ray stop
```

This configuration is trial-and-error. A common miss is `VLLM_HOST_IP`. Then you tear the cluster down, set the variable on `ray start`, and walk the whole sequence again.

For anyone who expected one symmetric command, that is a lot of friction. `ray symmetric-run` is the reply.

## What `symmetric-run` does

Same entrypoint on every node. The script handles Ray setup, job execution, and teardown — closer to `mpirun` or `torchrun`.

Same two-machine serve, in a SLURM sbatch script or via `mpssh`:

```bash
ray symmetric-run \
  --address <head_node_address>:6379 \
  --min-nodes 2 \
  --num-gpus 8 \
  -- vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

Every node runs that argv. Behavior underneath differs by role.

**Head:**

1. Starts Ray in `--head` mode.
2. Waits for nodes to register — the post’s prose says **four** nodes; the CLI example uses `--min-nodes 2`.
3. Runs the user command (`vllm serve Qwen/Qwen3-32B …`).
4. Shuts Ray down when done.

**Workers:** only `ray start --address head-node:6379`, then wait until the job ends and self-destruct. No extra SSH orchestration or startup scripts.

Prefix environment variables propagate into the Ray runtime:

```bash
ENV=VAR ray symmetric-run --address 127.0.0.1:6379 -- python test.py
```

## Conclusion

Symmetric run is for HPC and parallel-SSH. Docs: [Ray SLURM guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html). Issues: [ray-project/ray](https://github.com/ray-project/ray/). Community: [vLLM Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack), [Ray Slack](https://www.ray.io/join-slack).
