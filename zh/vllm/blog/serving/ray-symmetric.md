---
source: https://vllm.ai/blog/2025-11-22-ray-symmetric-run
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# `ray symmetric-run`：多机起 vLLM 不必先扮演 head/worker

英文对照：[en/vllm/blog/serving/ray-symmetric.md](../../../../en/vllm/blog/serving/ray-symmetric.md)  
原文：https://vllm.ai/blog/2025-11-22-ray-symmetric-run  
2025-11-22。Richard Liaw（Anyscale/Ray）、Kaichao You（vLLM）。写给 SLURM / `mpssh` / `pssh` 这类「每台机器敲同一条命令」的人。[Elastic EP](elastic-ep.md) 也依赖 Ray DP backend；这篇是把集群**先拉起来**的那条命令。

Ray 多了一条 `ray symmetric-run`：在集群的**每一台节点**上跑**同一条入口**。HPC 和并行 SSH 起多机 vLLM 时，姿势终于对上了。原文先写旧流程的别扭，再走两台机器的例子，最后才是新 API。

Ray 当时刚加入 PyTorch Foundation。vLLM 和 Ray 两边在对齐下一层基础设施。

本地图（原文版权仍归原站；学习对照用）：

![symmetric run](../../../../assets/vllm/blog/serving/ray-symmetric/01-symmetric-run.png)

Figure 1：Ray `symmetric-run` 总览。

## Context

来找 Ray 的人，手里已经有一套起法。

- 裸集群上的交互活：`mpssh` / `pssh`，一条命令，用「rank」当参数。
- HPC（SLURM、PBS）：**对称执行**——所有节点同时跑同一个入口，像 MPI。

Ray 推荐的作业姿势是另一种哲学。入口在 **head** 上跑；head 编排，把活派给 **worker**。运行时的生命周期和作业执行是两套事，于是你要记两套命令：一套把 head/worker 角色立起来，另一套才真正干活。

## Motivating Example

两台裸机器。不能用 Ray 的 cluster launcher，也不能用 KubeRay。

head 上先起 Ray：

```bash
ray start --block
```

每台 worker 连回去（worker 的第一个终端）：

```bash
ray start --block --address='ip:6379'
```

然后回到 head，**再开一个终端**跑作业：

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

收摊时，**每一台**都要：

```bash
ray stop
```

这种配法通常要试错。常见的漏是 `VLLM_HOST_IP`。漏了就要拆集群，在 `ray start` 上补变量，整条流程再走一遍。

指望「一条对称命令」的人，会在这里磨很久。`ray symmetric-run` 是对这件事的回答。

## What `symmetric-run` does

每一台节点同一条入口。脚本自己管 Ray 的搭建、作业执行、收摊——更接近 `mpirun` 或 `torchrun`。

还是上面那两台 serve，写进 SLURM sbatch，或交给 `mpssh`：

```bash
ray symmetric-run \
  --address <head_node_address>:6379 \
  --min-nodes 2 \
  --num-gpus 8 \
  -- vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

每台敲同一条 argv。底下的行为按角色分开。

**Head：**

1. 以 `--head` 启动 Ray。
2. 等节点登记——原文散文写等 **四** 台；示例 CLI 是 `--min-nodes 2`。
3. 跑你的命令（`vllm serve Qwen/Qwen3-32B …`）。
4. 做完后关掉 Ray。

**Worker：** 只做 `ray start --address head-node:6379`，等到作业结束再自杀。不必另写 SSH 编排或启动脚本。

环境变量写在命令前面，会进 Ray runtime：

```bash
ENV=VAR ray symmetric-run --address 127.0.0.1:6379 -- python test.py
```

## Conclusion

symmetric-run 面向 HPC 和并行 SSH。文档：[Ray SLURM 指南](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)。问题：[ray-project/ray](https://github.com/ray-project/ray/)。社区：[vLLM Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack)、[Ray Slack](https://www.ray.io/join-slack)。
