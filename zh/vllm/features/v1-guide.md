---
source: https://docs.vllm.ai/en/stable/usage/v1_guide/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# vLLM V1 指南（学习译文）

英文对照：`en/vllm/features/v1-guide.md`  
原文：https://docs.vllm.ai/en/stable/usage/v1_guide/  
发布故事：[v1-alpha](../blog/architecture/v1-alpha.md)。解剖：[Anatomy](../blog/architecture/anatomy.md)。调优顺序：[optimization.md](../optimization/optimization.md)。

V1 是当前引擎。V0 已经拆掉。统一 scheduler 用 `{request_id: num_tokens}` 给每条请求分配 token 预算——chunked prefill、prefix cache、spec decode 共用这套账本，不再各开各的旁门。

- 能开 chunked prefill 就**默认开**（V0 按模型条件开）。先排 decode，剩下的 `max_num_batched_tokens` 给 prefill，塞不下就切块。
- Prefix cache 相对 V0 接近零额外开销。
- 调度：FCFS 或 priority（`--scheduling-policy`）。
- 默认抢占：`RECOMPUTE`，不是 `SWAP`。被请出去的人回头把过去重读一遍；频繁发生时先给 KV 房间（提高 `gpu_memory_utilization` 或 TP），见 optimization 的 Preemption 节。
- 默认走 `torch.compile` + CUDA graph。关掉：`-O0` / `--enforce-eager`。编译本身：[torch.compile 博客](../blog/architecture/torch-compile.md)。

V0 带走的：`best_of`、每请求 logits processor、GPU↔CPU KV swap 等。需要把 KV 寄存在 CPU 时，走后来的 [Offloading Connector](../blog/serving/kv-offload.md)，不是 V0 那套同步 swap。

多进程是 V1 的房子结构：API server、engine core、每卡一个 worker。CPU 核不够时，GPU 会像在等端菜的人——optimization 把这件事放在旋钮顺序的第一位。
