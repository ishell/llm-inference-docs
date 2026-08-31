---
source: https://docs.vllm.ai/en/stable/usage/v1_guide/
lang: zh
fetched: 2026-08-31
---

# vLLM V1 指南（笔记）

V1 是当前引擎。统一 scheduler 用 `{request_id: num_tokens}` 给每条请求分配 token 预算——chunked prefill、prefix cache、spec decode 共用这套预算。

- 能开 chunked prefill 就**默认开**（V0 是按模型条件开）。
- 先排 decode，剩下的 `max_num_batched_tokens` 给 prefill，塞不下就切块。
- Prefix cache 相对 V0 接近零额外开销。
- 调度：FCFS 或 priority（`--scheduling-policy`）。
- 默认抢占：`RECOMPUTE` 不是 `SWAP`。

V0 已去掉：`best_of`、每请求 logits processor、GPU↔CPU KV swap 等。

调优顺序见 `optimization/optimization.md`（先 CPU 核 → `-O*` → `max_num_batched_tokens` → 并行）。
