---
source: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
lang: zh
fetched: 2026-08-31
---

# TRT-LLM KV Cache 系统

块池（每块 token 数须为大于 1 的 2 的幂）。头数/窗口相同的层共用一个池；GQA/MQA、不同窗口会再开池。

**跨请求复用：** 填满的块进 radix 树，相同前缀跳过计算并共享显存。驱逐：带优先级的 LRU（0–100）。目前只能驱逐叶子（对有限窗口层不理想）。可选卸到 CPU（`host_cache_size`，默认 0）。优先级低于 `secondary_offload_min_priority`（默认 35）的块直接丢、不卸。

`enable_block_reuse` 默认开。调度器的 `enable_prefix_aware_scheduling` 只影响入队/token 预算估计，不关实际 reuse。

`free_gpu_memory_fraction`（默认 0.9）和 `max_tokens` 取较小者。`dtype` 默认从模型推断。

`cache_salt` 做复用隔离（进块 hash，必须用密码学 hash）。多模态可传 `multi_modal_uuids`。

部分混合/稀疏模型默认 V2 manager。双模型 speculative decoding 不能用 V2。
