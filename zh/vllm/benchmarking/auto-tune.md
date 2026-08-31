---
source: https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
lang: zh
fetched: 2026-08-31
---

# vLLM auto_tune.sh

网格搜索 `max-num-seqs` × `max-num-batched-tokens` 最大化吞吐，可加 P99 e2e 延迟、前缀缓存命中率约束。

1. 从 0.98 往下找不 OOM 的最高 `gpu-memory-utilization`。
2. 每组参数：起服务 → `--request-rate inf` → P99 超标就降速率直到满足。
3. 记下合法吞吐最高的一组，并保存该次 profiler。

脚本路径里不要出现 `vllm`（`pkill -f vllm` 会把调参自己杀掉）。

结果在 `$BASE/auto-benchmark/<时间戳>/`。`best_max_num_seqs: 0` 表示没有满足约束的组合。
