---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html
lang: zh
fetched: 2026-08-31
---

# 调 Max Batch Size 和 Max Num Tokens

In-flight batching 把 context（prefill）和 generation 混在同一次 iteration。两个编译期上限决定谁能被调度：

- **`max_batch_size`**：同时在飞的请求数。默认 **2048**。太小 → 新请求排不上。建议扫 2 的幂。
- **`max_num_tokens`**：去 padding 后，一次 iteration 最多打包多少 token。默认 **8192**。太小 → 长 prompt 进不去；太大 → 长上下文把 KV 显存挤没（甚至 OOM）。

调度器**优先 generation token**，再用剩余 token 预算塞新的 prefill。超预算的请求起不来，除非开了 **context chunking**（依赖 paged context FMHA）。

官方案例：`max_batch_size=4`、`max_num_tokens=12`。两个 5-token prompt 占 10，剩 2 不够开第三条（无 chunking）。进入 generation 后每条只占 1 token，就能再塞 prefill——直到撞上 batch 上限。某条 EOS 被踢掉后，第 5 条才能进。

## 怎么设

```python
build_config = BuildConfig(max_batch_size=512, max_num_tokens=2048)
```

CLI：`trtllm-build --max_batch_size … --max_num_tokens …`

能网格搜索就两个一起扫。`max_num_tokens` 可试 ≥1024 的 2 的幂。

## 案例（Llama-3.3-70B，4×H100）

batch size：**512** 最好（相对默认 2048 吞吐大约 +20%；64 会堵）。延迟几乎不动。

`max_num_tokens` 在 batch=512 时 2048/8192/16384 差距很小，该负载上 2048 略好。必须实测。

## 为什么总建议开 paged context attention

Chunked prefill 把长 prompt 拆到多次 iteration：

1. 长 prompt 不会被在飞请求永远挡住（最差 TTFT 更好）。
2. `max_num_tokens` 不必 ≥ 最长 prompt，**显存可以留给 KV cache**。

NVIDIA：即使某次实验看不出收益，也建议开。

相对上一页调完编译旗标后再调这两个：吞吐约 **+21%**，延迟在噪声内。相对完全没调的 baseline：吞吐约 **+58%**，ITL 约 **-53%**。
