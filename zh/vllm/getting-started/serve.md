---
source: https://docs.vllm.ai/en/stable/cli/serve/
lang: zh
fetched: 2026-08-31
---

# `vllm serve` — 和推理性能相关的旗标

完整 CLI 是引擎参数的生成页，非常长。这里只留调优文档里会碰到的旋钮。原文：https://docs.vllm.ai/en/stable/cli/serve/

嵌套 JSON：`--json-arg '{"k": {"x": 1}}'` ≡ `--json-arg.k.x 1`。列表用 `+`。YAML：`--config`。

## 并行

| 旗标 | 作用 |
|---|---|
| `-tp` | Tensor parallel（默认 1） |
| `-dp` | Data parallel |
| `-ep` | MoE 用 expert parallel |
| `-dcp` | Decode KV 分片（本身不增大 world size） |
| `-pcp` | Prefill 按序列切开（会增大 world size） |
| `--api-server-count` | 前端进程数（默认跟 DP） |

## KV / 显存

| 旗标 | 作用 |
|---|---|
| `--gpu-memory-utilization` | 本实例占 GPU 比例。**默认 0.92**。同一张卡上两个实例要自己拆。 |
| `--kv-cache-memory-bytes` | 按字节指定 KV，**覆盖** utilization |
| `--kv-cache-dtype` | `auto` / `fp8` / `nvfp4` … |
| `--block-size` | 每块 KV 多少 token |
| `--enable-prefix-caching` | APC |
| `--prefix-caching-hash-algo` | 默认 `sha256`（更安全）；`xxhash` 更快但有碰撞风险 |
| `--max-model-len` | 上下文；`auto`/`-1` 按显存自适应 |

## 调度（对应 TRT-LLM 的 max_batch / max_num_tokens）

| 旗标 | 作用 |
|---|---|
| `--max-num-batched-tokens` | 每步最多处理多少 token。V1 **最重要的吞吐旋钮**，常试 >8192。 |
| `--max-num-seqs` | 每步最多多少条序列 |
| `--enable-chunked-prefill` | V1 能开则默认开 |
| `--scheduling-policy` | `fcfs`（默认）或 `priority` |
| `--async-scheduling` | 填 CPU 空隙，延迟和吞吐都更好 |
| `--watermark` | 预留一部分 KV block，减少 preemption（默认 0=关） |

## 编译 / 加速

| 旗标 | 作用 |
|---|---|
| `--optimization-level` | `-O0` 启动最快 … `-O3` 性能最好。**默认 2。** |
| `--performance-mode` | `balanced` / `interactivity`（小 batch 延迟）/ `throughput`（更大 graph、更狠 batching） |
| `-cc` | torch.compile + cudagraph |
| `--speculative-config` / `--spec-method` | ngram、EAGLE、MTP… |

调优顺序见 `../optimization/optimization.md`。不要把生成页里几百个旗标全拿去扫。
