---
source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 单机也要 P/D：MI300X 上的 MORI-IO

英文对照：[en/vllm/blog/serving/moriio.md](../../../../en/vllm/blog/serving/moriio.md)  
原文：https://vllm.ai/blog/2026-04-07-moriio-kv-connector  
2026-04-07。AMD Instinct MI300X，**一台 8 卡节点内**做 Prefill/Decode 分离。Connector：`MoRIIOConnector`（MORI = Modular RDMA Interface）。数字是 **Qwen3-235B-A22B-FP8**、**8 req/s**、输入 **2000** / 输出 **1000**。标题成绩：同一套 8 卡上，goodput 大约 **2.5×**。落地 [PR #29304](https://github.com/vllm-project/vllm/pull/29304)。库：[ROCm/mori](https://github.com/ROCm/mori)。前一篇 AMD / Embedded LLM 的 MoE playbook 是 TP/DP/PP/EP 的姊妹篇。

本地图（原文版权仍归原站；学习对照用）：

![read mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/01-read-mode-request-flow-diagram.svg)

![write mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/02-write-mode-request-flow-diagram.svg)

![SLO attainment](../../../../assets/vllm/blog/serving/moriio/03-SLO-attainment.png)

![read mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/04-read-mode-kv-transfer-sequence-diagram.svg)

![write mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/05-write-mode-kv-transfer-sequence-diagram.svg)

## 同一栋楼里的两场戏

Prefill 是 **compute-bound**（整段 prompt 上的大 GEMM，费用跟输入长度走）。Decode 是 **memory-bandwidth-bound**（一次一个 token，反复从 HBM 搬权重）。挤在同一套实例里：一条胖 Prefill 能让几十路 Decode 口吃；Decode 反过来挡新的 Prefill。ITL 开始乱跳。

「P/D 是机房的事」会把单机上的 goodput 留在桌子上。拆开：例如 **4 卡 Prefill + 4 卡 Decode**，中间把 KV 交过去——可以是数 GB。MORI-IO 用节点内 **RDMA**。这篇的范围：**一台盒子，8 张卡**。

## 架构（原文 Table 1）

| 零件 | 角色 |
| --- | --- |
| Prefill 实例 | Prompt → KV（例：GPU 0–3） |
| Decode 实例 | 用转过来的 KV 逐 token 说（GPU 4–7） |
| Proxy | 客户端入口，编排两段 |

模式由 `VLLM_MORIIO_CONNECTOR_READ_MODE` 决定。一对实例第一次 RDMA 之前，ZMQ 在**后台线程**交换基址、块大小、每层 stride；RDMA session 缓存起来。

### Read（`VLLM_MORIIO_CONNECTOR_READ_MODE=1`）

Proxy **串行**派发。Figure 1：

1. 客户端 → proxy
2. Proxy → Prefill（`max_tokens=1`）
3. Prefill → proxy：`remote_block_ids`、`remote_engine_id`
4. Proxy 把这些 ID 转给 Decode
5. Decode **拉** KV（`WAITING_FOR_REMOTE_KVS`）；调度器每步跳过，直到 RDMA read 完
6. Decode 通知 Prefill 放块
7. Token 经 SSE 回来

### Write（默认：不设或 `=0`）

Proxy **同时**开火。Figure 2：

1. 客户端 → proxy
2. Proxy 并行发给 Prefill **和** Decode（请求里带着对方的连接信息；proxy **不等** Prefill 返回）
3. Prefill 每算完一层就 **推**（`save_kv_layer` RDMA write 进 Decode 预分配的块）。Chunked Prefill：攒到最后一块再写
4. Decode 轮询 `pop_finished_write_req_ids`，状态仍是 `WAITING_FOR_REMOTE_KVS`
5. 进 ready 队列，开口
6. SSE 回来

玩具 proxy（`examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`）：READ 要 `await` Prefill，把 `remote_engine_id` / `remote_block_ids` 写进 `kv_transfer_params`；WRITE 直接 `asyncio.create_task` Decode。Read **必须**经 proxy 转 block id；write 不必。

### Table 2

| 性质 | Read | Write |
| --- | --- | --- |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `=1` | 不设 / `=0` |
| RDMA | Decode 拉 | Prefill 推 |
| Proxy | 串行 | 并发 |
| 经 proxy 转 `remote_block_ids` | 要 | 不要 |
| KV 清理 | 拉完 Decode 通知 Prefill | Prefill 按请求跟踪写完 |

## Goodput，不是裸吞吐

沿 DistServe：**goodput** = 同时满足 TTFT **&lt; T_ttft** 且 ITL **&lt; T_itl** 的最大请求率。这里：TTFT **&lt; 1 s**，ITL **&lt; 50 ms**/token。

### 8 req/s、100 条（Table 3）

| 指标 | Standard 1×TP8 | Standard 2×TP4 | MORI-IO Read 1P+1D | MORI-IO Write 1P+1D |
| --- | --- | --- | --- | --- |
| 两条 SLO 都达标 | 26/100 | 30/100 | 70/100 | **73/100** |
| 主要死因 | ITL 尖刺（P99 ≫ 50 ms） | ITL 双峰约 30 ms 与约 150 ms | 部分 TTFT &gt; 1 s | 部分 TTFT &gt; 1 s |
| 相对 | 0.9× | 1× | 2.4× | **2.5×** |

标准路径死在 ITL。分离之后 ITL 违规消失，剩下的是 TTFT。Write 比 Read 多过 3 条，因为并发 dispatch 把更多请求按在 1 s 以内。

### 请求率 0.5–10 的 SLO（Figure 4 / `03-SLO-attainment.png`）

- 1×TP8：低速率就开始 ITL 违规；rate 8 时 **26/100**
- 2×TP4：0.5 时 100% → 1 时约 **60%** → 2 时约 **25%**，然后平台
- Read：大约到 rate **5** 仍 100%，10 时大约 **44%**（TTFT）
- Write：大约到 rate **5.5** 仍 100%，10 时大约 **46%**

## ITL 为什么好、TTFT 为什么贵

共用引擎：一次 Prefill 前向远长于一步 Decode，同一 batch 里的 Decode 都得等 → ITL 胀。Decode 引擎只跑 Decode：ITL 稳定，**两种 mode 一样**。

标准 TTFT：

```text
TTFT = queue + prefill_forward_pass + sample_T1 + detokenize + SSE_encode + network
```

Read 多两笔（Figure 5）：

```text
TTFT = queue(prefill) + prefill_forward_pass
     + [proxy 串行：等 Prefill，再派 Decode]     # Overhead 1
     + RDMA（WAITING_FOR_REMOTE_KVS）            # Overhead 2
     + queue(decode) + sample_T1 + detokenize + SSE + 网
```

Write（Figure 6）：

```text
TTFT ≈ max(queue(prefill) + prefill_forward_pass + RDMA_write_time,
           queue(decode))
     + sample_T1 + detokenize + SSE + 网
```

Write 去掉 Overhead 1。RDMA write 和 Prefill 计算**重叠**，Overhead 2 不必整段加进墙钟。Decode 的 `RequestStatus.WAITING_FOR_REMOTE_KVS`：调度器 `_update_waiting_for_remote_kv`，否则跳过（`vllm/v1/request.py`、`vllm/v1/core/sched/scheduler.py`）。

**Table 4——什么时候拆：**

| 处境 | 建议 |
| --- | --- |
| 负载下 ITL p99 达不到 SLO | 拆 |
| TTFT 是绑死的 UX | 标准路径可能更好 |
| 高并发、长 prompt | 拆（Prefill 干扰最狠） |
| 低速率、短 prompt | 标准够用 |

## 怎么起（骨架）

Prefill 是 `kv_producer`，Decode 是 `kv_consumer`。实例用 ZMQ 向 proxy（`proxy_ping_port`）报到并续命。Proxy：`python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py`。Round-robin。文档：[disagg prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)。

```bash
vllm serve <model> \
  --gpu_memory_utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "20005",
      "handshake_port": "6301",
      "notify_port": "6105"
    }
  }'
```

Decode：`kv_consumer`，`http_port` `40005`，`handshake_port` `7301`，`notify_port` `7501`。

**Table 5——端口**（`MoRIIOConfig` / `moriio_common.py` 会加 per-rank 偏移）：

| 端口 | 用途 |
| --- | --- |
| `proxy_ping_port` | ZMQ 注册 |
| `http_port` | vLLM HTTP；proxy 往这里转 |
| `handshake_port` | 一次性 KV 布局元数据 |
| `notify_port` | 按请求：Prefill 通知 Decode 块已就绪 |

## 实验细节

vLLM 仓库里的 `Dockerfile.rocm_base`（构建说明里 MORI `2d02c6a9`；运行时库 commit [`c365eaed`](https://github.com/ROCm/mori/commit/c365eaed02b13e6b8f2e9c8215b21516d86856ce)）、`Dockerfile.rocm`。硬件：**8× MI300X（gfx942）**；**2× EPYC 9654**。驱动 **6.10.5**；容器 `rocm/vllm-dev`（ROCm **7.0.51831-a3e329ad8**）；vLLM **0.16.0rc1.dev1+gc46b0cd0a**（`c46b0cd0a`）；PyTorch **2.9.1+git8907517**。模型同上；random 数据集；**100** 条；速率 **0.5–10**，步长 **0.5**。

**Table 6：** 1×TP8 是单引擎混跑；2×TP4 是两份混跑、RR proxy——和 1P+1D **同一套 8 卡切成两组**，只差「每组混跑还是专职」。Read / Write 都是 GPU 0–3 Prefill、4–7 Decode，TP=4+EP；prefix cache **关掉**（Write 路径上 MORI-IO **要求**关掉）。MoE 的 expert routing 会把 ITL 抖动放大；Prefill/Decode 互抢这件事，稠密模型同样成立。

附录命令（Qwen3-235B-A22B-FP8）：标准 2×TP4 用 `VLLM_ROCM_USE_AITER=1`、`-tp 4 --enable-expert-parallel --max-model-len 16384 --max-num-batched-tokens 8192 --distributed-executor-backend mp --no-enable-prefix-caching`，端口 8100 / 8200，proxy 走 `benchmarks/disagg_benchmarks/round_robin_proxy.py`。分离路径再加 `HIP_VISIBLE_DEVICES`、`MORI_DISABLE_AUTO_XGMI=1`、`MORI_IO_ENABLE_NOTIFICATION=0`；Prefill `--port 20005 --max-num-batched-tokens 4096 --gpu_memory_utilization 0.9 --max_num_seqs 64` + producer 配置；Decode `--port 40005` + consumer。Proxy：`moriio_toy_proxy_server.py`。完整命令块见英文对照。

当时的下一步：多机——同一根 RDMA connector，声称不用改代码；按阶段拧旋钮——Prefill 可以更大 token budget / chunked Prefill，Decode 更小 batch，混跑做不到。免责声明：测于 **2026-03-12**。致谢名单见英文对照。

[Router](router.md) 是跨 pod 的 P/D 网关；这篇证明**同一台盒子里**也值得拆。KV 交接的门，和 Mooncake / NIXL / Offloading 是同一类插头，换的是传输实现。
