---
source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 单机也要 P/D：MI300X 上的 MORI-IO

英文对照：`en/vllm/blog/serving/moriio.md`  
原文：https://vllm.ai/blog/2026-04-07-moriio-kv-connector  
2026-04-07。AMD Instinct MI300X，**一台 8 卡节点内**做 Prefill/Decode 分离。Connector：`MoRIIOConnector`（MORI = Modular RDMA Interface）。数字是 Qwen3-235B-A22B-FP8、8 req/s、输入 2000 / 输出 1000。

「P/D 是机房的事」会把单机上的 goodput 留在桌子上。Prefill 是 compute-bound 的大 GEMM；decode 是 memory-bound 的反复搬权重。挤在同一套实例里，一条胖 prefill 能让几十路 decode 口吃；decode 反过来挡新的 prefill。ITL 开始乱跳。

拆开：例如 4 卡 prefill + 4 卡 decode，中间把 KV 交过去。交接可以是数 GB。MORI-IO 用节点内 RDMA。第一次在一对实例之间传之前，ZMQ 在后台交换基址、块大小、stride，RDMA session 缓存起来。


本地图（原文版权仍归原站；学习对照用）：

![read mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/01-read-mode-request-flow-diagram.svg)

![write mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/02-write-mode-request-flow-diagram.svg)

![SLO attainment](../../../../assets/vllm/blog/serving/moriio/03-SLO-attainment.png)

![read mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/04-read-mode-kv-transfer-sequence-diagram.svg)

![write mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/05-write-mode-kv-transfer-sequence-diagram.svg)

## Read vs Write

`VLLM_MORIIO_CONNECTOR_READ_MODE=1` 为 read；不设（或 0）为 write。

- **Read：** proxy 等 prefill 完，把 `remote_block_ids` 转给 decode，decode RDMA **拉** KV，再通知 prefill 放块。
- **Write（默认）：** proxy **同时** 发给 prefill 和 decode。prefill 每算完一层就 RDMA **推**进 decode 预分配的内存；chunked prefill 攒到最后一块再写。decode 轮询写完再开口。不必经 proxy 转 block id。TTFT 通常更好。

## Goodput（不是裸吞吐）

沿 DistServe：同时满足 TTFT **< 1 s** 且 ITL **< 50 ms** 的最大请求率。8 req/s、100 条：

| 部署 | 两条 SLO 都达标 | 相对 |
|---|---|---|
| Standard 1×TP8 | 26/100 | 0.9× |
| Standard 2×TP4 | 30/100 | 1× |
| MORI-IO Read 1P+1D | 70/100 | 2.4× |
| MORI-IO Write 1P+1D | **73/100** | **2.5×** |

标准路径死在 ITL 双峰（一簇约 30 ms，一簇约 150 ms）。分离之后 ITL 违规消失，剩下的失败是 TTFT 超时。Write 比 Read 多过 3 条，因为并发 dispatch 把更多请求按在 1 s 以内。

[Router](router.md) 是跨 pod 的 P/D 网关；这篇证明 **同一台盒子里** 也值得拆。KV 交接的门，和 Mooncake / NIXL / Offloading 是同一类插头，换的是传输实现。
