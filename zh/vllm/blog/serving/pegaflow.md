---
source: https://vllm.ai/blog/2026-05-18-pegaflow
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# PegaFlow：让 KV 活得比推理进程更长

英文对照：`en/vllm/blog/serving/pegaflow.md`  
原文：https://vllm.ai/blog/2026-05-18-pegaflow  
2026-05-18。Novita AI。Rust 守护进程 + 外部 KVConnector，不改 vLLM 源码。`vllm>=0.20.0`。图在原网页。和 [Mooncake](mooncake.md)、[KV offload](kv-offload.md) 同一扇门，把「池子」做成一台独立服务。

KV 可以占掉每机几百 GiB，分配和预热很慢，却常常比当时那波请求活得久。绑在 worker 进程里：崩溃、滚动升级、换模型，池子一起死，pinned 内存要重新申请。PegaFlow 把运行时搬到每机一台 daemon：pinned DRAM、SSD、拓扑、RDMA、索引、后台任务归它；vLLM 用 CUDA IPC（数据）和 gRPC（控制）连本地进程。一台 cache 伺候多引擎、多模型，namespace 隔离，共享同一份内存/盘/网。

## 数字（演示 / 生产抽样）

8×RTX 5090、Qwen3-8B TP8、dummy 权重、eager：只测约 **500 GiB** 主机池。池子在 vLLM 里 71.4 s ready；预先进 PegaFlow 后 vLLM **33.2 s**（**2.15×**）。

同一 500 GiB 预算、八个 Qwen3-8B：共享池 **11.97 req/s**、TTFT 5.26 s、命中 52%；八个隔离 62.5 GiB 池 **7.68 req/s**、8.22 s、11.8%。吞吐 **+56%**，命中约 **4.4×**。

DeepSeek-V3.2 MLA TP8：逻辑 KV 存一份 vs 每 rank 一份。吞吐 **1.81 vs 1.05**（+72%），命中 97% vs 65%。

内部集群、每节点 8×400 Gbps：≥1 GiB 的远端前缀拉取平均 **194 GB/s**（P99 250、峰值 261.6）。24 GiB 大约 **100 ms** 拉完——可以换掉数秒的 GPU prefill。

## 三层与命中率天花板

L1 pinned DRAM；L2 远端 DRAM（one-sided RDMA READ，连上之后远端不占 CPU）；L3 本地 SSD（`io_uring`）。单盘内部测峰值约 6.9 GB/s，稳态约 6.5–6.6，换更稳的尾延迟；RAID0 近似线性。扫描型流量可开 TinyLFU 准入（默认关）。

在线命中率会骗人。他们用 HyperLogLog 估理论上限 `r* = (N − U) / N`（N 块请求、U 首次见到的独特块），24 h 窗口 <1 MiB、误差约 0.8%。实测贴近上限 → 再加容量也没用；远低于上限 → 容量/准入/跨节点发现还有戏；上限本身很低 → 负载本来就没什么可复用。

```bash
uv pip install pegaflow-llm   # CUDA 12；cu13 换 pegaflow-llm-cu13
pegaflow-server --pool-size 30gb --ssd-cache-path <p> --ssd-cache-capacity 512gb
# 在线建议 --use-hugepages
vllm serve <model> --kv-transfer-config '{
  "kv_connector": "PegaKVConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "pegaflow.connector"
}'
```

`PEGAFLOW_HOST` / `PEGAFLOW_PORT` 默认 `http://127.0.0.1:50055`。多机先 `pegaflow-metaserver`，每节点 `--nics` + `--metaserver-addr`；P2P 时 `--addr` 必须是对端能连的 IP，不能是 `0.0.0.0` / `127.0.0.1`。公开 H800 / Llama-3.1-8B 参考：热缓存 TTFT 572.5→61.5 ms。仓库：novitalabs/pegaflow。
