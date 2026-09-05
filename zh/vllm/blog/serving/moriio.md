---
source: https://vllm.ai/blog/2026-04-07-moriio-kv-connector
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# 单机也要 P/D：MI300X 上的 MORI-IO

英文对照：[en/vllm/blog/serving/moriio.md](../../../../en/vllm/blog/serving/moriio.md)  
原文：https://vllm.ai/blog/2026-04-07-moriio-kv-connector

2026-04-07。AMD 与 Embedded LLM。学习译文，不是官方译本。Connector 类名：`MoRIIOConnector`（MORI = Modular RDMA Interface）。落地 [PR #29304](https://github.com/vllm-project/vllm/pull/29304)；库：[ROCm/mori](https://github.com/ROCm/mori)。数字是 **Qwen3-235B-A22B-FP8**、**8 req/s**、输入 **2000** / 输出 **1000**、同一套 **8× MI300X**。标题成绩：goodput 大约 **2.5×**。原文 Figure 3 是可拖阈值的 Plotly，笔记不收脚本，默认 SLO 见 Table 3。

**TL;DR。** Prefill 和 Decode 抢同一批 GPU，负载一上来 ITL 就会乱跳。这篇把它们拆开，仍住在**一台** 8 卡 MI300X 里，用 AMD 的 MORI-IO 交 KV：相对同卡数的 collocated serving，goodput **2.5×**，吐字更稳。配置见 Table 3 与 [Experimental Details](#experimental-details)。

本地图（原文版权仍归原站；学习对照用）：

![read mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/01-read-mode-request-flow-diagram.svg)

**图注（原文 Figure 1）。** Read mode 请求流。Proxy 串行派发——step 3（Prefill 响应）必须先完成，才有 step 4（派给 Decode）。

![write mode request flow diagram](../../../../assets/vllm/blog/serving/moriio/02-write-mode-request-flow-diagram.svg)

**图注（原文 Figure 2）。** Write mode 请求流。Proxy 同时开火 Prefill 和 Decode（step 2）；Prefill 按层 RDMA WRITE 推进 Decode 的内存（step 3），Decode 在等。

![SLO attainment](../../../../assets/vllm/blog/serving/moriio/03-SLO-attainment.png)

**图注（原文 Figure 4）。** 各请求率下，同时满足 TTFT 与 ITL 目标的请求占比。两条分离路径在全部测试速率上都高于所有标准 serving。

![read mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/04-read-mode-kv-transfer-sequence-diagram.svg)

**图注（原文 Figure 5）。** Read mode 时序。Overhead 1（proxy 串行）和 Overhead 2（RDMA READ）都会加进 TTFT。

![write mode kv transfer sequence diagram](../../../../assets/vllm/blog/serving/moriio/05-write-mode-kv-transfer-sequence-diagram.svg)

**图注（原文 Figure 6）。** Write mode 时序。RDMA WRITE 和 Prefill 计算重叠，Overhead 2 不必整段加进墙钟 TTFT。

## Introduction

上一篇 MoE 优化 [[1]](#ref-1) 讲的是：把一只很大的模型铺进一台 8 卡 AMD Instinct MI300X，用 Tensor / Pipeline / Data / Expert Parallelism。这篇要对付的是另一件会在并发上升时把人卡住的事：Prefill–Decode 互抢。AMD 的 MORI-IO 让这件事可以在**单机**上做完——goodput 更高，延迟更好预报，不必先凑一个多机集群。

HBM 已经装满，算力也配平了，vLLM 跑得很顺——直到你把并发拧上去。然后 ITL 开始乱跳。根因很朴素：Prefill 和 Decode 是两种活，却在抢同一批 GPU。

**Prefill 是 compute-bound：** 整段 prompt 一次并行吃进去，大 GEMM，费用跟输入长度走。

**Decode 是 memory-bandwidth-bound：** 一次一个 token，反复从 HBM 搬权重，每字节上的计算并不多。

两段戏挤在同一套实例里，就会互相挡。一条胖 Prefill 能让几十路正在说话的 Decode 口吃；Decode 反过来挡新的 Prefill 进调度。结果是：两段都跑不满，也都不稳。

## Key Highlights

- **同一套硬件上 goodput 大约 2.5×。** 一台 8 卡 MI300X，把 Prefill 和 Decode 拆开，SLO 达标吞吐明显抬头。
- **负载下消掉 ITL 尖刺。** Decode GPU 专职吐字，Prefill 不再闯进来。
- **单机就能拆，不必先有集群。** Prefill–Decode（PD）disaggregation 整段住在一个节点里，把原来空着的性能拿回来。
- **MORI-IO 快交 KV。** 基于 RDMA 的 KV 搬家，两段之间的交接才跟得上。
- **两种 mode，各有代价。** Write 成绩最好（TTFT 更低）；Read 编排更简单。两边都远好过标准 serving。

## The Misconception: "Disaggregation is Only for Datacenter Clusters"

工程师听见「Prefill–Decode (PD) Disaggregation」，脑子里往往先跳出机房：专职 Prefill 节点、专职 Decode 节点、中间一根 RDMA 布。接着就会说：「我只有一台 8 卡，这事跟我无关。」

这句话会把单机上的 goodput 留在桌子上。PD 拆分可以整段落在一台 8 卡里。若你在意严格的延迟 SLO，它常常才是对的路。

想法直接：两段戏交给两套实例。例如四张卡跑 Prefill，另四张跑 Decode。各自定尺寸、定并行、定调度，单实例那种 head-of-line blocking 就卸掉了。

难的是交接。Prefill 算出来的 KV 必须送到 Decode——可以是数 GB。交得慢，拆开的好处会被交接自己吃掉。

AMD 的答案是 **MORI-IO**：一条基于 RDMA 的 KV connector，已经贡献进 vLLM [[4]](#ref-4)，底下是开源框架 MORI（Modular RDMA Interface）[[5]](#ref-5)。

> **范围：** 这篇只谈单机 PD：一台盒子、8 张卡，把你现有硬件上的 goodput 拧出来。

## The Architecture: Serving with PD Disaggregation

把节点切开，等于从「一只巨兽」换成三件轻量微服务。原文 Table 1：

| Component | Role |
|-----------|------|
| Prefill instance | 处理输入 prompt，产出 KV cache（GPU 0–3） |
| Decode instance | 用转过来的 KV 一个一个吐 token（GPU 4–7） |
| Proxy server | 客户端入口；先走 Prefill，再走 Decode |

<p align="center"><em>Table 1. PD disaggregation 部署零件。</em></p>

两种 mode 都是把 Prefill 产出的 KV 交到 Decode，差在**谁发起传输**、**什么时候发起**：

- **Read mode：** Proxy 等 Prefill 做完，再把 KV 块位置转给 Decode。Decode 开口之前，用 RDMA 把 KV **拉**过来。
- **Write mode：** Proxy 同时派 Prefill 和 Decode。Prefill 每算完一层，就把 KV **推**进 Decode 已经预留好的内存——Prefill 一结束，Decode 就能开口。

### Request Flow in Detail

MORI-IO 两种传输 mode，差在 **谁发起 RDMA**、**proxy 怎么编排两段**。开关是环境变量 `VLLM_MORIIO_CONNECTOR_READ_MODE`。

#### Read Mode — Decode Pulls KV Cache

打开：`export VLLM_MORIIO_CONNECTOR_READ_MODE=1`

Read 里，proxy 对 Prefill 和 Decode **串行**派发：等 Prefill 做完，抽出远端 block ID，再转给 Decode。Decode 拿这些 ID，对 Prefill 的内存做 RDMA read。流程见 Figure 1。

一条请求的时间顺序：

1. **Client → Proxy：** 客户端送来推理请求。
2. **Proxy → Prefill：** Proxy 把 prompt 送到 Prefill 实例（`max_tokens=1`）。
3. **Prefill → Proxy（响应）：** Prefill 返回 `remote_block_ids` 和 `remote_engine_id`，标明 KV 住在哪。
4. **Proxy → Decode：** Proxy 把请求转给 Decode，带着那些远端 block ID。
5. **Decode 拉 KV**（`WAITING_FOR_REMOTE_KVS`）：Decode 对 Prefill 内存发 RDMA read。调度器每一步都跳过这条请求，直到传输完成。
6. **Decode → Prefill（清理）：** 所有 KV 块交完，Decode 通知 Prefill 放块。
7. **Decode → Proxy → Client：** 生成的 token 经 SSE 流回去。

#### Write Mode — Prefill Pushes KV Cache (Default)

打开：不设 `VLLM_MORIIO_CONNECTOR_READ_MODE`，或设 `=0`。

Write 里，proxy 对 Prefill 和 Decode **同时**开火——不等 Prefill 先做完。Prefill 每算完一层，就把 KV 按层推进 Decode 预分配好的内存。流程见 Figure 2。

一条请求的时间顺序：

1. **Client → Proxy：** 客户端送来推理请求。
2. **Proxy → Prefill 并且 Proxy → Decode（并发）：** Proxy 并行发出两个请求。Prefill 请求带着 Decode 的连接信息；Decode 请求带着 Prefill 的。Proxy **不**阻塞等待 Prefill 响应。
3. **Prefill 推 KV：** 每算完一层，`save_kv_layer` 发一次 RDMA write，直接写进 Decode 预分配的 KV 块。Chunked Prefill：块先攒着，到最后一块才真正发起写。
4. **Decode 等写完**（`WAITING_FOR_REMOTE_KVS`）：Decode 调度器每一步轮询 `pop_finished_write_req_ids`，直到所有块到齐。
5. **Decode 生成：** KV 块一齐，请求立刻进 ready 队列，开始自回归生成。
6. **Decode → Proxy → Client：** Token 经 SSE 流回去。

Proxy 里关键差别就是一处条件：

```python
# examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py

if TRANSFER_TYPE == "READ":
    # Serial: wait for prefill to finish, extract block IDs for decode to pull.
    prefill_response = await send_prefill_task
    req_data["kv_transfer_params"]["remote_engine_id"] = prefill_response[
        "kv_transfer_params"
    ]["remote_engine_id"]
    req_data["kv_transfer_params"]["remote_block_ids"] = prefill_response[
        "kv_transfer_params"
    ]["remote_block_ids"]

# In WRITE mode, execution falls through here immediately —
# no await on send_prefill_task. Both phases are already in flight.
decode_request_task = asyncio.create_task(
    start_decode_request(decode_instance_endpoint["request_address"], req_data, request_id)
)
```

Read 必须经 proxy 转 `remote_block_ids`：Decode 要知道该拉 Prefill 侧哪几块。Write 由 Prefill 做主，直接推到 Decode 的地址——不必转 block ID。

### Read Mode vs. Write Mode: At a Glance

底下，MORI-IO（在 vLLM 里暴露为 `MoRIIOConnector`）管 KV 交接。不论哪种 mode，一对实例第一次 RDMA 之前，MORI-IO 会经 ZMQ 做一次元数据交换：KV cache 基址、块大小、每层 tensor stride。这次握手跑在**后台线程**，不挡引擎循环；换来的 RDMA session 会缓存，后续请求共用。

握手和 RDMA 传输两边一样——差别全在 proxy 派发层和传输方向。原文 Table 2：

| Property | Read Mode | Write Mode |
|----------|-----------|------------|
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `=1` | 不设（或 `=0`） |
| RDMA 方向 | Decode 从 Prefill 拉 | Prefill 推向 Decode |
| Proxy 派发 | 串行（await Prefill → 再派 Decode） | 并发（Prefill 与 Decode 并行） |
| 经 proxy 转 `remote_block_ids` | 要 | 不要 |
| KV 清理信号 | 拉完后 Decode 通知 Prefill 放块 | Prefill 按请求跟踪写完成 |

<p align="center"><em>Table 2. Read 与 Write 的关键差别。</em></p>

## Results: 2.5x Higher Goodput

先看拆开之后到底换来了什么，再谈怎么配。

### Why Goodput, Not Throughput

裸吞吐会骗人——系统可以撑很高的请求率，同时对大多数人悄悄违约。他们用 **goodput** 做主指标，跟 DistServe 的口径 [[3]](#ref-3)：

**Goodput** = 能同时满足 TTFT &lt; *T_ttft* **并且** ITL &lt; *T_itl* 的最大请求率（req/s）。

成本和体验（SLO 达标）收进同一个数。他们的 SLO：**TTFT &lt; 1 秒**，**ITL &lt; 50 ms / token**。两条都过，这条请求才算进 goodput。

### Headline Result

**Figure 3** 是请求率 = 8 时的 goodput（原文 Plotly：每根柱一条请求，灰柱至少破一条 SLO；滑条可改阈值。默认 TTFT &lt; 1 s、ITL &lt; 50 ms。笔记不收交互控件）。默认阈值下的计数是 Table 3：

| Metric | Standard (1× TP8) | Standard (2× TP4) | MORI-IO Read (1P+1D) | MORI-IO Write (1P+1D) |
|--------|-------------------|---------------------|---------------------|----------------------|
| 两条 SLO 都达标的请求 | 26/100 | 30/100 | 70/100 | 73/100 |
| 主要失败模式 | ITL 尖刺（P99 ITL ≫ 50 ms） | ITL 尖刺（双峰：约 30 ms 与约 150 ms） | 部分请求 TTFT 超过 1 s | 部分请求 TTFT 超过 1 s |
| 相对 goodput | 0.9× | 1× | 2.4× | 2.5× |

<p align="center"><em>Table 3：请求率 = 8 时的 SLO 达标。负载：Qwen3-235B-A22B-FP8，ISL=2000，OSL=1000，8 req/s，100 条请求。完整配置见 Experimental Details。相对 goodput 以 Standard (2× TP4) 为基线。</em></p>

标准 serving 死在 ITL 分成两簇——高延迟那簇大约 150 ms，远过 50 ms。两条分离路径把 ITL 违规**整段消掉**；剩下的失败是请求率往上走时 TTFT 越线。Write 略胜 Read（73 对 70），因为 proxy 并发派发压低了 TTFT，更多请求留在 1 s 以内。

### SLO Attainment Across Request Rates

**Figure 4** 把请求率从 0.5 扫到 10：

- **Standard serving (1× TP8)：** 低请求率就开始 ITL 违规，整条扫掠都由它主导。rate = 8 时 26/100。
- **Standard serving (2× TP4)：** 掉得很快——rate 0.5 时 100%，rate 1 大约 60%，到 rate 2 塌到大约 25% 然后平台。ITL 违规很早就饱和。
- **MORI-IO Read (1P+1D)：** 大约到 rate 5 仍 100%，然后慢慢降，rate 10 大约 44%（TTFT 开始越线）。
- **MORI-IO Write (1P+1D)：** 大约到 rate 5.5 仍 100%，然后慢慢降，rate 10 大约 46%（同样是 TTFT）。

## Understanding the Trade-offs

### Why ITL Improves

标准部署里，Prefill 和 Decode 共用同一个 vLLM 引擎，在每个 batch 里抢调度。一次 Prefill——整段输入一次前向——远长于一步 Decode。同一 batch 里每一条 Decode 都要等这次 Prefill 完，才能吐下一个 token，ITL 被直接撑大。

拆开之后，Decode 引擎**只**跑 Decode batch。没有计算很重的 Prefill 来打断步频，ITL 变得稳、可预报，跟门口新来多少请求无关。这笔好处 Read / Write **一样**——Decode 引擎两边都被隔开了。

### Why TTFT Gets Worse

另一面：拆开会在通往第一个 token 的路上加开销。标准 serving：

```
TTFT = queue + prefill_forward_pass + sample_T1 + detokenize + SSE_encode + network
```

Read 多插两步（Figure 5）：

```
TTFT = queue(prefill) + prefill_forward_pass
     + [proxy serialization: await prefill, dispatch to decode]  <- Overhead 1
     + RDMA transfer (WAITING_FOR_REMOTE_KVS)                   <- Overhead 2
     + queue(decode) + sample_T1 + detokenize + SSE_encode + network
```

Write（Figure 6）：

```
TTFT ≈ max(
           queue(prefill) + prefill_forward_pass + RDMA_write_time,
           queue(decode)
       ) + sample_T1 + detokenize + SSE_encode + network
```

Write 去掉 Overhead 1。Proxy 同时派两套实例，Decode 排队和 Prefill 计算重叠。剩下的代价——RDMA 传输本身——和 Read 里的 RDMA read 结构上等价。

#### Overhead 1: Proxy Serialization (Read Mode Only)

Read 里，proxy 要等完整份 Prefill 响应，才派 Decode。整段 Prefill 计算时间，外加一次 proxy 往返，都会加进客户端看见的 TTFT。Write 跳过这块——Decode 请求在 Prefill 结束之前已经在飞。

```python
# examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py

if TRANSFER_TYPE == "READ":
    # In read mode, prefill and decode are executed serially.
    prefill_response = await send_prefill_task
    req_data["kv_transfer_params"]["remote_engine_id"] = prefill_response[
        "kv_transfer_params"
    ]["remote_engine_id"]
    req_data["kv_transfer_params"]["remote_block_ids"] = prefill_response[
        "kv_transfer_params"
    ]["remote_block_ids"]
```

#### Overhead 2: RDMA Transfer Wait

Decode 实例接到请求后进入 `WAITING_FOR_REMOTE_KVS`。调度器每一步都跳过它，直到 RDMA 传完，再立刻挪进 ready 队列。

```python
# vllm/v1/request.py

WAITING_FOR_REMOTE_KVS = enum.auto()

# vllm/v1/core/sched/scheduler.py
# KVTransfer: skip request if still waiting for remote kvs.

if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        request.status = RequestStatus.WAITING
    else:
        logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.",
                     request.request_id)
        self.waiting.pop_request()
        skipped_waiting_requests.prepend_request(request)
        continue
```

Read 里，这段等待发生在 Prefill **已经**结束之后。Write 里，Decode 请求一到就开始等——和另一套实例上还在进行的 Prefill 计算重叠。

**收束：** 拆开换来稳、可预报的 ITL，代价是第一个 token 要多等一会儿。多多久看 mode。Read：TTFT 至少多一整次 Prefill 前向（proxy 串行）再加 RDMA 传输。Write：proxy 串行没了——TTFT 只多 RDMA，而且和 Prefill 计算重叠，净惩罚更小。ITL 的好处两边一样。

### When Should You Use This?

原文 Table 4：

| Your situation | Recommendation |
|----------------|----------------|
| 生产负载下 ITL p99 超过 SLO | 拆——这是主用例 |
| TTFT 是绑死的约束（例如 chatbot UX） | 标准 serving 可能更合适 |
| 高并发、长 prompt | 拆——Prefill 干扰在这里最狠 |
| 低请求率、短 prompt | 标准 serving 够用 |

<p align="center"><em>Table 4: 部署决策。</em></p>

## How to Set It Up

成绩看过了，怎么落地。三件：Prefill 实例、Decode 实例、proxy。vLLM 官方 disaggregated prefill 文档见 [[2]](#ref-2)。

### Prefill Instance

Prefill 是 KV 生产者（`kv_role: kv_producer`）。它处理输入 prompt，算出 KV，让 Decode 经 RDMA 来读。

```bash
vllm serve <model> \
  ...
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

启动时，实例经 ZMQ 向 proxy 注册：角色、HTTP 地址、handshake / notify 端口、并行配置。之后继续周期性注册，好让 proxy 发现它不在了。

### Decode Instance

Decode 是 KV 消费者（`kv_role: kv_consumer`）。它在 Prefill 完成之后从 proxy 接到请求，再经 RDMA 拉 KV。

```bash
vllm serve <model> \
  ...
  --gpu_memory_utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "40005",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'
```

### Proxy Server

Proxy 是轻量 HTTP 服务，编排两段。它在 `proxy_ping_port` 上经 ZMQ 听实例注册，用 round-robin 转发每条请求。

```bash
python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py
```

READ 里，proxy 等 Prefill 完成，从响应抽出 `remote_block_ids`，交给 Decode，让它知道该拉哪几块 KV。

### Port Reference

每个实例几条端口，原文 Table 5。`MoRIIOConfig`（见 `moriio_common.py`）会加 per-rank 偏移：

| Port | Purpose |
|------|---------|
| `proxy_ping_port` | 实例向 proxy 注册的 ZMQ 端点 |
| `http_port` | vLLM HTTP；proxy 把推理请求转到这里 |
| `handshake_port` | 一次性元数据交换：consumer 拿到 producer 的 KV 布局 |
| `notify_port` | 按请求同步：Prefill 告诉 Decode KV 块已就绪 |

<p align="center"><em>Table 5: MORI-IO 端口。</em></p>

## Experimental Details

### Setup

环境可按仓库里的 Dockerfile 复现：`Dockerfile.rocm_base`（MORI commit `2d02c6a9`，来自 [ROCm/mori](https://github.com/ROCm/mori)）和 `Dockerfile.rocm`（vLLM main，[vllm-project/vllm](https://github.com/vllm-project/vllm)）。

**硬件：**

- GPU：8× AMD Instinct MI300X（gfx942）
- CPU：2× AMD EPYC 9654 96-Core Processor

**软件栈：**

- ROCm Driver：6.10.5（AMDGPU）
- 容器：rocm/vllm-dev（ROCm 7.0.51831-a3e329ad8）
- vLLM：0.16.0rc1.dev1+gc46b0cd0a（git sha：c46b0cd0a）
- PyTorch：2.9.1+git8907517（ROCm 7.0.51831-a3e329ad8）
- MORI 库：commit [`c365eaed`](https://github.com/ROCm/mori/commit/c365eaed02b13e6b8f2e9c8215b21516d86856ce)

**Benchmark 配置：**

- 模型：Qwen/Qwen3-235B-A22B-FP8
- 输入序列长度：2000 token
- 输出序列长度：1000 token
- 数据集：random
- 负载：共 100 条请求
- 请求率：0.5 到 10（步长 0.5）

### Baseline Configurations

原文 Table 6 对比的四种配置：

| Configuration | Description |
|---------------|-------------|
| Standard (1× TP8) | 单套 vLLM，8× MI300X 全用（TP=8），expert parallelism。一只引擎混跑 Prefill 和 Decode。 |
| Standard (2× TP4) | 两套相同的 vLLM，各 4× MI300X（TP=4）+ expert parallelism。Round-robin proxy 均分请求。两套都混跑 Prefill 和 Decode。 |
| MORI-IO Read (1P+1D) | 一套 Prefill（GPU 0–3）+ 一套 Decode（GPU 4–7），各 TP=4 + expert parallelism。两边都设 `VLLM_MORIIO_CONNECTOR_READ_MODE=1`。Proxy 串行：等 Prefill 返回 `remote_block_ids` 再转 Decode。Decode 用 RDMA 拉 KV。Prefix caching 关掉。 |
| MORI-IO Write (1P+1D) | 同一套切卡。KV 走 MORI-IO write mode。有状态的 proxy 做两段路由。Prefix caching 关掉——MORI-IO connector **要求**关掉。 |

<p align="center"><em>Table 6: 基线配置。</em></p>

> **为什么这样比？** Standard (2× TP4) 和分离配置用同一套总卡数（8× MI300X），都切成两组 4 卡，才是同卡数的公平对照。唯一差别：每组是混跑 Prefill+Decode（标准），还是专职 Prefill 或 Decode（分离）。Standard (1× TP8) 是额外参照：8 张卡装进一只引擎。

**外推：** 数字来自 MoE 模型（Qwen3-235B-A22B-FP8）。Prefill / Decode 互抢是 transformer 推理的基本形状，稠密模型同样成立。MoE 往往把效应放大：expert routing 让每步计算更抖，ITL jitter 更明显。

## Conclusions and Way Forward

这篇要证明的是：PD 拆分不是机房专属——一台 8 卡就能量出成绩。GPU 按阶段专职，MORI-IO 用 RDMA 交 KV，goodput 到 2.5×，并把 collocated 部署里那些 ITL 违规消掉。

### What's Next

- **多机部署：** 生产里 Prefill / Decode 可以跨节点——MORI-IO 已经走网上的 RDMA，同一条 connector 跨主机声称不用改代码。
- **按阶段拧旋钮：** 实例专职之后，Prefill 可以追计算吞吐（更大 token budget、chunked Prefill），Decode 追低延迟（更小 batch、更严的调度）。混跑做不到这种独立拧法。

## Appendix: Reproducible Configurations

预构建 nightly 镜像：[rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev)。或用 vLLM 仓库的 `Dockerfile.rocm_base` / `Dockerfile.rocm` 从源码构建（MORI commit [2d02c6a9](https://github.com/ROCm/mori/commit/2d02c6a9)，vLLM commit [c46b0cd0a](https://github.com/vllm-project/vllm/commit/c46b0cd0a)）。

下面是全部 benchmark 的完整命令行。每条都带环境变量、并行旗标、以及 Qwen3-235B-A22B-FP8 在 AMD Instinct MI300X 上的部署参数。

### Standard Serving

```bash
# Instance 1 (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --port 8100

# Instance 2 (GPU 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --port 8200

# Proxy
cd <path_to>/vllm
python benchmarks/disagg_benchmarks/round_robin_proxy.py
```

### Disaggregated Serving

```bash
# Prefill instance (GPU 0-3)
export VLLM_MORIIO_CONNECTOR_READ_MODE=1    # unset for write mode
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export MORI_DISABLE_AUTO_XGMI=1
export MORI_IO_ENABLE_NOTIFICATION=0

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --port 20005 \
  --max-num-batched-tokens 4096 \
  --distributed-executor-backend mp \
  --gpu_memory_utilization 0.9 \
  --max-model-len 16384 \
  --max_num_seqs 64 \
  --no-enable-prefix-caching \
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

# Decode instance (GPU 4-7)
export VLLM_MORIIO_CONNECTOR_READ_MODE=1    # unset for write mode
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=4,5,6,7
export MORI_DISABLE_AUTO_XGMI=1
export MORI_IO_ENABLE_NOTIFICATION=0

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --enable-expert-parallel \
  --port 40005 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --distributed-executor-backend mp \
  --gpu_memory_utilization 0.9 \
  --max-model-len 16384 \
  --max_num_seqs 64 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "http_port": "40005",
      "proxy_ping_port": "36367",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'

# Proxy
cd <path_to>/vllm
python examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py
```

## Acknowledgements

**AMD：** Hongxia Yang, Gilbert Lei, Mingzhi Liu, Niko Ma, Tian Di, Randall Smith, Feiyue Zhai, Peng Sun，以及 MORI 团队。

**Embedded LLM：** Pin Siang Tan, Jun Kang Chow, Ye Hur Cheong, Vensen Mu, Jeff Aw, Tun Jian Tan，以及 Embedded LLM 团队。

## References

1. <a id="ref-1"></a> AMD and Embedded LLM, "The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism" <https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html>
2. <a id="ref-2"></a> vLLM Disaggregated Prefill Documentation <https://docs.vllm.ai/en/latest/features/disagg_prefill/>
3. <a id="ref-3"></a> DistServe: Maximizing Goodput in LLM Serving <https://haoailab.com/blogs/distserve/>
4. <a id="ref-4"></a> MORI-IO Connector PR #29304 <https://github.com/vllm-project/vllm/pull/29304>
5. <a id="ref-5"></a> MORI (Modular RDMA Interface) <https://github.com/ROCm/mori>

## Disclaimer

测于 **2026-03-12**，在 AMD Instinct MI300X 上量推理 goodput。

**硬件配置**

- MI300X：AMD EPYC 9654 96-Core Processor 服务器，8× AMD Instinct MI300X（192GB，750W），NPS1（每 socket 1 个 NUMA），2.2TiB（24 DIMM，4800 MT/s，96 GiB/DIMM）

**软件配置**

Ubuntu 22.04 LTS，Linux kernel 5.15.0-153-generic，ROCm Driver 6.10.5（AMDGPU），ROCm 7.0.51831-a3e329ad8，PyTorch 2.9.1+git8907517，vLLM 0.16.0rc1.dev1+gc46b0cd0a，MORI 库 commit c365eaed

服务器厂商配置可能不同，成绩会跟着变。表现还取决于配置、软件、vLLM 版本、以及是否用上最新驱动和优化。

[Router](router.md) 是跨 pod 的 P/D 网关；这篇证明**同一台盒子里**也值得拆。KV 交接的门，和 Mooncake / NIXL / Offloading 是同一类插头，换的是传输实现。
