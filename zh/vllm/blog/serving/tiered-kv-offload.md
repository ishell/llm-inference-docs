---
source: https://vllm.ai/blog/2026-09-10-tiered-kv-offloading
lang: zh
voice: book-zh
fetched: 2026-09-11
---

# 分层 KV Offload：一切先经过 Host，再 cascade 到盘和邻居

英文对照：[en/vllm/blog/serving/tiered-kv-offload.md](../../../../en/vllm/blog/serving/tiered-kv-offload.md)  
原文：https://vllm.ai/blog/2026-09-10-tiered-kv-offloading  
2026-09-10。署名 **Or Ozeri, Danny Harnik, Ronen Schaffer, Itay Etelis, Varun Sundar Rabindranath**。学习译文，不是官方译本。本机 CPU offload 的后续：[kv-offload.md](kv-offload.md)。同一家 KVConnector：[mooncake.md](mooncake.md)、[pegaflow.md](pegaflow.md)。**v0.22** 起进 vLLM。用法：[docs.vllm.ai … kv_offloading_usage](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)。混合模型邻居：[glm53-hisparse.md](glm53-hisparse.md)。路由：[agentx.md](agentx.md)。

长上下文和多轮对话会生出很大的 KV cache。加速器内存（GPU HBM）装满之后，已经算过的 KV 会被挤走。下一条还要用它的请求，就得从头再算。

**分层 KV offload 把被挤走的 KV 留在 host、存储和远端邻居上。** 不必重算，vLLM 从更低的一层 reload。有了二级层，KV 还能**跨节点共享**：水平扩 cache、从共享存储给新实例预热、以及为分离 serving 或负载均衡在 peer 之间搬家。

## 以 Host 为中心

核心原则：**所有 KV 都先经过 host 内存（CPU DRAM）**。

Offload：加速器 → 先到 host，再从 host 到文件系统 / 对象存储 / 远端 peer。Reload：反过来。二级层先把数据提升进 host，加速器再从 host 装。

![architecture](../../../../assets/vllm/blog/serving/tiered-kv-offload/01-architecture.svg)

**图注（原文架构图）。** 所有 KV 经过 host 这一层主键。二级层把容量扩到 host DRAM 装不下的地方。Offload 时 chunk cascade 到所有层。Reload 时，谁先握着这块 chunk 谁来服务。

### 尽快释放加速器，按需再分配

加速器 → host 是一次本地 PCIe 拷贝。**这次拷贝一完成，加速器内存就释放**——二级层传输还没开始。盘上的写、网上的发、远端 RDMA 都从 host 那份拷贝走，不再碰加速器。**Reload 时，要等数据已经在 host 里才分配加速器内存**，不会在等分层传输时先占着。加速器内存只在真正用到的时候握着。

![offload flow](../../../../assets/vllm/blog/serving/tiered-kv-offload/02-offload-flow.svg)

**图注（offload 时间线）。** t2 释放加速器内存——host 拷贝一完成。二级层的写从 host 拷贝异步继续。

### 合并 I/O

`tensor_parallel_size=8` 时，每张卡握一份 KV 分片。框架把**所有分片收进一块共享的 host 区域**。

![consolidated I/O](../../../../assets/vllm/blog/serving/tiered-kv-offload/03-consolidated-io.svg)

**图注（合并 I/O）。** 多张加速器的分片汇进一块共享 host。二级层看见的是更少、更大的 I/O。

### 规范内存布局

Host 区域用一份规范布局：一页存一层的一块，跨 TP rank 的全部 KV head 收进一段连续区域。找一块 chunk 就是算偏移。

固定的 host 侧布局保证：即便各节点的 GPU 布局不同——加速器种类、attention backend（FlashAttention、FlashInfer、Triton）、并行配置——共享仍然正确。**配置不同的节点可以直接共享 KV**，不用重映射。TP=2 和 TP=4 对同一份 KV 在 host 侧产出相同的 chunk。

### 二级层可以很简单

每个 vLLM 实例一个进程。传输用 CPU 库（POSIX I/O、S3 SDK、RDMA verbs），从不碰加速器内存或 API。不必跨多个进程协调，也不必懂加速器专用布局。

## Offload 和 Reload 怎么走

操作单位是 **chunk**——覆盖一组 token 的固定大小 KV。默认一块 chunk 对应一块加速器 block。`blocks_per_chunk` 可以把 chunk 做大（对 host 和二级层是更大的 I/O）。

### Offload 路径

新 KV chunk 经异步 DMA 从加速器到 host。**加速器内存立刻释放**，二级层传输还没开始。分层管理器再同时 cascade 到**所有**已配置的二级层，读的是 host 那份拷贝。

Host 这一层主键是正经的 **LRU/ARC cache**，不是暂存缓冲。Chunk 留在 host 内存里直接服务后续命中。只有 host 容量用尽，才驱逐最少使用的那些——即便那时，它们仍活在已经收下它们的二级层里。

### Reload 路径

调度器先查 host cache。Host miss 时按配置顺序问二级层；谁先握着这块 chunk 谁来服务。该层再把 chunk 异步提升回 host；这段时间调度器收到 `RETRY`，下一轮再查。

同一条请求里的不同 chunk 可以来自不同层（文件系统对远端 peer）。

## 二级层

### 文件系统

每块 KV chunk 是本地或网络存储上的一个文件。按内容寻址：相同 token 序列映射到同一把钥匙，匹配的输入自动共享。

多个 vLLM 实例共用同一个挂载点（NAS，或同一台机器上的多个实例）时，**KV 自动共享**，不用再配。

要点：**查找不阻塞**、**原子写**、**读/写分开的线程池**。

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --kv-transfer-config '{
        "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 107374182400,
            "secondary_tiers": [{"type": "fs", "root_dir": "/mnt/kv-cache"}]
        }
    }'
```

`cpu_bytes_to_use` **107374182400** 是 100 GiB。

### 对象存储

KV chunk 经 NIXL 进 S3 兼容对象存储。同一套内容寻址。每 GB 通常比高性能文件存储便宜，实例之间仍能共享。

```bash
--kv-transfer-config '{
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 107374182400,
        "secondary_tiers": [{
            "type": "obj",
            "bucket": "my-kv-cache",
            "endpoint_override": "http://minio:9000"
        }]
    }
}'
```

### Peer-to-Peer（P2P）

跨实例在网上共享 KV。协调用 ZMQ，大宗传输用 NIXL 的 RDMA。**只在 host 之间**——两边都不碰加速器内存。

P2P 这一层**不决定**从哪个 peer 拉。那是编排层的事（例如 [llm-d](https://github.com/llm-d/llm-d)）。编排层通过请求的 `kv_transfer_params` 驱动跨节点传输。细节：[usage guide 的 orchestration-layer protocol](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/#orchestration-layer-protocol)。

```bash
--kv-transfer-config '{
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 107374182400,
        "secondary_tiers": [{"type": "p2p", "host": "10.0.0.1", "port": 5710}]
    }
}'
```

两种用法：

#### Prefill/Decode 分离

Prefill 实例算出 KV chunk，放在自己的 host 层。Decode 实例经 RDMA 从 prefiller 的 host 内存拉过来。

相对 GPU 侧 P/D 的一条好处：**合并 I/O 把许多细碎的每 GPU 传输收成更少、更大的 RDMA**。配合 *chunked prefill*，每完成一块 Prefill chunk 立刻可以传——计算和搬家重叠，TTFT 下来。

#### 负载均衡

把 KV 从过载的实例搬到还有容量的实例。任何节点都可以从任何 peer 拉。

llm-d 上的 P2P 另见 [llm-d.ai 博客](https://llm-d.ai/blog/p2p-kv-cache-sharing-llm-d)。

## 混合模型

和 vLLM 的 hybrid memory allocator 接在一起。Full attention、sliding window、MLA、Mamba 透明处理。

规范布局把所有 KV 格式收成统一的 byte buffer。**Host 上每块 chunk 的字节大小固定**，不管里面有哪些层类型。不同层类型往同一块 chunk 里装的 token 数不同——Mamba 状态层每块覆盖的 token 远多于 full attention，所以 offload 更少。

- **Sliding window 层**只 reload 窗口里的 token，不是整段历史。
- **状态空间层**（Mamba）把状态和 attention KV 一起 offload、reload。

页上点名：DeepSeek V4、GLM 5.3、Nemotron 3 以及其它。

## 可观测性

Prometheus 指标走 vLLM 标准 `/metrics`：

- **Host cache 占用**——主键这一层的填充比
- **传输吞吐**——加速器 ↔ host 的字节和时间
- **每层延迟**——查找和数据传输
- **每层命中率**

二级层可以**自己定义指标**（counter、histogram、gauge），自动注册、自动暴露。

## KV events

Chunk 在层之间移动时，框架发出结构化的 **KV events**：哪些 chunk 被存下或挤走、来自哪一层、本地还是远端。二级层也可以发自己的事件。

编排系统用这些事件把请求送到最可能命中 cache 的实例。[llm-d](https://github.com/llm-d/llm-d) 和 [Dynamo](https://github.com/ai-dynamo/dynamo) 写在页上。llm-d 还用这些事件编排 peer 之间的 P2P 传输。

## 加一层新的二级层

核心四个方法：

```python
class SecondaryTierManager(ABC):

    def lookup(self, key, req_context) -> LookupResult:
        """Does this tier have a chunk? Returns HIT, MISS, or RETRY."""

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """Start async store from host to this tier."""

    def submit_load(self, job_metadata: JobMetadata) -> None:
        """Start async load from this tier to host."""

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll completed transfers."""
```

构造时每层拿到共享 host 区域的一份 **直接 memoryview**。`submit_store()` 从这里读；`submit_load()` 往这里写。**没有中间拷贝，也不序列化。** 每层自己管驱逐。

树内参考实现：[`vllm/v1/kv_offload/tiering/example/`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/kv_offload/tiering/example)。仓外二级层：在配置里写 `module_path`，vLLM 加载自定义的 SecondaryTierManager，不必改主干。

## 性能：把用户数再往上推

主要收益：从更便宜的一层 reload，避开反复 Prefill。

并发对话少时，各种缓存看起来都不错——HBM 装得下工作集。池子变大之后：

- **大约到 64 路对话**——HBM 装得下工作集，各方法都好。
- **64–128 路**——HBM 装满；不开 offload 吞吐会掉。CPU offload 还能顶住。
- **超过 128 路**——CPU cache 也满了。存储 offload 仍保持高命中，吞吐比其它选项高出一倍以上。

![performance](../../../../assets/vllm/blog/serving/tiered-kv-offload/04-performance.svg)

**图注（页上性能图）。** 存储延迟高于 CPU，所以到不了峰值吞吐。规模大了，选项是存储命中还是整段重算——存储赢。

**页上的实验设置：**

- 模型：`Qwen/Qwen3.6-35B-A3B`，2× NVIDIA H100（TP=2）
- 存储层：本地 NVMe 上的文件系统后端
- 负载：多轮，首轮 12K token prompt + 每轮 4K token，8 轮
- 最大请求并发：64
- 只测 prefiller 吞吐（Prefill/Decode 已分离）

复现脚本：[neuralmagic/fs-offload-experiments](https://github.com/neuralmagic/fs-offload-experiments)。

## 致谢

Liran Schour、Chang Guo、Srinivas Krovvidi、Rotem Shavitt、Effi Ofer、Omer Paz、Kfir Toledo、Michal Malka，以及社区的代码、评审和反馈。
