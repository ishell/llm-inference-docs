---
source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Decode Context Parallelism：长上下文别再按头去切 KV

英文对照：[en/vllm/blog/performance/dcp.md](../../../../en/vllm/blog/performance/dcp.md)  
原文：https://vllm.ai/blog/2026-08-07-decode-context-parallelism  
2026-08-07。作者 **Seonghee Lee, Sungsoo Ha, Omri Almog (NVIDIA), Lucas Wilkinson (Red Hat AI)**。vLLM 支持 DCP 已近一年；agent 把上下文拉到 **64K–1M** 之后，这篇才把它写清楚。CLI：`--decode-context-parallel-size`（帮助里也有 `-dcp`）。NVIDIA TensorRT-LLM 侧相近的方向叫 [Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md)。文档：[Decode Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel)。摘要里的成绩：相对标准 TP，长上下文 agent 负载上吞吐约 **3×**。

本地图（原文版权仍归原站；学习对照用）。原文 2.2 节说「见下表」；活页上的表是 JS / Plotly 控件，笔记不收。正文数字都在。

## 1. Introduction

长上下文推理正在变成 agent 的基本功：助手要读大仓库、带着长对话。Agent-trace 基准现在从 64K 一路到 1M token，KV 跟着胀。基线 **tensor-parallel (TP)** 按 **attention head** 切 KV，有一块硬地板。

现代模型两种注意力都会撞上这块地板。

- **Grouped-query attention (GQA)：** KV head 本来就少。TP 最多切到**每卡一个 KV head**；`tensor_parallel_size` 超过 KV head 数，cache 就开始在 GPU 之间**复制**。
- **Multi-head latent attention (MLA)：** Key/Value 收成一份低秩 **latent**，所有 query head 共用——等于只有一个 KV head。普通 TP 没头可切，latent 在每个 TP rank 上**整份复制**。

复制把 HBM 吃掉，并发上不去，吞吐和每 token 成本一起坏。[分布式推理](../serving/distributed-inference.md) 那张「TP 给 KV 腾房间」的超线性图，在 MLA 上会反过来。

Decode Context Parallelism 按**序列维**切：每卡只存、只读自己那一段 KV。房子空出来，每卡才能再加人、batch 才能再涨。要高带宽的卡间互联，才能在许多长 agent 同住时仍保持可交互。

vLLM 支持 DCP 已近一年。这篇现在才写，是因为长上下文 agent 把收益推到了台前。

![kv parallelism overview](../../../../assets/vllm/blog/performance/dcp/01-kv-parallelism-overview.svg)

**图注（原文）。** 纯 TP 下两种注意力都在复制 KV 上浪费内存：GQA 只能切到每卡一个 KV head，再加就复制；MLA 像只有一个 KV head，latent 在每个 rank 上整份复制。DCP 按序列维切，每卡握着独一无二的切片，容量不浪费在副本上。

## 2. Performance Results

同一套 GPU、模型、负载；只改 Decode 时 KV 怎么切。

![figure 1](../../../../assets/vllm/blog/performance/dcp/02-figure-1.png)

![figure 2](../../../../assets/vllm/blog/performance/dcp/03-figure-2.png)

### 2.1 Dataset

公开的长上下文 agent 轨迹，**Mooncake-trace** 格式：[JSONL](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl)（[说明](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/README.md#dataset)）。每行一次请求，字段：`input_length`、`output_length`、`hash_ids`。可用兼容 harness 重放（例如 `aiperf --custom-dataset-type mooncake_trace`）。`hash_ids` 编的是共享前缀块，适合测 prefix cache / KV 复用。

形状：长进短出，贴着长地平线 agent。中位输入约 **67K**，输出约 **400**。分布是**双峰**，不是清一色的巨长：

- 大约 **53%** 在 **64K+**（尾部到约 **1M**）
- 大约 **47%** 低于 64K；大约 **18%** 低于 8K
- 大约 **8%** 超过 128K；大约 **3–4%** 超过 256K

### 2.2 Benefits of Decode Context Parallelism

单机 **8×B200**，**Kimi K2.6 NVFP4**，vLLM。并发从 **16 扫到 512**。原文写「见下表」——活页表是 JS 控件，这里只留正文。整条吞吐–交互 Pareto 上，DCP 能撑住高得多的并发，每 GPU 吞吐也更高。

![figure 3](../../../../assets/vllm/blog/performance/dcp/04-figure-3.png)

差别在 KV 住哪儿。基线 TP 在每张 GPU 上复制 KV，峰值内存很快顶满：并发 **64** 时 KV **100%**，吞吐卡在大约 **1,863 tok/s/GPU**——再塞不进人。DCP 按序列切，每卡只存每条请求 KV 的 **1/N**。并发 **512** 仍只有约 **82%** KV，**6,091 tok/s/GPU**。

**核心价值：** 恰恰在「复制 KV 的 TP 先 OOM」的长上下文上，DCP 还能把并发撑上去。

### 2.3 Comparison by Sequence Length

![figure 4](../../../../assets/vllm/blog/performance/dcp/05-figure-4.png)

按完整序列长度（input + output）再画一条吞吐–交互 Pareto。请求分成五档：**&lt;32K**、**32–64K**、**64–128K**、**128–200K**、**200K+**。DCP 在 **200K+** 仍停在同一条高而稳的前沿上；短桶和长桶几乎重叠：吞吐随并发涨，单用户速度在长上下文上仍可用。复制 KV 的 TP 在这里已经没有房间。

## 3. Challenges of Serving Long Contexts

TP 下 KV 按 **attention head** 切。每个 KV head 有自己的 K/V 张量；head 是 TP 能交给一张卡的最小单位。标准 TP **切不开一颗 head 内部的 KV**。K 个 KV head，可以给每张卡不同的子集，直到每卡一颗。超过 K，两张卡就会握着**同一颗 head 的副本**，而不是独一无二的切片。

## 4. What is DCP?

不像纯 TP，DCP 按**序列（上下文）维**切 KV。每张 GPU 负责同一条序列上一段 **token 位置** 的 KV。例子：一条 **200K**，四张卡分别管 0–50K / 50K–100K / 100K–150K / 150K–200K。卡越多，每卡 KV 越瘦，batch 才能再涨。

![figure 5](../../../../assets/vllm/blog/performance/dcp/06-figure-5.png)

### 4.1 Decode Context Parallelism Process

标准节奏：**AllGather Q → Compute → AllGather + ReduceScatter**。

- **AllGather Q。** 每卡只有 Q 的一片，attention 却要对任意 key 用完整 query。在 DCP 组里 all-gather。Decode 时 Q 只有**一个 token**，这一步便宜。MLA 可选：[PR #45964](https://github.com/vllm-project/vllm/pull/45964) 在**加载时**于 DCP 组内复制那份很小的 query 投影，Decode **连这步 AllGather 也省**（`VLLM_DCP_Q_REPLICATE=1`）。
- **Compute。** 用 gather 来的 Q 对**本地** KV 切片做 attention。vLLM：MLA 走 `k_up`，GQA 走 `tensor_broadcast`。
- **AllGather + ReduceScatter（`cp_lse_ag_out_rs`）。** 共享部分输出和 LSE；LSE 再加权合并（online softmax）；ReduceScatter 求和，每卡只拿回自己的 head-slice。

## 5. vLLM Usage

DCP 多一个参数：`decode_context_parallel_size`，和现有 TP 放在一起。

### 5.1 Offline

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    tensor_parallel_size=2,
    decode_context_parallel_size=2,
)
outputs = llm.generate(prompts, sampling_params)
```

### 5.2 Online

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

### 5.3 MLA Backend

**Models：** DeepSeek-V2 / V3 / R1，以及用 Multi-head Latent Attention 的 Kimi K2.6。

**Why it's different。** MLA 把 Key/Value 收成一份低秩 latent，所有 query head 共用——等于一个 KV head。纯 TP 没头可切，latent 在每个 TP rank 上整份复制。TP 帮不上忙，所以 MLA 是 DCP 的理想对象：整份 cache 都是多余的，整份都可以按序列切。

**What they do。** DCP 按序列切 latent KV；attention 时每 rank 对自己的 latent 切片做 **up-project**（`k_up`）还原 K/V。有效 KV head 数是 1，序列可以切到满 TP：

- `tensor_parallel_size >= decode_context_parallel_size`
- `tensor_parallel_size % decode_context_parallel_size == 0`

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 8
```

### 5.4 GQA Backend

**Example models：** Qwen3-235B，以及其他 Grouped-Query-Attention 模型（Llama 家族等）。

**Why it's different。** GQA 存 `num_key_value_heads` 个 KV head，TP 先按这些头切。干净只到 `num_key_value_heads`；`tensor_parallel_size` 超过它，就开始出现 `tp // num_key_value_heads` 份相同副本。

**What they do。** DCP 去填那些副本，换成**不同的序列块**；共享的 KV head 再广播到 query head（GQA 的 `tensor_broadcast`）。序列切分的上限就是这份复制因子：

- `(tensor_parallel_size // num_key_value_heads) >= decode_context_parallel_size`
- `(tensor_parallel_size // num_key_value_heads) % decode_context_parallel_size == 0`

```python
# Qwen3-235B has num_key_value_heads = 4; tp=8 gives 8//4 = 2 redundant copies,
# so dcp can be up to 2.
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2
```

## 6. Future Work

当时还在做：更细的 TP / DCP 粒度，把过度切分浪费的效率要回来；更好的 DCP **A2A** kernel（多机和单机），上下文变长、卡变多时少露通信、多和 compute 重叠；更好支持 **MTP / 投机解码**，不要把投机换来的延迟吃掉；把 **P/D** 分离上的 DCP 做硬；混合模型和 Dynamic Chunked Pipeline Parallelism；更多后端。

社区在接 **GLM-5.2**、**Kimi K3**。更长的路线：**Prefill Context Parallelism（PCP）**。Kimi K3 的 DCP 成绩当时还在做。部署与历史说明见 [vLLM Decode Context Parallel docs](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/#decode-context-parallel)。

## 7. Conclusion

DCP 是在重想长上下文推理时 GPU 怎么组织。不要逼 GPU 复制 KV，也不要让它们闲着：attention 时按序列切，随后让同一批 GPU 再摊 FFN 权重加载。系统该随上下文变长而扩，而不是被它压垮。

vLLM 里已经是原生支持。它和行业里同一方向站在一起——NVIDIA 在 TensorRT-LLM 里做的 [Helix Parallelism](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog22_Helix_Parallelism_Scaling_Multi_Million_Token_Decoding_with_KV_Cache_Sharding.md)。Kimi K3 的 DCP 成绩当时还在做，成熟后再发。

## About Us

感谢 NVIDIA 团队审稿、bench 和工程意见：Anahita Bhiwandiwalla, Xin Li, Pavani Majety, Nidhi Bhatia, Roman Ageev, Pen Chung Li, Chris Hoge。初版 DCP 由 [Moonshot AI](https://www.moonshot.cn/) 上游到 [vLLM #23734](https://github.com/vllm-project/vllm/pull/23734)；[Lucas Wilkinson](https://github.com/LucasWilkinson) 做了后续加固和扩展。更广的 vLLM 社区让这次基准测得下去。

数字测在 **NVIDIA B200**、Kimi K2.6 **NVFP4**；用支持 `--decode-context-parallel-size` 的现行 vLLM 复现。Kimi K3 的 DCP 成绩当时还在做。

长上下文的地图：TP 切头、DCP 切序列、Mooncake 把前缀放到池子里、P/D 把阅读和说话拆开。四件事不是互斥的。
