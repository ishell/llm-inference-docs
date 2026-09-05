---
source: https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Qwen3.5 25K TPS/GPU：GDN + 异构 cache 搬完才到 Pareto 左端

英文对照：[en/vllm/blog/serving/qwen35-25k-tps.md](../../../../en/vllm/blog/serving/qwen35-25k-tps.md)  
原文：https://vllm.ai/blog/2026-08-06-qwen35-25k-tps  
2026-08-06。署名 **vLLM Team**。GB200 NVL72。模型：[`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)。ISL/OSL=**8192/1024**。hybrid-attention 祖宗：[qwen3-next.md](qwen3-next.md)。后来 Max 级 day-0：[qwen38.md](qwen38.md)。异构 cache / 双描述符：[hybrid-ssm.md](hybrid-ssm.md)。Decode 侧亲戚：[../performance/dcp.md](../performance/dcp.md)。NIXL P/D 路线图：[issue #33702](https://github.com/vllm-project/vllm/issues/33702)。**系统 TPS/GPU ≠ 单用户 TPS。** 他们扫的是 Pareto **左端**（总吞吐），并发 **64–5120**，没测 1–32。

decode 固定 **1×DEP8**；prefill **4–8×DEP2**。峰值 **25,000 tok/s/GPU**。GSM8K 五套拓扑都是 **88%**，和聚合跑对齐。复现：Docker `vllm/vllm-openai:nightly-d223c90`、Dynamo `1.2.0.dev20260526`、srt-slurm `v1.0.32`。菜谱：[srt-slurm-recipes …/Qwen3.5/GB200/8k1k/vllm/disagg](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg)。

本地图（原文版权仍归原站；学习对照用）：

![pareto curves by prefill endpoints](../../../../assets/vllm/blog/serving/qwen35-25k-tps/01-pareto-curves-by-prefill-endpoints.png)

![pareto frontier qwen35 nvfp4](../../../../assets/vllm/blog/serving/qwen35-25k-tps/02-pareto-frontier-qwen35-nvfp4.png)

## Introduction

Qwen3.5（2026 年初）是 hybrid：满 attention 层夹 **Gated Delta Network (GDN)**。拆开 serving 就多两件活：Blackwell 上的 GDN kernel，以及 Prefill/Decode 之间搬 **异构** 的 attention/GDN 状态。社区把这条路养熟了。这篇是 **你** 在 GB200 NVL72 上怎么摸过 **25K total TPS/GPU**——贡献、数字、菜谱。不是 1–32 用户的延迟扫描。

## Challenges and key optimizations

SSM 的 P/D 是顺着 [NIXL disaggregation roadmap](https://github.com/vllm-project/vllm/issues/33702) 推的。布局 / 逻辑块对物理块 / TP 状态搬运：见 [hybrid-ssm.md](hybrid-ssm.md)。对 Qwen3.5 真正要紧的是三刀。

### 1. Blackwell 上的 GDN Prefill

[FlashInfer PR #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001)——Blackwell GDN prefill kernel。相对先前 FLA/Triton：大约 **1.02×–5.78×**，覆盖 Qwen3.5 各尺寸、TP、序列长、batch 形状。

接到 vLLM Prefill：[vLLM PR #40717](https://github.com/vllm-project/vllm/pull/40717)。**8×B200**、Qwen3.5-397B-A17B-NVFP4：

- 微基准 GDN kernel 最高 **5.92×**
- Prefill-only 端到端吞吐 **1.13×**（ISL/OSL = **8192/1**）
- 同一 8K/1 Prefill-only 负载，mean TTFT **−12%**

支持的 Blackwell 上，`auto` 会选 FlashInfer。显式：

```text
--gdn-prefill-backend flashinfer
```

### 2. Hybrid cache 和 GDN 状态搬运

hybrid SSM-attention 的 P/D 坐在 [[Core][KVConnector] Support HMA+NixlConnector #35758](https://github.com/vllm-project/vllm/pull/35758) 和 [hybrid-ssm.md](hybrid-ssm.md) 那叠 connector 上。必要前置：把 HMA 逻辑块映射到对的物理区，NIXL 只搬属于该层类型的 cache。描述符 **4284 → 1650**；小规模同节点 H100 上吞吐大约 **+7%**。**单有 HMA 不够**伺候 Mamba 风格状态（布局、大小、搬运语义都不一样）。

主 PR：[[PD][Nixl] Add support for hybrid SSM-FA models #36687](https://github.com/vllm-project/vllm/pull/36687)——双描述符视图 + 同构 TP，Prefill 和 Decode 能经 NIXL 同时搬满 attention KV **和** Mamba 风格 SSM 状态。同一叠：

- [[Kernel] Mamba support different layout for Conv state #37416](https://github.com/vllm-project/vllm/pull/37416)
- [[NIXL][Mamba][3/N] Heterogeneous TP: 3-read conv state transfer #37635](https://github.com/vllm-project/vllm/pull/37635)
- [[SSM/Mamba] Follow-up: N-1 prefill for P/D disaggregation #37310](https://github.com/vllm-project/vllm/pull/37310)

Qwen3.5 专有的 GDN：[PD disagg with NIXL Connector: GDN support (Qwen3.5) #41869](https://github.com/vllm-project/vllm/pull/41869)。

### 3. 无竞态的 async scheduling

两补丁。不修的话 `--async-scheduling` 不能用——**精度掉到零**。跨过 25K tok/s/GPU 靠它，所以两处竞态必须先落地。

- [[KV Connector] Fix PD async scheduling race condition for hybrid attn models #48481](https://github.com/vllm-project/vllm/pull/48481)
- [[Bugfix] Defer block freeing until in-flight steps finish under async scheduling + PD KV consumer #45357](https://github.com/vllm-project/vllm/pull/45357)

## Performance

### Environment

GB200 集群，NVLink72。ISL/OSL = **8192/1024**。模型 [`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)。decode 拓扑 **钉死**：一个 endpoint，**DEP8**（8 卡 DP+EP）。prefill：**4 到 8** 个 endpoint，每个 **DEP2**。

复现：当时的 `vllm/vllm-openai:nightly-d223c90`（[Hub layer](https://hub.docker.com/layers/vllm/vllm-openai/nightly-d223c900d85224c02f2162ee2c757a769e99f519/images/sha256-987393f42c48b8a649961a3484d95d400db184b64e4e1bb7f77cb91536d0f05e)）、[Dynamo](https://github.com/ai-dynamo/dynamo) `1.2.0.dev20260526`、[srt-slurm](https://github.com/NVIDIA/srt-slurm) `v1.0.32`。菜谱在 [srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes)。

### Accuracy

每套 serving 拓扑都跑 GSM8K，走 srt-slurm：

```yaml
benchmark:
  type: "gsm8k"
```

**五**套都是 **88%**，和聚合跑的 Qwen3.5 对齐。如果不是约 88，P/D 路径就错了——别读 Pareto。

### Recipe settings（测量）

固定 ISL/OSL，随机集，`random_range_ratio=0.8`。flag 见下。

### Results

**Figure 1.** 不同 Prefill 实例数下，拆开 serving 的 Pareto。

**Figure 2.** 合成后的 Pareto 前沿，Qwen3.5 NVFP4。

总 TPS/GPU 摸到 **25,000** tok/s。并发 **64…5120**。**没测** 1–32：目标是 Pareto **左端**（总 TPS/GPU 最大），不是每用户 Gen TPS。超过 5120，他们钉死的那一个 8×GB200 Decode endpoint 的 KV 不够。更高并发可以，但要加 Decode GPU。

## Recipes and best practices

启动：

```shell
srtctl run --file <recipe>.yaml
```

命名：`NxDEP2-1xDEP8`——N 个 DEP2 Prefill 对一个 DEP8 Decode。五套底：**4×DEP2 … 8×DEP2**。每套三个衍生：底文件用 sa-bench 扫 cc **64…3072**；`-acc` 同一拓扑跑 GSM8K **五**次；`-cc4096` / `-cc5120` 是单个高并发点，Decode `--max-cudagraph-capture-size` 分别抬到 **640** 和 **768**。

值得单独说的（其余大多是共享样板）：

- `VLLM_SSM_CONV_STATE_LAYOUT=DS`——SSM 模型做 P/D **强制**；不设 conv-state 搬不动。菜谱还传了 `--no-disable-hybrid-kv-cache-manager`；后来几版 HMA 已是默认，这 flag 不再需要。
- `--async-scheduling`——摸到 25K tok/s/GPU 的关键之一。要已经带上那两处竞态修复的构建。
- `--mamba-ssm-cache-dtype bfloat16`——抬 Decode 有效 KV 容量。
- `--language-model-only`——Qwen3.5 是多模；纯文本关掉多模态，**并且**打通 fused QK-norm + RoPE + gate。
- Prefill `--max-num-batched-tokens 16384` = **2×ISL**。Prefill endpoint 少时（{4,5,6}×DEP2）Prefill 把 Decode 饿着；一步 batch 两条满 prompt，高并发大约 **+8%** 总 TPS/GPU。
- Decode `--max-cudagraph-capture-size`——最高两个点用 `cc/8 + 128`（cc=4096 时 640，cc=5120 时 768）；8 = Decode 的 DP rank 数。默认上限 **512** 到 cc=3072 够用。他们 **不确定** Pareto 数字是否真需要这一条；防着设。
- Prefix caching **关**——随机集上买不到东西。
- `--stream-interval 100`——高并发砍前端；**按 100 token 一块缓冲**，会动 ITL/TPOT。优化每 token 延迟而不是总 TPS 时别用。

实操：

调查某一配置时 `--api-server-count 1`。DP endpoint 上 vLLM 默认 API server 数等于 DP size；多于一个就会 **关掉默认 stats**，免得打出残缺数字。强制 1，每 10 秒日志回来（`VLLM_LOG_STATS_INTERVAL`）：prompt/generation 吞吐和 KV 利用率。他们说没有这些几乎找不到各配置瓶颈。

三个环境变量：`DYN_LOG=error`、`DYN_SDK_DISABLE_ANSI_LOGGING=1`、`VLLM_LOGGING_COLOR=0`。第一个砍 Dynamo 噪音；后两个去掉一部分（不是全部）ANSI。否则日志人读不了。

## What's next

这篇挤的是 Pareto **左端**（总 TPS/GPU）。下一步要扫最大化 **每用户 Gen TPS** 的 PD 配置。那一档会离开 DEP，转向 **TEP**（TP+EP）或纯 TP。加 GPU 是另一根杠杆。后来 2.4T 复用这套骨架：[qwen38.md](qwen38.md)。更早的 hybrid GDN/满 attention 交错：[qwen3-next.md](qwen3-next.md)。

## Acknowledgements

Artem Perevedentsev (NVIDIA)、Vadim Gimpelson (NVIDIA)、Jiangyun Zhu (Inferact)、Nicolò Lucchesi (Mistral)、Zhanqiu Hu (Red Hat)、Nick Hill (Inferact)、Linxuan Li (Alibaba)、JingZe Cui (NVIDIA)、Cyrus Chang (NVIDIA)、Xin Li (NVIDIA)。
