---
source: https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Qwen3.5 25K TPS/GPU：GDN + 异构 cache 搬完才到 Pareto 左端

英文对照：[en/vllm/blog/serving/qwen35-25k-tps.md](../../../../en/vllm/blog/serving/qwen35-25k-tps.md)  
原文：https://vllm.ai/blog/2026-08-06-qwen35-25k-tps  
2026-08-06。署名 **vLLM Team**。GB200 NVL72。模型：[Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)。ISL/OSL = **8192/1024**。更早的 hybrid：[qwen3-next.md](qwen3-next.md)。异构 cache 搬运：[hybrid-ssm.md](hybrid-ssm.md)。后来的 3.8：[qwen38.md](qwen38.md)。Decode 侧长上下文：[dcp.md](../performance/dcp.md)。**系统 TPS/GPU ≠ 单用户 TPS。** 扫的是 Pareto **左端**（总吞吐），并发 **64–5120**，没测 1–32。

Qwen3.5 的 hybrid attention（满 attention + Gated Delta Network）让分离 serving 更难，也更有东西可拧。社区把 P/D 路径养熟了。这篇：主要贡献、GB200 NVL72 数字、配方——让 **你** 也能摸到 **25K total TPS/GPU**。

本地图（原文版权仍归原站；学习对照用）：

![pareto curves by prefill endpoints](../../../../assets/vllm/blog/serving/qwen35-25k-tps/01-pareto-curves-by-prefill-endpoints.png)

![pareto frontier qwen35 nvfp4](../../../../assets/vllm/blog/serving/qwen35-25k-tps/02-pareto-frontier-qwen35-nvfp4.png)

## 难点和关键优化

两件事：在 Blackwell 上加速 GDN，以及在 Prefill / Decode worker 之间把异构的 attention/GDN 状态搬对。

SSM 的 P/D：[NIXL disaggregation roadmap](https://github.com/vllm-project/vllm/issues/33702)。布局、逻辑/物理映射、TP 状态搬运：[hybrid SSM 分离](hybrid-ssm.md)。

### 1. Blackwell 优化过的 GDN Prefill

[FlashInfer #3001](https://github.com/flashinfer-ai/flashinfer/pull/3001)。相对 FLA/Triton：约 **1.02×–5.78×**，扫过 Qwen3.5 尺寸、TP、序列长度、batch 形状。

Prefill 侧接入：[vLLM PR #40717](https://github.com/vllm-project/vllm/pull/40717)。**8×B200**，Qwen3.5-397B-A17B-NVFP4：

- 测过的微基准里 GDN kernel 最高 **5.92×**
- Prefill-only（ISL/OSL = 8192/1）端到端 Prefill 吞吐 **1.13×**
- 同一套 Prefill-only 8K/1，mean TTFT **−12%**

支持的 Blackwell 上，GDN backend 设 `auto` 会自己选 FlashInfer。显式：

```
--gdn-prefill-backend flashinfer
```

### 2. Hybrid cache 和 GDN 状态搬运

底座是 [[Core][KVConnector] Support HMA+NixlConnector #35758](https://github.com/vllm-project/vllm/pull/35758) 和 [hybrid-ssm.md](hybrid-ssm.md) 里那一摞。HMA 逻辑块映到对的物理区，NIXL 只搬属于该层类型的 cache：描述符 **4,284 → 1,650**，小规模单机 H100 吞吐大约 **+7%**。Mamba 风格的状态在布局、大小、搬运语义上仍差得够多，光有 HMA 不够做对、做快 P/D。

主 PR：[[PD][Nixl] Add support for hybrid SSM-FA models #36687](https://github.com/vllm-project/vllm/pull/36687) — 双 descriptor 视图、同构 TP，Prefill 和 Decode 经 NIXL 同时搬满 attention KV 和 Mamba 风格 SSM。后续：[#37416](https://github.com/vllm-project/vllm/pull/37416)（conv-state 布局）、[#37635](https://github.com/vllm-project/vllm/pull/37635)（异构 TP，3-read conv state）、[#37310](https://github.com/vllm-project/vllm/pull/37310)（P/D 的 N-1 Prefill）。

Qwen3.5 专有：[PD disagg with NIXL Connector: GDN support (Qwen3.5) #41869](https://github.com/vllm-project/vllm/pull/41869)。

### 3. 没有竞态的 Async Scheduling

两处补丁。没有它们，`--async-scheduling` 会把 **准确率打到零**。跨过 25K tok/s/GPU，async scheduling 是钥匙之一。

- [[KV Connector] Fix PD async scheduling race for hybrid attn models #48481](https://github.com/vllm-project/vllm/pull/48481)
- [[Bugfix] Defer block freeing until in-flight steps finish under async scheduling + PD KV consumer #45357](https://github.com/vllm-project/vllm/pull/45357)

## 性能

### 1. 环境

GB200 集群，NVLink72。ISL/OSL = 8192/1024。Decode：**一个** endpoint，**DEP8**。Prefill：**4–8** 个 endpoint，每个 **DEP2**。

复现：当时的 vLLM 镜像 `vllm/vllm-openai:nightly-d223c90`（页上有 digest）、[Dynamo](https://github.com/ai-dynamo/dynamo) `1.2.0.dev20260526`、[srt-slurm](https://github.com/NVIDIA/srt-slurm) `v1.0.32`。配方：[srt-slurm-recipes](https://github.com/NVIDIA/srt-slurm-recipes)。

### 准确率

五套拓扑的 GSM8K 都是 **88%**，和聚合跑的 Qwen3.5 对齐。

```yaml
benchmark:
  type: "gsm8k"
```

### 2. 配方旋钮怎么选

固定 ISL/OSL，随机集，`random_range_ratio=0.8`。真正要紧的设置见下一节。

### 3. 成绩

**Figure 1。** 按 Prefill 实例数画出的 Pareto。

**Figure 2。** 合在一起的 Pareto 前沿。

每 GPU 总 TPS 摸到 **25,000** tok/s。并发 **64 → 5120**。**没测** 1–32：目标是左端 Pareto——把总 TPS/GPU 拧满。停在 5120，因为 Decode KV 容量在单台 8×GB200 endpoint 上见底了。并发还能往上，但要给 Decode 加 GPU。

## 配方和做法

[srt-slurm-recipes … Qwen3.5/GB200/8k1k/vllm/disagg](https://github.com/NVIDIA/srt-slurm-recipes/tree/main/recipes/multi-node/Qwen3.5/GB200/8k1k/vllm/disagg)。一条命令：

```shell
srtctl run --file <recipe>.yaml
```

命名：`NxDEP2-1xDEP8` — N 个 DEP2 Prefill 对着一个 DEP8 Decode。五套底，**4×DEP2** 到 **8×DEP2**。每套三个派生：底文件用 sa-bench 扫 cc 64…3072；`-acc` 同一拓扑跑五次 GSM8K；`-cc4096` / `-cc5120` 各抓一个高并发点，Decode 的 `max-cudagraph-capture-size` 抬到 **640** 和 **768**。

值得点名的旗：

- `VLLM_SSM_CONV_STATE_LAYOUT=DS` — SSM 模型做分离 serving **强制**；没有它 conv-state 搬不动。配方里还传了 `--no-disable-hybrid-kv-cache-manager`；HMA 后来已默认，那面旗不再需要。
- `--async-scheduling` — 25K tok/s per GPU 的钥匙之一。要带上面竞态修复的构建。
- `--mamba-ssm-cache-dtype bfloat16` — 抬 Decode 侧有效 KV 容量。
- `--language-model-only` — Qwen3.5 是多模态；纯文本负载关掉多模态，**同时** 打通 fused QK-norm + RoPE + gate。
- Prefill `--max-num-batched-tokens 16384` = **2× ISL**。Prefill endpoint 少时（{4, 5, 6}×DEP2）Prefill 饿着 Decode；每步 Prefill 塞两条完整 prompt，高并发大约 **+8%** 总 TPS/GPU。
- Decode `--max-cudagraph-capture-size` — 两个最高点用 `cc/8 + 128`（cc=4096 时 640，cc=5120 时 768）；8 = Decode 上的 DP rank。默认 capture 上限 512，cc=3072 够用。他们不敢肯定这对报出的 Pareto 是必须的；当预防针。
- 前缀缓存 **关**：随机集上买不到东西。
- `--stream-interval 100` — 高并发砍前端。流式输出按 100 token 一块缓冲，**会** 动到测到的逐 token 延迟。优化 ITL/TPOT 而不是总吞吐时，别开。

实践：

查某一套配置时 `--api-server-count 1`。DP endpoint 上 vLLM 默认 API server 数等于 DP size；多于一个就会 **关掉** 默认 stats，免得打出不完整的数。强制 1，日志回来：每 10 秒（`VLLM_LOG_STATS_INTERVAL`）prompt / generation 吞吐和 KV 利用率。没有这些，他们很难找到各配置的瓶颈。

另外：`DYN_LOG=error`、`DYN_SDK_DISABLE_ANSI_LOGGING=1`、`VLLM_LOGGING_COLOR=0`。第一条砍 Dynamo 日志；后两条压掉一部分（不是全部）ANSI。不然日志文件常常没法读。

## 下一步

到现在：左端 Pareto，总 TPS/GPU。下一步：把 **每用户 Gen TPS** 拧满的 PD 配置。那一区要从 DEP 挪向 **TEP** 或纯 **TP**，一般更照顾单用户。加 GPU 是另一根杠杆。

## 致谢

Artem Perevedentsev（NVIDIA）、Vadim Gimpelson（NVIDIA）、Jiangyun Zhu（Inferact）、Nicolò Lucchesi（Mistral）、Zhanqiu Hu（Red Hat）、Nick Hill（Inferact）、Linxuan Li（Alibaba）、JingZe Cui（NVIDIA）、Cyrus Chang（NVIDIA）、Xin Li（NVIDIA）。
