---
source: https://vllm.ai/blog/2026-07-06-vllm-hpc-ops
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# HPC-Ops：H20 上混长 Decode 的 attention，和小 expert GEMM 的 MoE

英文对照：[en/vllm/blog/performance/hpc-ops.md](../../../../en/vllm/blog/performance/hpc-ops.md)  
原文：https://vllm.ai/blog/2026-07-06-vllm-hpc-ops  
2026-07-06。署名 **Tencent Hunyuan AI Infra Team and vLLM Team**。学习笔记；页上 H20 数字不是你的 SLA。Attention [PR #46020](https://github.com/vllm-project/vllm/pull/46020)，MoE [PR #45924](https://github.com/vllm-project/vllm/pull/45924)。Hopper，尤其 **H20**。邻近的 attention backend：[triton-attn.md](../architecture/triton-attn.md)。硬件入口：[hardware-plugin.md](../architecture/hardware-plugin.md)。

[HPC-Ops](https://github.com/Tencent/hpc-ops) 的 Attention 和 MoE kernel 进 vLLM `main` 当一等 backend。不改引擎源码，也不养长期 fork。Attention 当时只认 Hy3 系；MoE 只认 FP8。不是通用默认，是 Hunyuan 产线 kernel 走 backend 接口进 main。

**页上的 TL;DR：**

- Attention：每步负载均衡的 Decode 调度 + 熔在一起的 RoPE / QK-Norm / KV-write prologue。混长 Decode 相对静态 split-KV 最高约 **2.95×**，相对 FlashInfer / FlashAttention 平均约 **2.25×**。
- MoE：整条 fused 低延迟 FP8 管线。相对 Triton / CUTLASS：TP8 / EP1 平均 **1.59×**，TP1 / EP8 **1.21×**，输出质量对齐。
- 8×H20 上 Hy3 两端一起：TTFT 约 **−24%**，TPOT 约 **−17%**，对 vLLM 默认 backend。

```bash
--attention-backend HPC_ATTN
--moe-backend hpc
```

本地图（原文版权仍归原站；学习对照用）：

![dynamic partitioning](../../../../assets/vllm/blog/performance/hpc-ops/01-dynamic-partitioning.png)

![fused moe latency](../../../../assets/vllm/blog/performance/hpc-ops/02-fused-moe-latency.png)

![decode dynamic vs static](../../../../assets/vllm/blog/performance/hpc-ops/03-decode-dynamic-vs-static.png)

## 为什么这事要紧

生产 serving 不再是当初 kernel 调参时那种齐整、单轮 batch。真实流量混长；模型越来越多是长上下文 MoE；agentic 两边一起压。延迟很大一块，看 kernel 怎么给 GPU 排活、阶段之间怎么搬数据，不是 matmul 峰值单独能解释。

固定 split-KV 的 Decode：batch 里长短混在一起，总时间被最重的 CTA 钉死，短请求那几只 CTA 先做完就闲着。MoE decode：专家 GEMM 小，周围 gather / launch / HBM 往返更贵——常规路径先把 token 收进 per-expert buffer，每阶段付一次 launch，中间结果在 HBM 里来回跳。

vLLM 已经是快而灵活的 serving 引擎；剩下的延迟和吞吐，看 attention 和 MoE kernel 能不能吞下这种脏流量。HPC-Ops 对着这块来——腾讯大规模生产里炼过的算子库，伺候 Hy3 的同一套 kernel，现在当一等 Attention / MoE backend 进了上游。

## 先说一句 Hy3

Hy3 是腾讯混元给 agentic 执行、编码、长程推理用的 MoE。**295B** 里激活 **21B**。原文还写过：同档里 agent 能力能跟大 **2–3×** 的开源旗舰较劲，并压幻觉——那是模型卡话，这篇不评。底下：**192** expert，top-8 路由，GQA（64 head、8 KV head、head dim 128），**256K** 上下文，**3.8B** MTP 层。发 BF16 和 FP8（Hy3-FP8）。这篇写的是伺候它的 kernel，不是模型卡。

## HPC-Ops 进 vLLM

[HPC-Ops](https://github.com/Tencent/hpc-ops) 是混元 AI Infra 的开源算子库：attention、MoE、GEMM、sampling、normalization、通信计算融合；原生 BF16 和 FP8；Python API 打算直接丢进框架。对着 Hopper 拧，尤其 H20。这次上游两只 backend：

| vLLM backend | What it optimizes | Precision | Merged in |
| --- | --- | --- | --- |
| Attention | 负载均衡 Decode + fused RoPE/QK-Norm prologue | BF16 / FP8 | [PR #46020](https://github.com/vllm-project/vllm/pull/46020) |
| Fused MoE | 整条 fused 低延迟 MoE 管线 | FP8 | [PR #45924](https://github.com/vllm-project/vllm/pull/45924) |

后文只盯这两只。

## Attention backend：动态负载均衡调度

### 难处：每个 batch 都是混长 Decode

Decode 每步要对整份 KV 做 attention。16K 上下文大概是刚起步 1K 的 **16×**。连续 batch 把不同生成阶段的请求塞进同一次 launch，所以一只 batch 里长短经常搅在一起。

现有 Decode kernel 用固定 grid 把活映射到 CTA：KV head × request × split-KV chunk。Split 度数必须齐：

- 钉死 split 数 → 最长序列当家；短请求的 CTA 先做完就闲。
- 钉死 chunk 大小 → split 数要按任何人需要的最大来；短请求会 launch 空 chunk，占调度槽。

墙钟时间是最重那只 CTA。

### 办法：每步一只负载均衡的 Decode 调度器

三阶段，扁平 persistent 设计，跟着这步 batch 的真实长度分布走，而不是 launch 时写死的 split 政策。

- **Assign。** 轻量 kernel 把每条 KV 切成齐整的 **64-token** tile。总 tile 数 / 可用 CTA 数 = 每 CTA 的 bucket 大小。Tile 按 head-major、batch-minor 填桶；满了就溢到下一只 CTA。长序列按长度比例拆开；短的只贡献几块 tile，独占不了一只 CTA。每 CTA 还有最低工作量地板：总活太少时防止切过细，combine 的税会吃掉调度收益。任务图 **每个 Decode 步算一次**，这一步里每一层 transformer 复用，摊到接近零。
- **Compute and combine。** Persistent grid：每只 CTA 循环自己的 task bin，写出 partial 输出 + log-sum-exp 到 split buffer，碰到 terminator 才停。任务之间不重新 launch。轻量 combine kernel 按 (head, request) 把 per-chunk partial 收成 BF16。

CTA 差不多一起收工；静态长尾 stall 没了。

**Figure 1。** Dynamic partitioning：齐整 tiling，再均衡分桶。

### 熔在一起的 attention prologue

QK-Norm、RoPE、KV-cache 写入——FP8 再加 query quant——本来是分开的、绑内存的 launch。`HpcRopeNorm` 从 fused QKV projection 起把它们焊在一起，按模型要求的顺序（Hy3 是 **先** 做 normalization **再** RoPE），把 K/V 写进 paged cache；FP8 路径再吐出 per-token、per-head 的 FP8 query 和 scale，attention 不必再量化一遍。Prefill 和 Decode 都走这条。一层 prologue 少几次 HBM 往返。

### 跟 vLLM 怎么接

`HpcAttentionBackend` 继承 `AttentionBackend`，走标准注册，跟 FlashAttention、FlashInfer 并列。

## MoE backend：fused 低延迟 FP8 管线

### 难处：小专家 GEMM，以及它周围的税

高吞吐、大 batch 的 MoE 是算力绑，现有 kernel 一般够用。低延迟 Decode 反过来：每个专家只拿到一把 token，GEMM 小、绑内存，tile 数每步还在变，很难在 GPU 上摊匀。

GEMM 周围更添乱。常规路径：路由、gather 进 per-expert HBM buffer、Gate-Up GEMM、activation + quant、Down GEMM、top-k 加权收——gather 先在 HBM 物化一份，每阶段自己付 launch 和中间结果往返。Decode 里 GEMM 已经小，这些税跟 GEMM 叠在一起。

### 办法

路由和 index 预处理、Gate-Up GEMM、activation + quant、Down GEMM、top-k 加权收，收成一条紧凑路径：

- **Routing and index build。** Shared-memory counting 给 token 划出连续的 per-expert 区间（少全局 atomic 压力），造出 GEMM 直接吃的 routing index / per-tile 任务图。
- **Gate-Up GEMM。** 经 routing index 读原始 token——没有单独的 gather。Activation + FP8 quant 是另一只 fused kernel，Down GEMM 直接读它的输出。
- **Occupancy-first，不做 warp specialization。** 一只 warp group 既搬数据又算；延迟隐藏从 CTA 内软件流水改成跨 CTA 的硬件调度。Persistent grid 把 SM 填满，把不均的 per-expert tile 摊开。
- **PDL-chained stages。** Programmatic Dependent Launch 让每次 launch 叠在上一阶段尾巴上，气泡抹到最后的 top-k reduce（共享专家输出也可以折进来）。

专家走 FP8，per-tensor 和 block-wise scaling 都有；输出质量跟基线对齐。

### 跟 vLLM 怎么接

`HPCExperts` 继承 `FusedMoEExpertsModular`，跟 DeepGEMM、Triton 并列注册。

## 怎么开 HPC-Ops backend

先从源码装：

```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops
make wheel
python3 -m pip install dist/*.whl
```

Attention 当时 **只认 Hy3 系**：

```bash
vllm serve tencent/Hy3 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN
```

Hy3-FP8 还要多几面旗：

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --attention-backend HPC_ATTN \
    --kv-cache-dtype fp8_e4m3 \
    --block-size 64
```

**页上的 tip：** 自定义模型要在 `forward` 里把 `rope_norm` 换成 `HpcRopeNorm`。见 PR #46020。

MoE **只认 FP8 模型**：

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 8 \
    --moe-backend hpc
```

硬件：只认 NVIDIA Hopper；最好在 H20。

## H20 上的数字

### Fused MoE vs Triton / CUTLASS

Hy3 配置。按 batch 平均：TP8 / EP1 相对最好基线 **1.59×**，TP1 / EP8 **1.21×**。最大收益落在主导低延迟 Decode 的中小 batch。

**Table 1。** FusedMoE 延迟（µs），TP8 / EP1（专家权重切在 8 个 rank 上）。

| Batch | HPC-Ops | Triton | CUTLASS |
| ---: | ---: | ---: | ---: |
| 4 | 42.0 | 56.4 | 74.5 |
| 16 | 85.7 | 124.2 | 209.2 |
| 32 | 124.0 | 184.3 | 275.6 |
| 64 | 147.2 | 374.9 | 330.3 |
| 128 | 161.5 | 302.9 | 345.3 |
| 256 | 170.1 | 310.9 | 351.6 |
| 512 | 194.5 | 331.6 | 369.2 |
| 1024 | 281.4 | 652.7 | 438.3 |
| 2048 | 491.8 | 731.5 | 794.4 |
| 4096 | 872.0 | 1366.0 | 1230.7 |
| 8192 | 1695.0 | 2216.8 | 2362.9 |
| 16384 | 3241.9 | 4329.1 | 4364.4 |

**Table 2。** FusedMoE 延迟（µs），TP1 / EP8。

| Batch | HPC-Ops | Triton | CUTLASS |
| ---: | ---: | ---: | ---: |
| 4 | 118.6 | 147.4 | 140.4 |
| 8 | 136.7 | 192.8 | 170.7 |
| 16 | 149.8 | 198.4 | 263.5 |
| 32 | 153.6 | 214.6 | 264.4 |
| 64 | 166.5 | 358.1 | 266.8 |
| 128 | 213.5 | 251.7 | 272.6 |
| 256 | 386.2 | 454.9 | 493.5 |
| 512 | 705.5 | 691.7 | 741.7 |
| 1024 | 1342.6 | 1369.1 | 1359.1 |
| 2048 | 2513.9 | 2668.7 | 2530.4 |

读表时别只看平均。TP1 / EP8、batch **512** 那一行 Triton **691.7 µs** 比 HPC-Ops **705.5 µs** 还快一点；大 batch 两边也接近。页上说最大增益在中小 batch，这张表对得上。

**Figure 2。** HPC-Ops FusedMoE on H20 — Hy3。

### 混长 batch 下的 Decode

FP8 Decode，从齐整扫到很偏（标签 A×B = A 条请求、KV 长度 B）。相对静态 split-KV 的优势随偏斜涨：小而齐的 batch 打平，到 1×128K + 31×4K 是 **2.95×**。相对 FlashInfer / FlashAttention 里更好的那只，平均 **2.25×**。

**Table 3。** Decode 延迟（ms），按 KV 长度分布。

| Decode scenario | HPC-Ops dynamic | HPC-Ops static | FlashInfer | FlashAttention | Dynamic vs static |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64×0.5K | 0.013 | 0.013 | 0.050 | 0.025 | 1.00× |
| 64×4K | 0.033 | 0.043 | 0.221 | 0.095 | 1.32× |
| 32×0.125K + 32×4K | 0.020 | 0.033 | 0.119 | 0.053 | 1.59× |
| 2×32K + 30×4K | 0.032 | 0.056 | 0.169 | 0.094 | 1.76× |
| 1×64K + 15×4K | 0.042 | 0.097 | 0.118 | 0.065 | 2.32× |
| 1×128K + 31×4K | 0.063 | 0.186 | 0.220 | 0.097 | 2.95× |

**Figure 3。** Decode attention on H20 — Hy3：动态对静态调度。

### Attention vs FlashAttention / Triton / FlashInfer

vLLM attention 基准，Prefill / Extend / Decode。页上的说法：几乎每个 case 都跟三家里最快的打平或更快。格子里有两处例外，记下来：`2q1ks4k` Extend 上 FlashInfer **1.829** / FlashAttention **1.830**，HPC-Ops **1.835**；`16q1s2k` Decode 上 FlashInfer **0.052**，HPC-Ops **0.054**。不是「每格都赢」。

**Table 4。** Attention 延迟（ms）。

| Batch spec | Type | Batch | HPC-Ops | FlashAttention | Triton | FlashInfer |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q512 | Prefill | 1 | 0.047 | 0.069 | 0.123 | 0.070 |
| q1ks2k | Extend | 1 | 0.406 | 0.431 | 1.132 | 0.431 |
| q2k | Prefill | 1 | 0.530 | 0.574 | 1.525 | 0.609 |
| q4k | Prefill | 1 | 2.002 | 2.093 | 5.816 | 2.144 |
| q8k | Prefill | 1 | 7.883 | 7.957 | 22.702 | 8.084 |
| 2q1ks4k | Extend | 2 | 1.835 | 1.830 | 5.046 | 1.829 |
| 8q1s1k | Decode | 8 | 0.019 | 0.031 | 0.035 | 0.021 |
| 16q1s2k | Decode | 16 | 0.054 | 0.098 | 0.106 | 0.052 |
| 32q1s1k | Decode | 32 | 0.057 | 0.102 | 0.080 | 0.058 |
| 64q1s4k | Decode | 64 | 0.299 | 0.620 | 0.510 | 0.340 |

### 端到端：Hy3 on 8× H20

两只 backend 一起对 vLLM 默认。TTFT 平均大约低 **24%**；TPOT 大约 **17%**，最大 batch 涨到大约 **30%**。TTFT 表关掉了 Chunked Prefill 和 Prefix Caching。

**Table 5。** TPOT（ms），output length = 4K。

| Batch | Baseline | HPC | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 8.00 | 7.76 | +3.0% |
| 4 | 11.14 | 10.67 | +4.2% |
| 8 | 13.49 | 11.31 | +16.2% |
| 16 | 17.98 | 13.56 | +24.6% |
| 32 | 24.13 | 18.32 | +24.1% |
| 64 | 31.10 | 21.90 | +29.6% |

**Table 6。** TTFT（ms），input length = 8k。

| Batch | Baseline | HPC | Improvement |
| ---: | ---: | ---: | ---: |
| 1 | 565.69 | 431.00 | +23.8% |
| 4 | 1920.15 | 1471.43 | +23.4% |
| 8 | 3948.22 | 3035.44 | +23.1% |
| 16 | 7807.18 | 5885.63 | +24.6% |

**Table 7。** TTFT（ms），batch size = 16，扫输入长度。

| Input length | Baseline | HPC | Improvement |
| --- | ---: | ---: | ---: |
| 2k | 1792.62 | 1363.13 | +24.0% |
| 4k | 3704.27 | 2886.40 | +22.1% |
| 8k | 7807.12 | 5893.93 | +24.5% |

Table 6 的 batch 16 / 8k 是 **7807.18 / 5885.63**；Table 7 同一形状是 **7807.12 / 5893.93**。页上两套格子，不要合成一条。

## 接下来 / 致谢

更长的合作；成熟了再往上游送。页上点名：

- **Tencent Hunyuan AI Infra**（kernel 和 backend）：Sethran Liu, Chase Shao, Shengy Wei, Theo Cheng, Ryann Xue, Lando Jiang, Looper Zhao, Haank Lin, Aiden Ren, Lehua Ding, Chengv Jiang, Steven Kuang, Liqi He, Kipper Gong, Reedlau Liu, Raccoon Liu, Dick Zhu。
- **Tencent Network Platform Department**（通信优化）：Xuan Zhang, Haoran Zhao, Yuanyuan Gong, Yadong Liu, Jinzhu Wang, Yinben Xia, Xiang Li, Quan Wen, Zekun He。
- **vLLM/Inferact**（开放 backend 接口、评审、设计）：Kaichao You, Yongye Zhu, Yifan Qiao。
- **NVIDIA**（kernel 和性能）：Yuanhang Sun, Perkz Zheng, Yuxi Chi, Jiang Shao, Jun Gu, Meng Wang, River Liu, Gary Ji, Chandler Zhou。

基线对着 CUTLASS/CuTe、TensorRT-LLM、FlashInfer、FlashAttention、Triton 测。
