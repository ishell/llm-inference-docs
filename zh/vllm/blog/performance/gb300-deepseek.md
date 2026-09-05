---
source: https://vllm.ai/blog/2026-02-13-gb300-deepseek
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DeepSeek-V3.2 on GB300：验证部署，不是刷峰值

英文对照：[en/vllm/blog/performance/gb300-deepseek.md](../../../../en/vllm/blog/performance/gb300-deepseek.md)  
原文：https://vllm.ai/blog/2026-02-13-gb300-deepseek  
2026-02-13。署名 **The DaoCloud and vLLM team**。页上的 bench，不是你的 SLA。快照：**vLLM v0.14.1**，**CUDA 13.0**。GB300 / B300 **288 GB**。Day-0 稀疏注意力：[deepseek-v32](../architecture/deepseek-v32.md)。后来的压缩栈：[deepseek-v4](../architecture/deepseek-v4.md)。P/D 亲戚：[Mooncake](../serving/mooncake.md) / [large-scale](../serving/large-scale.md)。

适用：在 Blackwell Ultra 上选 NVFP4、TP2 / EP2、要不要 MTP、1P1D 什么时候该加 P。不适合：把 **7360 TGS** 当成承诺——原文自己写的是 **可复现基线**，不是调到顶。

v0.14.1 的 P/D 当时还要手工补 [PR #32698](https://github.com/vllm-project/vllm/pull/32698)；后来的 main 已经合进去。

## 概览

V3.2（NVFP4 + TP2）在 GB300（SM103，Blackwell Ultra）上跑通。靠 FP4，Prefill-only 单卡 **7360 TGS**；混上下文 ISL=2k / OSL=1k 输出 **2816 TGS**。相对 R1，V3.2 在 vLLM 里推理还有明显空间。

两张 GB300 上，R1（NVFP4 + EP2）Prefill-only **22476 TGS**（ISL=2k，OSL=1，batch=256）；混上下文 **3072 TGS**。相对 Hopper：Prefill 约 **8×**，混上下文约 **10–20×**。

## 三种测法

- **Prefill-only。** OSL = 1，墙钟几乎全是 Prefill。用来比架构和并行怎么吃长输入。
- **混上下文、短输出。** ISL=2k，OSL=64/128。
- **混上下文、中等输出。** 更像在线 serving；常用 ISL=2k，OSL=1k，Prefill 和 Decode 都占时间。

页上的例子：

```bash
vllm bench serve --model nvidia/DeepSeek-R1-0528-NVFP4 \
  --seed $RANDOM \
  --dataset-name random \
  --base-url http://${PROXY_NODE_IP}:8000 \
  --tokenizer /mnt/models/DeepSeek-V3.2 \
  --num-prompts 1000 \
  --max-concurrency $MAX_CONCURRENCY \
  --random-input-len $ISL \
  --random-output-len $OSL \
  --ignore-eos
```

图都用 `vllm bench serve`：Prefill 吞吐 = 总 token 吞吐（tok/s）；Decode 吞吐 = 输出 token 吞吐（tok/s）。

## FP4 配方

Blackwell 第五代 Tensor Core 原生 NVFP4。

1. Hugging Face 权重：[DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)、[DeepSeek-R1-0528-NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4)。
2. Blackwell 上 FP4 MoE 要显式开 FlashInfer：

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=1
```

3. 单卡 288 GB，两卡装得下 DeepSeek 系列 NVFP4：

```bash
vllm serve nvidia/DeepSeek-V3.2-NVFP4    -tp 2
# or
vllm serve nvidia/DeepSeek-R1-0528-NVFP4 -tp 2
```

4. Prefill 边界 batch：`--max-num-batched-tokens`，R1 **32768**，V3.2 **20480**。

## Blackwell 把哪一头抬起来

### FP8 vs FP4（V3.2）

NVFP4 用 **一半 GPU 数** 也能整体更好。可低精度单独不够，并行策略同样关键。

赢家是 NVFP4 + **TP2**。Prefill-only（ISL=2k，OSL=1，batch=64）：相对 FP8 **1.8×**，最高 **7360 TGS**。混上下文（ISL=2k，OSL=1k）：输出 **2816 TGS**（**8×**）。TP4 就客气了——Prefill 只 **14%**，混上下文 **2×**——所以 TP2 才是效率选择。

两件事：内存税降了，attention 算子简单了。NVFP4 松带宽（抬输出吞吐），attention 简化（压 Prefill 延迟）。

**为什么是 NVFP4 + TP2：** 量化把权重和 KV 变小，batch 才能涨；TP2 让每卡活还够大，Tensor Core 吃得下 FP4 的 FLOPs 和带宽。TP4 把每卡活摊薄，增益就抓不住。

![dsv32 fp4 vs fp8 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/01-dsv32-fp4-vs-fp8-throughput.png)

**Figure 1。** V3.2 FP4 vs FP8。走 FP8：换 FP8 权重，`VLLM_USE_FLASHINFER_MOE_FP8=1`，`-tp 4`（四卡）。

### Blackwell Ultra vs Hopper（R1）

同一套请求、同一份 vLLM，比单卡总吞吐：GB300（NVL72）、B300（HGX）、上一代 H200。

- Prefill-only（ISL=2k）：GB300 比 B300 高 **14%**，比 H200 **8×**。
- 短输出混上下文（ISL=2k，OSL=128）：GB300 比 B300 高 **12%**，比 H200 **20×**。

![dsr1 h200 b300 gb300 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/02-dsr1-h200-b300-gb300-throughput.png)

**Figure 2。** R1 在 H200 / B300 / GB300 上的单卡吞吐。

理由不只 FP4：B300 FLOPs 是 Hopper 的 **7.5×**（峰值约 **15 PFLOPs**）；SM 的 SFU 帮 Prefill 的 attention；**288 GB** 是 H200 的 **2×**，带宽几乎翻倍；Blackwell Ultra 的 NVFP4 FLOPs 让 MoE 比 Hopper FP8 快——Decode 那一跳也来自这里。页上点名 [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)。小规模 intra-node TP2，GB300 相对 B300 仍有一点边。

## 部署怎么拧

### EP2 vs TP2

R1 权重两张 B300 的 HBM 就装得下。往上扩：DP 叠在 TP2 上，还是叠在 EP2 上？EP2 的 CLI：`-dp=2 --enable-expert-parallel`。

**Prefill-only（ISL=2k，OSL=1）。** EP2（蓝）天花板 **22476 TGS**，吞吐和 TTFT 斜率都比 TP2（绿）好看。EP 那种「大包、低频」通信，高并发时更能吃 RDMA/NVLink。蓝线会抖：专家路由不均，每批专家负载和 all-to-all 量不一样。

![dsr1 ep2 tp2 throughput prefill only](../../../../assets/vllm/blog/performance/gb300-deepseek/03-dsr1-ep2-tp2-throughput-prefill-only.png)

![dsr1 ep2 tp2 ttft prefill only](../../../../assets/vllm/blog/performance/gb300-deepseek/04-dsr1-ep2-tp2-ttft-prefill-only.png)

**Figure 3–4。** Prefill-only 的 EP2 vs TP2 吞吐和 TTFT。

**短输出混上下文（ISL=2k，OSL=64）。** TP2 每步 Decode 都要跨卡通信 → TPOT 比 EP2 差 **50% 到 2×**。TP 也把 TTFT 改善约 **50%**（每步更快）。这一头抵掉 TPOT，输出 token 吞吐净赚 **5%–20%**。

![dsr1 ep2 tp2 pd throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/05-dsr1-ep2-tp2-pd-throughput.png)

![dsr1 ep2 tp2 pd ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/06-dsr1-ep2-tp2-pd-ttft.png)

![dsr1 ep2 tp2 pd tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/07-dsr1-ep2-tp2-pd-tpot.png)

**Figure 5–7。** 一体机混跑的吞吐 / TTFT / TPOT。

**原文结论：**

- 拆开 Prefill 的 R1 on GB300：EP 更适合作 prefiller（再加 DP 扩）。Prefill 天花板比 TP2 大约 **10–15%**；TTFT 随并发涨得更慢——排队和尾延迟更好控。
- P+D 一体：ISL 大、OSL 小时 Prefill 是瓶颈 → **TP2**，免得 attention 把 Decode 的 GPU 时间挤掉。输出重：EP2 的 TPOT 优势占上风。

### MTP 的好处——也不是银弹

MTP 抬 Decode，但不总是。内置 draft 一次猜 **1** 个 token：

```bash
--speculative-config.method mtp \
--speculative-config.num_speculative_tokens 1
```

上下文不太长时，开 MTP（蓝）在并发 **≤256** 里比不开（绿）更高（接受率可以 **>80%**）。高并发一开 MTP，吞吐会断崖。

混上下文 ISL=2k / OSL=64：Decode 占比极低。MTP 多出来的算、内存、调度摊不掉。低并发摊不掉税；高并发再去挤 Prefill batch。两端吞吐都 **不如关掉 MTP**。

![dsr1 mtp throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/08-dsr1-mtp-throughput.png)

![dsr1 mtp ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/09-dsr1-mtp-ttft.png)

![dsr1 mtp peak output throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/10-dsr1-mtp-peak-output-throughput.png)

![dsr1 mtp tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/11-dsr1-mtp-tpot.png)

**Figure 8–11。** MTP 开 / 关：吞吐、TTFT、峰值输出、TPOT。

## V3.2 还有一截路

同一套 GB300：R1 的 Prefill 大约是 V3.2 的 **3×**。

- R1 EP2 Prefill 峰值约 **22476 TGS**。
- V3.2 EP2 Prefill 峰值约 **7360 TGS**。
- 都用 TP2：R1 的 TTFT 比 V3.2 低约 **55%**。

混上下文 ISL=2k / OSL=1k：输出吞吐和 TPOT 差距 **不大**。

![dsr1 vs v32 throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/12-dsr1-vs-v32-throughput.png)

![dsr1 vs v32 ttft](../../../../assets/vllm/blog/performance/gb300-deepseek/13-dsr1-vs-v32-ttft.png)

**Figure 12–13。** R1 vs V3.2 的吞吐和 TTFT。

**为什么 Prefill 是 R1 赢。** V3.2 加了 Indexer / Sparse MLA（`Indexer` + `SparseAttnIndexer`），还有带独立 cache 的 `DeepseekV32IndexerBackend`。Prefill 多付一层量化 / 索引。Profile：一层 DSA 的 kernel 时间是 MLA 的 **2.7×**。Indexer 以外，NVFP4 MoE kernel 选法和 R1 一样——Prefill 差在 Indexer / Sparse Attention。FP8 KV 那条线：[fp8-kvcache](fp8-kvcache.md)。

DSA 是给超长上下文用的。上下文不够长，税就扎眼。再往长走，Decode 侧 DSA 的 TPOT 优势大约在 **10k–20k** token 翻过来，斜率大约 **6×** 更陡。`DeepseekV32IndexerBackend` 当时还新。

## 拆开 Prefill（V3.2）

RDMA 上 1P+1D 的速通（页上预告下一篇讲 NVL72 GB200）。Nixl KV Connector；P 和 D 都是 **TP2**。

```bash
# Prefill Node
export VLLM_USE_FLASHINFER_MOE_FP4=1
export UCX_NET_DEVICES=mlx5_bond_0:1   # optional，告诉 NIXL 用哪块 RDMA
export VLLM_NIXL_SIDE_CHANNEL_HOST=${PREFILL_NODE_IP}
vllm serve nvidia/DeepSeek-V3.2-NVFP4 -tp 2 --max-num-batched-tokens 20480 \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda"}' \
  --port 8000

# Decode Node：环境和 CLI 一样，只换 VLLM_NIXL_SIDE_CHANNEL_HOST=${DECODE_NODE_IP}

# Proxy
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8000 \
  --prefiller-hosts ${PREFILL_NODE_IP}   --prefiller-ports 8000 \
  --decoder-hosts ${DECODE_NODE_IP}      --decoder-ports   8000
# 多 P 或多 D：往 hosts/ports 后面追加

vllm bench serve --model nvidia/DeepSeek-V3.2-NVFP4 \
  --seed $RANDOM --dataset-name random \
  --base-url http://${PROXY_NODE_IP}:8000 \
  --tokenizer /mnt/models/DeepSeek-V3.2   \
  --num-prompts 500    --max-concurrency 100 \
  --random-input-len 4096  --random-output-len 1024 \
  --ignore-eos
```

**原文注：** v0.14.1 做 PD 拆分要手工打 [PR #32698](https://github.com/vllm-project/vllm/pull/32698) 的补丁。更新的 main 已经合了。

并发涨起来，拆分比一体机吞吐更好（缺口变大），TTFT / TPOT 更低，延迟斜率也更稳。batch **256** 时拆分把 TPOT 压在 **60 ms** 内；一体机超过 **80 ms**。1P1D 和 3P1D 的 TPOT 都赢一体机。

![dsv32 pd disagg throughput](../../../../assets/vllm/blog/performance/gb300-deepseek/14-dsv32-pd-disagg-throughput.png)

![dsv32 pd disagg tpot](../../../../assets/vllm/blog/performance/gb300-deepseek/15-dsv32-pd-disagg-tpot.png)

**Figure 14–15。** V3.2 拆分 vs 一体的吞吐和 TPOT。

ISL 从 2k 到 8k，1P1D 的 Prefill 成瓶颈：请求在 P 上排队，Decoder 吃不饱。再加两份 P（3P1D）把更多 Prefill 并行掉，总吞吐才上来。单卡吞吐未必最高，但多投硬件能换更好的 Goodput 和 SLO。

![dsv32 pd disagg throughput isl8k](../../../../assets/vllm/blog/performance/gb300-deepseek/16-dsv32-pd-disagg-throughput-isl8k.png)

**Figure 16。** ISL=8k：1P1D vs 3P1D 的总吞吐。

## 致谢（页上点名）

- [Verda](https://verda.com/?utm_source=vllm&utm_medium=referral&utm_campaign=gb300-deepseek) 出 GB300 集群。
- DaoCloud：Xingyan Jiang、Nicole Li、Peter Pan、Kebe Liu。
- InferAct：Jie Li、Kaichao You。
