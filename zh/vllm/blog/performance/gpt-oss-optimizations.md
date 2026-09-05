---
source: https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# gpt-oss on Blackwell：推的是 Pareto，不是单点 TPS

英文对照：[en/vllm/blog/performance/gpt-oss-optimizations.md](../../../../en/vllm/blog/performance/gpt-oss-optimizations.md)  
原文：https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations  
2026-02-01。署名 **The vLLM and NVIDIA team**。学习笔记。B200/GB200 上的 bench，不是你的 SLA。更早的 InferenceMAX 铭牌：[blackwell-inferencemax.md](blackwell-inferencemax.md)。Day-0 模型文：[gpt-oss.md](../serving/gpt-oss.md)。`torch.compile` 那扇门：[torch-compile.md](../architecture/torch-compile.md)。FP8 KV：[fp8-kvcache.md](fp8-kvcache.md)。持续榜：[SemiAnalysis Inference MAX](https://inferencemax.semianalysis.com/)、[vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)。系统 TPS ≠ 每用户 TPS。

适用：在 Blackwell 上伺候 gpt-oss-120b，拧 FlashInfer / async / stream-interval。不适合：把 **+38%** 当成你机房的承诺。

**原文 TL;DR。** gpt-oss-120b MXFP4 MoE on Blackwell。硬软共设计：FlashInfer、`torch.compile` fusion、async scheduling、stream interval。max-throughput 约 **+38%**，min-latency 约 **+13%**——Pareto 两端一起动（TPS/GPU vs TPS/user）。菜谱：`--cuda-graph-capture-size 2048`；高并发 `--api-server-count 20` 或 `--stream-interval 20`；`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`。当时 DEP2 投影比 TP 好，实测更差（MoE kernel 选错）。跟 [Issue #30758](https://github.com/vllm-project/vllm/issues/30758)。

## 引言

只优化一个数——最高 TPS 或单 batch 延迟——对不上真实部署。SLO 和并发都不一样。真正要推的是 **Pareto frontier**：每 GPU 的 TPS（TCO）对上每用户的 TPS（交互性）。往右上走 = 单用户更快，同一块硅上还能多塞人。InferenceMAX 就是为量这条包络。

`gpt-oss-120b` 原生 4-bit（MXFP4）MoE。页上写 SoTA-for-size 加 agentic。vLLM 在 InferenceMAX showcase 上用 B200/GB200 伺候它。Blackwell 带来原生 FP4 Tensor Core 和每卡 **192 GB** HBM——必要，不够。剩下是 kernel fusion、少通信、host–device overlap。

## FlashInfer + torch.compile fusion

Blackwell 上 attention、MoE、其它 fused 计算，主 backend 是 FlashInfer。

**计算 kernel：**

- **MoE：** `trtllm-gen`（[#23819](https://github.com/vllm-project/vllm/pull/23819)）和 CUTLASS（[#23696](https://github.com/vllm-project/vllm/pull/23696)）走 FlashInfer。专家路由 / 计算选更快的那只。FlashInfer 还 JIT、autotune、缓存 kernel。
- **FP8 KV-cache：** 同一份 KV 预算能塞更多 in-flight；一部分 attention 算子走 FP8，算力和内存都轻。FlashInfer attention：[#25674](https://github.com/vllm-project/vllm/pull/25674/)。

**`torch.compile` 做图融合。** 不是手写死融合。vLLM 的 [compilation 基建](https://github.com/vllm-project/vllm/tree/main/vllm/compilation) 自动熔——更好推广、更好养。

- **AR + RMSNorm：** AllReduce 和 RMSNorm 熔在一起。TP 上通信否则会占上风。[#20691](https://github.com/vllm-project/vllm/pull/20691)。
- **Pad+Quant / Finalize+Slice：** 当时还在滚（[#30647](https://github.com/vllm-project/vllm/pull/30647)），MoE 路径上预期约 **6%**。

新的 fused op 继续走同一套基建。

## 运行时

Blackwell 上 GPU 可能在等 host：dispatch、`prepare_batch`、调度、sampling。kernel 之间留缝。

**Async scheduling**（[#23569](https://github.com/vllm-project/vllm/pull/23569)）：

- GPU 还在跑当前 batch，CPU 已经在准备下一批。
- 更强的卡（H200 / B200 / GB200）大约 **10%**。gpt-oss 的高吞吐和 min-latency 两边都用得上。
- 后来的发版默认开。

**Stream interval**（[#27869](https://github.com/vllm-project/vllm/pull/27869)）：

- 后续 token 先缓冲再 HTTP/gRPC 发出去。**首 token 仍立刻发**（TTFT 不垫高）。
- 少付每 token 序列化的 CPU 税。gpt-oss-20b、**1024** 并发：他们报端到端约 **57%**——那是 **输出队列** 瓶颈被松开，不是 kernel 变 57%。TPOT 也更好。
- `--stream-interval <num_tokens>`。默认 `1`。冲吞吐的菜谱试 `10` 或 `20`。

## 部署菜谱

多数优化在新发版里已经是默认。复现 gpt-oss on B200/GB200（Recipes 页同一套）：

- `--cuda-graph-capture-size 2048`
- 高并发：`--api-server-count 20` **或** `--stream-interval 20`（把 HTTP 从引擎上拆开）
- MoE：`VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`（CUTLASS FP8/FP4 MoE）

## 结果

相对 [InferenceMAX 立项那篇](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax)：max-throughput **+38%**，min-latency **+13%**——整条曲线，不是一个工作点。

![gpt-oss 120b 8k/1k Nov–Jan](../../../../assets/vllm/blog/performance/gpt-oss-optimizations/01-gpt-oss-120b-8k-1k-nov-jan.png)

**Figure（页上）。** gpt-oss-120b 8K/1K Pareto，十一月 → 一月。

## 下一步（[Issue #30758](https://github.com/vllm-project/vllm/issues/30758)）

- **拆分。** Prefill 和 Decode 放到不同 GPU；还在找能赢一体机 TPS/GPU 的配置。
- **DEP2。** 投影：两卡 Attention DP + MoE EP，同一 TPS/user 下该赢 TP1/TP2。实测：**更差**，MoE kernel 选错。在修。
- **Min-latency**（TP8，concurrency 8）：RoPE+Q+Cache fusion（kernel 在 FlashInfer，vLLM 接入当时还在做）；router / `fc_qkv` / `fc_o_proj` 用带 PDL 的 tiny GEMM。

## 致谢（页上点名）

- Red Hat：Michael Goin、Alexander Matveev、Lucas Wilkinson、Luka Govedič、Wentao Ye、Ilia Markov、Matt Bonanni、Varun Sundar Rabindranath、Bill Nell、Tyler Michael Smith、Robert Shaw
- NVIDIA：Po-Han Huang、Pavani Majety、Shu Wang、Elvis Chen、Zihao Ye、Duncan Moss、Kaixi Hou、Siyuan Fu、Benjamin Chislett、Xin Li、Vadim Gimpelson、Minseok Lee、Amir Samani、Elfie Guo、Lee Nau、Kushan Ahmadian、Grace Ho、Pen Chun Li
- vLLM：Chen Zhang、Yongye Zhu、Bowen Wang、Kaichao You、Simon Mo、Woosuk Kwon、Zhuohan Li
- Meta：Yang Chen、Xiaozhu Meng、Boyuan Feng、Lu Fang
