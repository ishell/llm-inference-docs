---
source: https://vllm.ai/blog/2026-08-07-decode-context-parallelism
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Decode Context Parallelism：长上下文别再按头去切 KV

英文对照：[en/vllm/blog/performance/dcp.md](../../../../en/vllm/blog/performance/dcp.md)  
原文：https://vllm.ai/blog/2026-08-07-decode-context-parallelism  
2026-08-07。vLLM 支持 DCP 已近一年，这篇是在 agent 把上下文拉到 64K–1M 之后把它写清楚。`vllm serve` 里的 `-dcp` / `--decode-context-parallel-size`。NVIDIA TensorRT-LLM 侧相近的方向叫 Helix Parallelism。

Agent 要读仓库、要带着长对话。KV 按这个长度长。基线 TP 按 **attention head** 切 KV：GQA 切到每卡一个 KV head 就到底，再加卡就开始复制；MLA 等于只有一个 KV head，latent 在每个 TP rank 上整份复制。[分布式推理](../serving/distributed-inference.md) 那张「TP 给 KV 腾房间」的超线性图，在 MLA 上会反过来。房子被复制填满，并发上不去。

DCP 按 **序列维** 切：同一条 200K 的请求，四张卡可以各管 50K 的 KV。每卡只存、只读自己那一段，batch 才能再涨。要高带宽的卡间互联。


本地图（原文版权仍归原站；学习对照用）：

![kv parallelism overview](../../../../assets/vllm/blog/performance/dcp/01-kv-parallelism-overview.svg)

![figure 1](../../../../assets/vllm/blog/performance/dcp/02-figure-1.png)

![figure 2](../../../../assets/vllm/blog/performance/dcp/03-figure-2.png)

![figure 3](../../../../assets/vllm/blog/performance/dcp/04-figure-3.png)

![figure 4](../../../../assets/vllm/blog/performance/dcp/05-figure-4.png)

![figure 5](../../../../assets/vllm/blog/performance/dcp/06-figure-5.png)

## 成绩（演示）

8×B200，Kimi K2.6 NVFP4，单机。Mooncake-trace 格式的公开 agent 轨迹（中位输入约 67K、输出约 400；约一半 ≥64K，尾部到约 1M）。并发从 16 扫到 512。

基线 TP：并发 64 时 KV 100%，吞吐卡在大约 **1,863 tok/s/GPU**。DCP：并发 512 仍只有约 **82%** KV，大约 **6,091 tok/s/GPU**。按序列长度分桶，200k+ 仍能停在同一条吞吐–交互 Pareto 上——复制 KV 的 TP 在这里已经 OOM。

CATALOG 摘要写「长上下文 agent 上大约 3× 吞吐」；精确数字以原文表为准。

## 一拍怎么走

AllGather Q → 本地算 attention → AllGather+ReduceScatter 合并（`cp_lse_ag_out_rs`）。Decode 时 Q 只有一个 token，AllGather 便宜。MLA 可选 `VLLM_DCP_Q_REPLICATE=1`，加载时在 DCP 组内复制 query 投影，decode 连这步 AllGather 也省。合并用 online-softmax 的 LSE 再加权。

## 怎么开

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --tensor-parallel-size 2 \
    --decode-context-parallel-size 2
```

**MLA**（DeepSeek-V2/V3/R1、Kimi K2.6）：整份 latent 都该按序列切。约束：`TP >= DCP` 且 `TP % DCP == 0`。DeepSeek-R1 可以 TP8 + DCP8。

**GQA**（Llama、Qwen3-235B）：TP 先按 KV head 切；`TP > num_kv_heads` 才出现复制，DCP 去填那些副本。约束：`(TP // num_kv_heads) >= DCP` 且能整除。Qwen3-235B `num_kv_heads=4`，TP8 最多 DCP=2。

下一步：更细的 TP/DCP 组合、更好的 A2A、MTP / 投机解码、P/D 分离上的 DCP、Prefill Context Parallelism（PCP，serve CLI 里的 `-pcp`）。社区也在接 GLM-5.2、Kimi K3。

长上下文的地图：TP 切头、DCP 切序列、Mooncake 把前缀放到池子里、P/D 把阅读和说话拆开。四件事不是互斥的。
