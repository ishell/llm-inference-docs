---
source: https://vllm.ai/blog/2026-08-22-rdt-weight-transfer
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# RDT 分片权重搬运：Kimi K2 7.53 秒

英文对照：`en/vllm/blog/serving/rdt-weight-transfer.md`  
原文：https://vllm.ai/blog/2026-08-22-rdt-weight-transfer  
Kimi K2 BF16：48×(8×H100)，训练 32 节点、推理 16 节点，**7.53 s** 同步约 **7.9 TB**，聚合带宽约 **1049 GB/s**。

NCCL broadcast 把整份 HF 权重灌进每个推理 rank：TP8 只留 ⅛，宽 EP 更浪费，集体通信还怕掉队。RDT（NIXL）改成 **推理 rank 按需 pull 自己的分片**。


本地图（原文版权仍归原站；学习对照用）：

![rdt blog overview](../../../../assets/vllm/blog/serving/rdt-weight-transfer/01-rdt_blog_overview.png)

![layerwise reloading](../../../../assets/vllm/blog/serving/rdt-weight-transfer/02-layerwise_reloading.webp)

![rdt blog init flow](../../../../assets/vllm/blog/serving/rdt-weight-transfer/03-rdt_blog_init_flow.png)

![AllScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/04-AllScenes.gif)

![ExpertScenes](../../../../assets/vllm/blog/serving/rdt-weight-transfer/05-ExpertScenes.gif)

![rdt pipelined execution 2x](../../../../assets/vllm/blog/serving/rdt-weight-transfer/06-rdt_pipelined_execution-2x.png)

![rdt qwen weight sync latencies](../../../../assets/vllm/blog/serving/rdt-weight-transfer/07-rdt_qwen_weight_sync_latencies.png)

![rdt fault tolerance](../../../../assets/vllm/blog/serving/rdt-weight-transfer/08-rdt_fault_tolerance.png)

## 录像张量

vLLM loader 的 fuse / 转置 / GQA 下 Q 与 KV 切法不同 / Llama-4 expert 拆分，没法在训练侧手写。初始化时塞一只 **recording tensor**（有 shape/dtype、没数据），把 view/narrow/transpose 链记成 sharding plan。训练侧按 plan 切 BF16 分片再传；推理侧做 process + copy 进 live 权重，CUDA graph 还能留。

SkyRL 上 Qwen3-235B：NCCL 64.72 s → 分片 V1 25 s → PP/EP 本地 V2 5.61 s → pipeline V3 **3.49 s**。Kimi 层内 MoE 一块约 30 GB，不能每卡 gather 全专家。

SkyRL：`generator.inference_engine.weight_sync_backend=sharded_rdt`，且 `colocate_all=false`。别家框架实现 `WeightSource` iterator。

当时限制：loader 必须可记录；RDT 缓冲 **不计入** `gpu_memory_utilization`；**不能**和 EPLB 一起；按 trainer PP 串行传以免 layerwise OOM。推理副本挂了，NIXL 还能让剩下的继续 sync——broadcast 集体做不到。和 [Native RL](native-rl.md) 一起读：那篇是 pause/keep，这篇是 **怎么搬权重**。
