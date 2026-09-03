---
source: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Hybrid SSM 的 P/D 分离：两种记忆，同一根 RDMA 管子

英文对照：[en/vllm/blog/serving/hybrid-ssm.md](../../../../en/vllm/blog/serving/hybrid-ssm.md)  
原文：https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg  
2026-04-21。`vllm>=0.20.0`。Nemotron-H 这类模型：Mamba 式 SSM 层和满注意力层插花。标准 Transformer 的 NIXL P/D 按统一 KV 块来；SSM 不是那种块。这篇是加法：没有 SSM 层时旧路径不动。

FA 层的 KV 仍是按 token 的 K/V。Mamba 存的是整段历史的**塌缩**：conv state 例如 `(3072, 3)` bf16，SSM state 例如 `(32, 64, 128)` fp32。没有「token 维」，一块就是一份完整快照，`block_size` 对它等于 1。

Hybrid Memory Allocator 按类型分组、再让两组共用同一块物理张量。同一页，FA 看成 K/V，Mamba 看成 conv+SSM+padding。页大小被抬到对齐。NIXL 原来一份统一 `(address, length)` 描述符无法同时索引两种视图。


本地图（原文版权仍归原站；学习对照用）：

![transfer volume vs isl](../../../../assets/vllm/blog/serving/hybrid-ssm/01-transfer-volume-vs-isl.png)

![disagg vs colocated](../../../../assets/vllm/blog/serving/hybrid-ssm/02-disagg-vs-colocated.png)

## 三件补丁

**双描述符。** 同一片物理内存登记两套 NIXL 描述符，拼在一个 transfer handle 里：前面 FA（K/V 分开，为了异构 TP 能按 head 切），后面 Mamba。`block_id → descriptor_id` 按组选不同的 stride。

**物理块 vs 逻辑块。** FlashInfer 一类要固定物理块（例如 16 token），和用户/HMA 的逻辑块可能不同。FA 按比例拆；SSM 没有 token 可拆，始终用逻辑块数。P 和 D 的 TP 不同时，这个比例还可能两边不一样。

**Conv 的三个描述符。** 同构 TP 直接整块读。异构（例如 P_TP=1、D_TP=4）时，SSM 按 heads 切还好切；conv 在 SD 布局 `(state_len, dim)` 里 x/B/C 交错，RDMA 零拷贝捞不到连续字节。要求 `VLLM_SSM_CONV_STATE_LAYOUT=DS`（`(dim, state_len)`），x、B、C 各自连续，再登记三份描述符。同构 TP 可以退化成 Conv+SSM 两份。

相关 PR：#36687 双描述符与同构 TP；#37416 DS conv 布局；#37635 异构 3-desc；#37310 Mamba P/D 的 N-1 prefill。HMA 接口是底座。

Router 那篇的 P/D 假定记忆长得一样。混合模型把「一块」拆成两种方言——管子还是 NIXL，词典要两本。
