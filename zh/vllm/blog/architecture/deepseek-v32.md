---
source: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DeepSeek-V3.2-Exp：细粒度稀疏注意力进了 continuous batch

英文对照：[en/vllm/blog/architecture/deepseek-v32.md](../../../../en/vllm/blog/architecture/deepseek-v32.md)  
原文：https://vllm.ai/blog/2025-09-29-deepseek-v3-2  
2025-09-29。署名 **vLLM Team**。[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 的 Day-0，带 DeepSeek Sparse Attention（DSA）（[论文](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)）。后续 V4 压缩 / 1M 上下文栈见 [deepseek-v4](deepseek-v4.md)；GB300 上 V3.2 vs R1（DSA 单层相对 MLA 约 **2.7×** kernel 时间）见 [gb300-deepseek](../performance/gb300-deepseek.md)；宽 EP 见 [gb200-wideep](../serving/gb200-wideep.md)；FP8 KV 布局亲戚：[fp8-kvcache](../performance/fp8-kvcache.md)。第一版落地笔记；官方精度还在对，**expert parallelism 当时有 bug**。

适用：V3.2-Exp 用 tensor parallelism 在 **16×H100 / 8×H200 / 8×B200** 上 serve，`block_size` 64，indexer K cache 和 MLA KV 分开，因果窗口用 `ks` / `ke` 标。不适合：把这页当后来的 V4 hybrid-KV 栈；也不要假定 day-0 的 EP 是干净的。

菜谱：[DeepSeek-V3_2-Exp](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)。初始支持 PR：[#25869](https://github.com/vllm-project/vllm/pull/25869)。已知问题：[#25877](https://github.com/vllm-project/vllm/issues/25877)。

DSA：lightning indexer 挑 **top-2048**，再稀疏注意力。Prefill / decode 布局不同，continuous batch 要把因果窗口标出来。Kernel：DeepGEMM 的 lightning indexer CUDA，FlashMLA 的稀疏注意力。跟 NVIDIA 一起把 Blackwell 做上：**B200** 和 **GB200** 能直接跑。

![dsa explained](../../../../assets/vllm/blog/architecture/deepseek-v32/01-dsa-explained.png)

**Figure 1。** DeepSeek Sparse Attention（DSA）（学习对照；版权仍归原站）。

## 怎么用

装好之后，16×H100、8×H200 或 8×B200 上走 tensor parallelism（EP 有个小 bug 当时在修）：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp --tensor-parallel-size 8
```

规模化他们预告：那一周稍后用 `llm-d` 一键 Kubernetes——vLLM 配 **NIXL** 做 PD 分离，再把请求按 data-parallel rank 分到各 P / D 实例。文档页上写「很快」。

建议拿 **长输入、或会吐长输出的 prompt** 测。对照 **V3.1-Terminus**（同一份数据配比上继续预训练）。

精度：还在对官方数字。**上一版** 权重上，GSM8K 和 GPQA-Diamond 对上了预期，跟 V3.1-Terminus 接近。

## vLLM 里的 Top-K 稀疏注意力

### 新的 cache 条目和量化格式

Lightning indexer 给索引单独 cache **K**。每个 token 因此多一份 K cache，跟 MLA 的分开。vLLM 给 indexer K 和 MLA K **分开分配 buffer**。

![mla indexer block](../../../../assets/vllm/blog/architecture/deepseek-v32/02-mla-indexer-block.png)

**Figure。** MLA KV vs indexer-key 的 block 布局（学习对照）。

这模型支持 FP8 KV。**MLA** 每 token 的 KV 是 **656 bytes**：

- 前 **512** 字节：量化 NoPE——512× `float8_e4m3`
- 接着 **16** 字节：scale——4× `float32`。第一个 `float32` 管前 128 个 `float8_e4m3`，第二个管下 128 个，以此类推
- 最后 **128** 字节：RoPE——64× `bfloat16`。**不量化**（精度）

Indexer key 按 **block** 存。所以这模型当时只支持 `block_size` **64**——另一原因是 FlashMLA 也按这个切。前 `block_size * head_dim` 是值，后面是 scale：

```
x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
x_fp8[ :, block_size * head_dim :] = scales.view(num_blocks, block_size).view(dtype=torch.uint8)
```

Indexer 里，**一个 token 的 cache 不连续**。

### 带 mask 的新计算

每个新 query token 先过 indexer，挑要 attend 的 **top 2048**。Query 是 `(h, d)`（`h` 个 query head，`d` 是 head dim）。长度 `n` 的 context 是 `(n, d)`。Logits（query 对 context）是 `(n, h)`。乘 head 权重 `(h,)` → `(n,)`。产出 `(2048,)` 的整数下标，不够 2048 就填 **`-1`**。

单 query 好懂；batch 不容易。DeepGEMM 调用：

```
logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
```

**同一请求里多个 query（prefill）：** query `(q, h, d)`，context 仍 `(n, d)`，logits `(q, n, h)`，乘 head 权重后 `(q, n)`，下标 `(q, 2048)`。因果：每个 query 只看它前面的 token。`ks` / `ke` 是 `(q,)` 整数，标 context 起止。这时 `ks` 全 0，`ke` 是 `list(range(n - q, n, 1))`。

**多请求：** `b` 条，query 数 `q1…qb`，context 数 `n1…nb`。Query 拼成 `(q1+…+qb, h, d)`，context 拼成 `(n1+…+nb, d)`，logits `(q1+…+qb, n1+…+nb, h)`，下标 `(q1+…+qb, 2048)`。`ks` 和 `ke` 长度都是 `q1+…+qb`。

页上印的：`ks` 是 `[0] * q1 + [q1] * q2 + … + [q1 + q2 + … + qb] * qb`（`*` 表示列表重复）。`ke` 是 `list(range(n1 - q1, n1, 1)) + … + list(range(nb - qb, nb, 1))`，**再加上 `ks` 的偏移**。

Logits 之后做 `topk`。他们点名的坑：**高 batch × 长上下文** 会先把整张 logits **物化**，再做 row-wise `topk`——性能洞。

### 融合、更多 kernel、Blackwell

页上先摘的低垂果实：

- 融合 Top-K。DeepSeek 的 TileLang kernel 当参考
- MLA latent 和 indexer key **写入 vLLM page table 时就量化**——不轻松，因为上面那套格式是新的

开箱的 **Blackwell** 支持，跟 NVIDIA 一起。他们希望以后发模型 Blackwell 就是一等公民。

## 还在做的（发文时）

DSA / 稀疏注意力才刚摸到皮：

- Hopper / Blackwell 以外的架构
- AMD 和 TPU；插件门在，别人可以直接加——[vllm-ascend](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0rc0) 和 [vllm-mlu](https://github.com/Cambricon/vllm-mlu) **已经** 有 V3.2
- 大规模 wide EP 和分离，还在测
- 端到端 RL 环「很快」
- DeepSeek 说的 **短序列 prefill 的 masked MHA mode**
- 这版 **去掉了 Hadamard**——他们看精度没差；还要再查

## 致谢

社区里点名的组：

- **vLLM：** Chen Zhang、Yongye Zhu、Kaichao You、Simon Mo、Zhuohan Li
- **Red Hat：** Lucas Wilkinson、Matt Bonanni、Wentao Ye、Nicolo Lucchesi、Michael Goin、Robert Shaw、Tyler Michael Smith
- **Meta：** Lucia Fang、Xiaozhu Meng、Lu Fang
- **NVIDIA：** Ray Wang、Barry Kang、Daniel Campora、Julien Demouth、Siyuan Fu、Zeyu Wang、Pen Chun Li

DeepSeek 开源模型、技法和 kernel；DeepSeek 管理层对 vLLM 的信任。
