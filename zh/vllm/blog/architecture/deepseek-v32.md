---
source: https://vllm.ai/blog/2025-09-29-deepseek-v3-2
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DeepSeek-V3.2-Exp：细粒度稀疏注意力进了 continuous batch

英文对照：`en/vllm/blog/architecture/deepseek-v32.md`  
原文：https://vllm.ai/blog/2025-09-29-deepseek-v3-2  
2025-09-29。Day-0。后续 V4 压缩栈见 [deepseek-v4](deepseek-v4.md)；GB300 上 V3.2 vs R1 见 [gb300-deepseek](../serving/gb300-deepseek.md)；宽 EP 见 [gb200-wideep](../serving/gb200-wideep.md)。

DSA：lightning indexer 挑 top-2048，再稀疏注意力。Indexer 有自己的 K cache，和 MLA KV 分开分配。Prefill / decode 布局不同，continuous batch 要把因果窗口用 `ks` / `ke` 标出来。


本地图（原文版权仍归原站；学习对照用）：

![dsa explained](../../../../assets/vllm/blog/architecture/deepseek-v32/01-dsa-explained.png)

![mla indexer block](../../../../assets/vllm/blog/architecture/deepseek-v32/02-mla-indexer-block.png)

## Cache 长什么样

MLA 每 token **656 bytes**：512× `float8_e4m3` NoPE + 4× `float32` scale（每 128 个一份）+ 64× `bfloat16` RoPE（不量化）。Indexer key 按 **block** 存：前 `block_size * head_dim` 是值，后面是 scale。所以当时只支持 `block_size` 64——FlashMLA 也按这个切。一个 token 的 indexer cache **不连续**。FP8 KV 的另一条线：[fp8-kvcache](../performance/fp8-kvcache.md)。

## Top-K 怎么 batch

单 query：`(h, d)` 对 `(n, d)` → `(n,)` 再 top-2048，不够填 `-1`。同请求多 query（prefill）：`(q, h, d)`，`ks` 全 0，`ke = range(n-q, n)`。多请求：query / context 沿 batch 维拼，`ks` / `ke` 按每条的因果窗口加偏移。`deep_gemm.fp8_mqa_logits(...)`。高 batch × 长上下文会先物化整张 logits 再 row-wise topk——这是性能坑。融合 topk（TileLang）、写入 page table 时量化 MLA latent 和 indexer key。

```bash
vllm serve deepseek-ai/DeepSeek-V3.2-Exp --tensor-parallel-size 8
```

当时 16×H100 / 8×H200 / 8×B200。EP 有 bug 在修。计划：llm-d + NIXL P/D、核 Hopper/Blackwell 以外的硬件（`vllm-ascend` / `vllm-mlu` 已有 V3.2）、短序列 masked MHA、Hadamard 当时去掉了。Indexer 还新，GB300 文里 DSA 单层相对 MLA 约 **2.7×** kernel 时间。
