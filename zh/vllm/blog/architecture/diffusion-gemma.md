---
source: https://vllm.ai/blog/2026-06-10-diffusion-gemma
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# DiffusionGemma：把画布当成一次全拒的草稿

英文对照：`en/vllm/blog/architecture/diffusion-gemma.md`  
原文：https://vllm.ai/blog/2026-06-10-diffusion-gemma  
2026-06-10。Google DeepMind。第一只进 vLLM 的 dLLM。数字是单卡 batch=1 的演示。执行核见 [mrv2](mrv2.md)；草稿账本见 [spec-decode](../performance/spec-decode.md)；多模态流水线见 [vllm-omni](../serving/vllm-omni.md)。

26B，Gemma4 骨干。不是从左往右吐字：对 **256 token 画布** 反复去噪，用算力换带宽——低 batch 时带宽才是瓶颈。块内并行，块间仍从左到右。


本地图（原文版权仍归原站；学习对照用）：

![ar vs diffusion](../../../../assets/vllm/blog/architecture/diffusion-gemma/01-ar-vs-diffusion.svg)

![sampling loop horizontal](../../../../assets/vllm/blog/architecture/diffusion-gemma/02-sampling-loop-horizontal.svg)

![denoising grid](../../../../assets/vllm/blog/architecture/diffusion-gemma/03-denoising-grid.svg)

![self conditioning](../../../../assets/vllm/blog/architecture/diffusion-gemma/04-self-conditioning.svg)

![stack](../../../../assets/vllm/blog/architecture/diffusion-gemma/05-stack.svg)

![per seq causal attention](../../../../assets/vllm/blog/architecture/diffusion-gemma/06-per_seq_causal_attention.svg)

![per seq sliding window](../../../../assets/vllm/blog/architecture/diffusion-gemma/07-per_seq_sliding_window.svg)

![perf](../../../../assets/vllm/blog/architecture/diffusion-gemma/08-perf.svg)

## 两套注意力，一份权重

**Encoder：** 因果注意力，写 KV。每块两次：prefill prompt；块收敛后 commit。  
**Decoder：** 双向注意力，只读 KV。去噪。

因果 KV 跟自回归一样写，自动 prefix cache 不用改。熵预算：按置信从高到低收 token，累加熵超预算就停；早期只钉几个锚，邻居跟着锐。收敛：argmax 连续几步不变且均值熵低于阈值，或撞步数上限。Commit 的是干净 argmax，不是噪声画布。Self-conditioning：下一步吃上一步 softmax 的加权 embedding（gated MLP），只在 decoder。

## 接到投机路径上

画布 = 一大串 draft，整块接受或整块拒绝。`num_sampled=0` 时 KV 指针不动。**ModelState** 钩子：`prepare_inputs` / `prepare_attn` / `custom_sampler` / `add_request`。新块扩散模型只需一只 ModelState，不必改 runner。

`DiffusionSampler`（`torch.compile`）：prefill 随机初始化画布；denoise 用 Gumbel-max + 熵边界；commit 吐 256 token。同 batch 里 prefill / denoise / commit 混在一起 → **逐请求因果性**（`TRITON_ATTN` / `FLASH_ATTN`）。Sliding window 在画布上变成对称 `2W+1`。

FP8 / NVFP4 checkpoint 在 RedHatAI。H200 FP8 生成约 **1288 tok/s**（相对普通 AR ~6×，相对 MTP ~3×）；H100 约 **1008 tok/s**。演示。
