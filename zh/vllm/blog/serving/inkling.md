---
source: https://vllm.ai/blog/2026-07-15-inkling
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# TML Inkling：1T 多模，相对位置，短卷积当 KV

英文对照：[en/vllm/blog/serving/inkling.md](../../../../en/vllm/blog/serving/inkling.md)  
原文：https://vllm.ai/blog/2026-07-15-inkling  
2026-07-15。署名 **vLLM Team**。数字是 **4× GB200** 上的演示。权重：[`thinkingmachines/Inkling-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-NVFP4)、[`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling)（BF16）。接入：[PR #48768](https://github.com/vllm-project/vllm/pull/48768)。FA4 kernel：[vllm-project/tml-fa4](https://github.com/vllm-project/tml-fa4)。TML 预告：[interaction models](https://thinkingmachines.ai/blog/interaction-models/)。model-runner 旗：[../architecture/mrv2.md](../architecture/mrv2.md)。投机路径：[../performance/spec-decode.md](../performance/spec-decode.md)。**不是新引擎**——sconv cache 被当成虚拟 SWA 层的 KV。当时 **AMD 未支持**（缺 relative-attn kernel）。

Thinking Machines 的 1T 多模：text/image/audio → text，原生 1M（Tinker 暴露 64K/256K）。66 层：11 full + 55 sliding-window GQA。位置不是 RoPE，是 **relative attention**。每层四个 window-4 **sconv**。MoE：256 routed top-6 + 2 shared **expert sink**。NVFP4 只量化 routed expert；MTP 8 头全 BF16。

本地图（原文版权仍归原站；学习对照用）：

![image1](../../../../assets/vllm/blog/serving/inkling/01-image1.png)

![inkling model architecture](../../../../assets/vllm/blog/serving/inkling/02-inkling-model-architecture.png)

![sconv tp sharding](../../../../assets/vllm/blog/serving/inkling/03-sconv-tp-sharding.png)

**Figure（社交 / logo）。** vLLM × Thinking Machines。

## Quick start

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

vllm serve thinkingmachines/Inkling-NVFP4 \
      --tokenizer-mode inkling \
      --reasoning-parser inkling \
      --tool-call-parser inkling \
      --enable-auto-tool-choice \
      --tensor-parallel-size 8 \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 8}' \
      --kernel-config.enable_flashinfer_autotune=False \
      --trust-remote-code
```

页上命令写的是 `--tensor-parallel-size 8`；下面 TPS 是 **4× GB200** 演示。两件事不要捏成一件。

## TL;DR

- **模型：** 上面两个 Hub ID，声称功能对齐。
- **硬件：** NVIDIA Blackwell 和 Hopper。更广的硬件「进行中」。**当时不支持 AMD GPU。**
- **模态：** text/image/audio 进 → text 出。
- **上下文：** 原生到 **1M**（Tinker 窗口 64K / 256K）。
- **功能：** LoRA、MTP 投机解码、TP/DP/EP/PP、prefix caching、拆开 serving。
- **优化：** 认 sconv 的 TP 切分、Lamport fused collective、kernel fusion、multi-streaming、PDL。
- **性能（演示）：** SPEED-Bench 8K in / 1K out，4× GB200：MTP8 **380 tok/s/user**（mean accept **4.5**），无 MTP **140**。
- **精度：** MMAU、MMMU-Pro、BFCL、NIAH-1M、HLE，对一份参考 NVFP4。

## Model architecture

**Figure 1.** 骨架；图上省略 RMSNorm / residual。

**模态。** 1T，原生多模。图像编码器很轻：**hMLP**；音频嵌入 **dMel**（[TML 预告](https://thinkingmachines.ai/blog/interaction-models/)）。嵌入进 decoder-only Transformer。

**注意力。** 66 层：**11** 满 + **55** sliding-window。SWA 比重大，1M 上下文才买得起。所有层 **GQA**，head size **128**。

位置机制是 *relative attention*：学来的相对位置项加在 **pre-softmax** logits 上。**不是 RoPE。**

**Sconv。** 短卷积，窗口 **4**。每层四个模块：attention 的 K、V、输出，以及 MoE 输出。局部混合，算力和内存都小。

**MoE。** 256 routed，**top-6**，外加 **2** 个 shared。每个 token 过 **8** 个专家。shared 是 **expert sink**：参与 routing 分数（吃掉概率质量），但 **不进** top-6 候选。

`Inkling-NVFP4`：只有 **routed** 专家量化成 NVFP4；shared 和 qkvr 线性层仍是 BF16。`Inkling`：MoE 权重也是 BF16。

**MTP。** **8** 个 MTP 头，一步最多 **9** token。头是 **链式** 的：每个吃前一头的 hidden 和已采样 draft token。每头单层 Transformer（满或 SWA）+ dense MLP。**MTP 权重全 BF16。**

## vLLM 接入与优化

**管 sconv cache。** 短卷积要留最后 `W-1` 个 hidden。vLLM 把它当成一层 **虚拟 sliding-window attention** 的 KV。统一 KV manager：滑出窗口的标成可驱逐；prefix caching 同样作用在 sconv 状态上。**不是新缓存池。**

**Figure 2.** 认 sconv 的 TP 切分。

**Sconv-aware TP sharding。** 朴素 TP：all-reduce（例如 `o_proj` 后）→ sconv → residual → RMSNorm。每张卡都对 **完整** hidden 做 sconv，compute 和 cache **整份复制**。

sconv 沿 channel 独立，于是按 channel 切：**reduce-scatter / all-gather** 换掉 all-reduce。每卡只存一份 sconv cache 分片、只算自己的 channel。想法像 sequence parallelism，但切的轴是 **channel** 不是 token。

**低延迟 fused collective。** Lamport 协议的 reduce-scatter / all-gather（跟周围 op 焊在一起），从 FlashInfer 低延迟 all-reduce 扩出来。用 **数据值轮询** 同步，不用显式 barrier。bs=1：kernel **40 µs → 8 µs（5×）**。

**带 sheared bias 的 FA4。** relative attention 把注意力 kernel 的访存打乱。TML 和 Colfax Research 发了 [FA4](https://github.com/vllm-project/tml-fa4)，带 **sheared-bias**；vLLM 直接用。再按配置（batch、TP、KV 长）选 FA4 的 `num_splits`。

**重算 MTP KV。** 每个 MTP 头吃前一头的 draft token，拒绝之后 KV 就脏了。vLLM 缓存基座模型最近几个 token 的 hidden，rejection sampling 之后用 **已接受 token 重跑 MTP 头**。

另外还有 kernel fusion、PDL、multi-streaming。细节在 [PR #48768](https://github.com/vllm-project/vllm/pull/48768)。

### Performance（演示）

4× GB200，SPEED-Bench **8K** in / **1K** out：MTP8 **380 tok/s/user**（mean acceptance length **4.5**），无 MTP **140**。

## Accuracy evals

他们列的每种模态，都对一份参考 NVFP4。长上下文：到 **221K** 完全对齐；到 **513K** 差在约 **1 pp** 内。**800K+** 的 NIAH 跑间方差大；他们说还在把那一档收紧。

| Benchmark / metric | vLLM NVFP4 | Reference NVFP4 | Delta vs Reference |
|---|---:|---:|---:|
| MMAU overall | 76.10% (761/1,000) | 75.50% | +0.60 pp |
| BFCL exact calls | 78.61% (1,062/1,351) | 78.16% | +0.45 pp |
| BFCL All-Live macro | 75.86% | 73.54% | +2.32 pp |
| MMMU-Pro overall micro | 71.12% (3,691/5,190) | 70.52% (3,660/5,190) | +0.60 pp |
| MMMU-Pro Standard 10-option | 70.23% (1,215/1,730) | 70.00% (1,211/1,730) | +0.23 pp |
| MMMU-Pro Standard 4-option | 76.47% (1,323/1,730) | 76.30% (1,320/1,730) | +0.17 pp |
| MMMU-Pro Vision | 66.65% (1,153/1,730) | 65.26% (1,129/1,730) | +1.39 pp |
| HLE | 29.33% (633/2,158) | 26.65% | +2.68 pp |
| NIAH (2K-221K) | 99.09% (436/440) | 99.09% (436/440) | 0.00 pp |
| NIAH (294K-513K) | 95.68% (421/440) | 96.82% (426/440) | −1.14 pp |
| NIAH (586K-805K) | 81.36% (358/440) | 84.09% (370/440) | −2.73 pp |
| NIAH (878K) | 70.91% (78/110) | 80.91% (89/110) | −10.00 pp |

音频 = MMAU；视觉 = MMMU-Pro；tool calling = BFCL；reasoning = HLE；长上下文 = NIAH。

## Roadmap（当时）

- **全局注意力上 FP8：** 全局 attn 仍是 BF16；算力和 KV 容量都会卡。计划改新的 FA4 kernel。
- **图像 / 音频编码器上 CUDA graph：** 这两段当时 **eager**。通常只在 Prefill；上 graph 是为了干掉 CPU 开销。
- **AMD GPU：** **还没有。** relative attention 要单独 kernel。页上写 “coming soon”——没有日期，没有 ROCm flag。

## Acknowledgements

Thinking Machines Lab。模型支持由 [Inferact](https://inferact.ai/) 牵头。
