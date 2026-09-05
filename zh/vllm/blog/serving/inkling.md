---
source: https://vllm.ai/blog/2026-07-15-inkling
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# TML Inkling：1T 多模，相对位置，短卷积当 KV

英文对照：[en/vllm/blog/serving/inkling.md](../../../../en/vllm/blog/serving/inkling.md)  
原文：https://vllm.ai/blog/2026-07-15-inkling  
2026-07-15。署名 **vLLM Team**。数字是 **4× GB200** 上的演示。权重：[`thinkingmachines/Inkling-NVFP4`](https://huggingface.co/thinkingmachines/Inkling-NVFP4)、[`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling)（BF16）。集成：[PR #48768](https://github.com/vllm-project/vllm/pull/48768)。Runner：[mrv2.md](../architecture/mrv2.md)。投机：[spec-decode.md](../performance/spec-decode.md)。P/D：[large-scale.md](large-scale.md)。当时 **AMD 未支持**（缺 relative-attention kernel）。**不是新引擎**——sconv cache 被当成虚拟 SWA 层的 KV。

TML Inkling 是 [Thinking Machines Lab](https://thinkingmachines.ai/) 的 **1T** 多模：**text / image / audio** 进、text 出，原生 **1M**。相对 attention、短卷积、shared expert sink，都接进 vLLM。演示：MTP 最高 **380 tok/s/user**，无 MTP **140**，4 张 GB200。声称功能对齐：LoRA、TP/DP/EP/PP、前缀缓存、分离 serving。准确率和工具解析验过。

本地图（原文版权仍归原站；学习对照用）：

![image1](../../../../assets/vllm/blog/serving/inkling/01-image1.png)

![inkling model architecture](../../../../assets/vllm/blog/serving/inkling/02-inkling-model-architecture.png)

![sconv tp sharding](../../../../assets/vllm/blog/serving/inkling/03-sconv-tp-sharding.png)

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

## TL;DR

- **模型：** Inkling-NVFP4 和 Inkling（BF16）
- **硬件：** NVIDIA Blackwell 和 Hopper。更宽的硬件还在做。
- **模态：** text/image/audio → text
- **上下文：** 原生到 1M（Tinker 暴露 64K 和 256K）
- **功能：** LoRA、MTP、TP/DP/EP/PP、前缀缓存、分离 serving
- **优化：** 认 sconv 的 TP 切法、低延迟 fused collective、kernel 融合、多流、PDL
- **性能：** 4 张 GB200 上 **380 tok/s/user（MTP）**、**140（无 MTP）**
- **准确率：** MMAU、MMMU-Pro、BFCL、NIAH-1M、HLE 对参考实现

## 模型架构

**Figure 1。** TML Inkling 架构（图上省略了 RMSNorm 和残差）。

**模态。** 1T，原生多模态。很轻的图像编码器（hMLP）和音频 embedding（dMel）；见 [TML 的 interaction model preview](https://thinkingmachines.ai/blog/interaction-models/)。Embedding 进 decoder-only Transformer 骨干。

**Attention。** 66 层：**11** 满 attention + **55** sliding-window。SWA 用得重，**1M** 上下文才买得起。全部 GQA，head size 128。

位置不是 RoPE，是 **relative attention**：学来的相对位置项加到 pre-softmax logits。细节在 TML 博。

**Sconv。** 短卷积用得很凶，窗口 **4**。每层四个 sconv：attention 的 K、V、输出，以及 MoE 输出。很小的局部 attention，算力和内存都轻。

**MoE。** 256 routed，top-6，外加 **2 个 shared expert**——每 token 8 个专家。**Expert sink：** 两个 shared 参与 routing 分数（吃概率质量），但 **不进** top-6。

`Inkling-NVFP4`：只有 **routed** 专家是 NVFP4；shared 和 qkvr 线性层仍 BF16。`Inkling`：MoE 权重也是 BF16。

**MTP。** **8 个 MTP 头**，一次 forward 最多 9 个 token。头是 **链式** 的：每个吃上一头的 hidden 和采样出的 draft token。每个头是单层 Transformer（满或 SWA）加稠密 MLP。MTP 权重全 **BF16**。

## vLLM 集成和优化

**管 sconv cache。** 短卷积要留最后 `W-1` 个 token 的 hidden。vLLM 把它当成 **虚拟 sliding-window attention 层** 的 KV。统一 KV cache manager：窗外的状态可驱逐；前缀缓存和 sconv cache 一起工作。

**Figure 2。** 认 sconv 的 TP 切法。

**认 sconv 的 TP。** 朴素 TP：all-reduce（例如 `o_proj` 之后）→ sconv → 残差 → RMSNorm。每张 GPU 都对 **完整** hidden 做 sconv——计算和 sconv cache 都复制一份。

Sconv 沿 channel 维独立，于是按 **channel** 切开：reduce-scatter / all-gather 走 channel 轴，而不是 all-reduce。每张 GPU 只存一份 sconv cache 分片，只算自己的 channel。像 sequence parallelism，但切的是 **channel**，不是 token。

**低延迟 fused collective。** 把 FlashInfer 低延迟 all-reduce 的 **Lamport** 协议延到 fused reduce-scatter / all-gather（连周围的 op）。用数据值轮询同步，不要显式 barrier。bs=1：kernel **40 µs → 8 µs（5×）**。

**带 sheared bias 的 FA4。** Relative attention 把访存弄复杂，attention 流水线变慢。TML 和 Colfax Research 发了 [带 sheared-bias 的 FA4](https://github.com/vllm-project/tml-fa4)；vLLM 接进来，并按配置（batch、TP、KV 长度）选 FA4 的 `num_splits`。

**重算 MTP KV。** 每个 MTP 头吃上一头的 draft token。一拒绝，那份 KV 就脏了。vLLM 缓存骨干最后几个 token 的 hidden，拒绝采样之后用 **接受的 token 重跑** MTP 头。

另外还有 kernel 融合、PDL、多流。细节：[PR #48768](https://github.com/vllm-project/vllm/pull/48768)。

### 性能

**4× GB200**，SPEED-Bench **8K** 进 / **1K** 出：MTP8 **380 tok/s/user**（mean acceptance length **4.5**），无 MTP **140**。

## 准确率

各模态和参考对齐。长上下文：到 **221K** 完全对齐，到 **513K** 大约 **1 pp** 内。800K+ 的 NIAH 跑间方差更大；他们在收紧可复现性。

| Benchmark / metric | vLLM NVFP4 | Reference NVFP4 | Delta vs Reference |
| --- | ---: | ---: | ---: |
| MMAU overall | 76.10% (761/1,000) | 75.50% | +0.60 pp |
| BFCL exact calls | 78.61% (1,062/1,351) | 78.16% | +0.45 pp |
| BFCL All-Live macro | 75.86% | 73.54% | +2.32 pp |
| MMMU-Pro overall micro | 71.12% (3,691/5,190) | 70.52% (3,660/5,190) | +0.60 pp |
| MMMU-Pro Standard 10-option | 70.23% (1,215/1,730) | 70.00% (1,211/1,730) | +0.23 pp |
| MMMU-Pro Standard 4-option | 76.47% (1,323/1,730) | 76.30% (1,320/1,730) | +0.17 pp |
| MMMU-Pro Vision | 66.65% (1,153/1,730) | 65.26% (1,129/1,730) | +1.39 pp |
| HLE | 29.33% (633/2,158) | 26.65% | +2.68 pp |
| NIAH (2K-221K) | 99.09% (436/440) | 99.09% (436/440) | 0.00 pp |
| NIAH (294K-513K) | 95.68% (421/440) | 96.82% (426/440) | -1.14 pp |
| NIAH (586K-805K) | 81.36% (358/440) | 84.09% (370/440) | -2.73 pp |
| NIAH (878K) | 70.91% (78/110) | 80.91% (89/110) | -10.00 pp |

## Roadmap

- **全局 attention 的 FP8：** 现在是 BF16；算力和 KV 容量都可能卡。打算改新的 FA4 kernel 试 FP8。
- **图像和音频编码器的 CUDA Graph：** 现在 eager。通常只在 Prefill；上 graph 是为了砍 CPU。
- **AMD：** 还没有——relative attention 要专用 kernel。在路上。

## 致谢

Thinking Machines Lab。模型支持由 [Inferact](https://inferact.ai/) 牵头。
