---
source: https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# EAGLE3 on Instinct：Quark MXFP4，先量 served acceptance length

英文对照：[en/vllm/blog/performance/eagle3-amd.md](../../../../en/vllm/blog/performance/eagle3-amd.md)  
原文：https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct  
2026-07-13。署名 **Larry Li, Chao Li, Haichen Zhang, Chun Fang, Andy Luo, Spandan Tiwari, and Ashish Sirasao**（AMD Quark）。学习笔记。MI355X / InferenceX 上的 bench，不是你的 SLA。验收数学：[spec-decode.md](spec-decode.md)。CUDA 侧 EAGLE：[p-eagle.md](p-eagle.md) / [eagle-3-1.md](eagle-3-1.md)。后来五条 ROCm 路：[spec-decode-amd.md](spec-decode-amd.md)。Hidden 导出：[extract-hidden-states.md](../architecture/extract-hidden-states.md)。ROCm attention：[rocm-attention.md](../architecture/rocm-attention.md)。草稿家族：[parallel-drafting.md](parallel-drafting.md)。

适用：在 Instinct 上用 vLLM 训 EAGLE3、Quark 量化、ROCm 上 serve。不适合：把 **2.00×** 当承诺——BF16 / FP8 两条 Kimi sweep 的构建和 MML 都不一样。

**原文 TL;DR。**

- 三块：vLLM 进训练环训 EAGLE3 draft；Quark MXFP4/FP8 给 target 和 draft；ROCm/vLLM 上 serve。InferenceX 打在 **MI355X**。
- Kimi-K2.5 1K/1K：**1.69–2.00×**（对上同一条 no-spec 基线）。MiniMax-M2.5：**1.38–1.79×**。他们训的 MiniMax-M3 draft：SPEED-Bench 平均 acceptance length **2.80**。
- draft 精度不必跟 verify 同一档——量化吃的是 **draft 带宽**。随机 prompt 是吞吐微基准。换模型先复测 acceptance length。

## 为什么是投机解码和 EAGLE3

Prefill 可以很快；Decode 仍是 target 一步一个 token。MoE / 注意力重的 target（Kimi-K2.5、MiniMax-M2.5），这个顺序环卡住 serving TPS。

投机解码相对 target 无损：轻草稿先猜几枚；target 一次前向核对。贪心：匹配前缀收下。采样：按 target / draft 概率接受或纠正。第一次拒绝，verifier 吐纠正，草稿从那儿再起；全中则 verifier 再吐一枚 **bonus** token。

**Conditional acceptance rate** = 前面都收下的前提下，这一位还收下的概率。**Acceptance length** = 每个验收周期吐出的 token 数。AL 高能少跑 target 步；落地 TPS 仍要付 drafting + verification 税。

![greedy speculative decoding](../../../../assets/vllm/blog/performance/eagle3-amd/01-figure1.png)

**Figure 1。** 贪心投机，γ=5：target 收下 α=3 的前缀，第一次 mismatch 拒绝，后面草稿丢掉，再吐纠正 → α+1=4。γ 枚全中，多出来的那枚是 target 生成的 bonus。

EAGLE → EAGLE2 → EAGLE3：特征级草稿，再抬接受，再用 target 的低 / 中 / 高层特征加 training-time testing。不是随便一只小 LM。生产上只记一句：生成 TPS 上去，verify 之后输出行为仍是 target 的。

页上点名的别的家族：小 draft、MTP、Medusa、DFlash、DSpark。

## AMD Quark MXFP4

MXFP4 = OCP Microscaling 4-bit 浮点：小块共用 scale。内存接近 INT4，数值更好。Instinct **MI350X / MI355X** 有原生 FP4 矩阵加速——MXFP4 权重直接上硅，松 MoE Decode 的带宽和容量。

[AMD Quark](https://quark.docs.amd.com/latest/) 在大模型落地时出 **day-0** MXFP4（和 FP8）盘，ROCm/vLLM 开箱能跑。例子：[amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4)、[amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4)。这些盘就是 EAGLE3 训练和投机 serving 的 **target**。执行路径：支持的 MXFP4 + **AITER MoE**。投机不改 target 分布：每枚草稿都要核对。

## 用 vLLM 训 EAGLE3

高接受草稿是系统问题。vLLM 坐在训练环里，不只是 serve。跑通的例子：Quark 团队在 Instinct 上训 MiniMax-M3 EAGLE3。后面推理图里的 Kimi-K2.5 / MiniMax-M2.5 draft 是 **社区 HF 盘**，不是这篇训的。

![vLLM-centric EAGLE3 pipeline](../../../../assets/vllm/blog/performance/eagle3-amd/02-figure2.png)

**Figure 2。** 一套 vLLM-on-ROCm：on-policy 合成（Stage 1），流低 / 中 / 高层 hidden（Stage 2），FSDP2 冷启动单层 EAGLE3 头（Stage 3），按测到的 acceptance length 做 in-loop serve-eval（Stage 4），导出并 EAGLE3 serve（Stage 5）。

1. **On-policy 合成。** Quark MXFP4 target 起成 vLLM-ROCm。Chat 走 `/v1/chat/completions`，模板跟后来 serve 一模一样；原始 `/v1/completions`（绕过模板）给非 chat / OOD。引擎和模板不要换。

2. **Hidden 抽取。** 草稿吃 target 内部（低 / 中 / 高 + `fc_norm`），不是另一只小模型。三种模式：**online**（target 和 trainer 同机）、**offline**（落到盘）、**streaming**（活 serve → trainer，不落盘）。**420B MXFP4 MoE** 单节点能训，靠的是 streaming。同一扇门：[extract-hidden-states.md](../architecture/extract-hidden-states.md)。

3. **FSDP2 冷启动。** 单层 EAGLE3 头从零训；TTT loss 加 position-decay。Verifier 就是 Quark MXFP4 target——草稿学的激活空间，部署时还是那一块。

4. **环里 serve-eval。** 训练 loss 会把真实接受说高。周期性导出，用 vLLM 投机解码 serve，按 **served acceptance length** 选盘。

5. **导出。** Hugging Face 格式，vLLM 能吃的 draft 目录，ROCm EAGLE 投机解码。

### SPEED-Bench：11 个域和长上下文

Acceptance length（AL）= 每次 target 验收平均吐出的 token。AL = 1 是一次 verify 吐一个，还没算 drafting 税。

| Domain | AL |
| --- | ---: |
| Coding | 3.32 |
| Math | 3.14 |
| RAG | 3.12 |
| Multilingual | 3.04 |
| Reasoning | 2.89 |
| STEM | 2.86 |
| Summarization | 2.86 |
| Humanities | 2.71 |
| QA | 2.55 |
| Writing | 2.33 |
| Roleplay | 2.01 |
| **Average** | **2.80** |

结构 / 技术域最强；写作 / roleplay 仍有 AL **2.01–2.33**。Prompt 从 1K 到 32K：AL 几乎平（**2.69 → 2.65**）。猜三枚时，第一 / 二 / 三位大约 **76% / 56% / 43%**（累计）。

![MiniMax-M3 AL vs context](../../../../assets/vllm/blog/performance/eagle3-amd/03-figure3.png)

**Figure 3。** MiniMax-M3 EAGLE3 的 acceptance length 随输入长度。虚线 AL=1 是每个验收周期吐一个。

Draft：[amd/MiniMax-M3-EAGLE3.1](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1)，target：[amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4)：

```bash
export VLLM_ROCM_USE_AITER=1
vllm serve amd/MiniMax-M3-MXFP4 --trust-remote-code --tensor-parallel-size 8 \
--block-size 128 --attention-backend TRITON_ATTN --moe-backend emulation \
--speculative-config '{"method":"eagle3","model":"amd/MiniMax-M3-EAGLE3.1","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

## 他们说的端到端栈

- **Target：** day-0 MXFP4/FP8 + ROCm/vLLM。
- **Draft：** EAGLE3 训练（这篇的 M3）、Quark 的 FP8/MXFP4、ROCm/vLLM。
- **胶水：** on-policy 数据、hidden 抽取、serve-eval、导出、投机 serve——都走 vLLM。

## 加速数字

这一节只有 **1K/1K**（ISL=1024，OSL=1024）。加速比 = EAGLE3 TPS / **同一** vLLM 构建和 MML 的 no-spec TPS。MML = `--max-model-len`（prompt + 生成）。随机 prompt 吞吐微基准，不是应用级负载。

### Kimi-K2.5：BF16 和 Quark FP8 draft

硬件：MI355X，TP=4，随机 prompt，`num_prompts=10 × concurrency`，`num_warmups=2 × concurrency`，每格 10 个 seed（算术平均）。

| Path | Docker | MML |
| --- | --- | ---: |
| BF16 | `vllm/vllm-openai-rocm:v0.19.0` | 2248 |
| FP8 | `vllm/vllm-openai-rocm:nightly-fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` | 2304 |

Target：[amd/Kimi-K2.5-MXFP4](https://huggingface.co/amd/Kimi-K2.5-MXFP4)。BF16 draft：[lightseekorg/kimi-k2.5-eagle3](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3)。FP8 draft：[amd/kimi-k2.5-eagle3-fp8](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8)（Quark FP8 工作流；共用 target 的 BF16 LM head）。FP8 draft 走 `RowWiseTorchFP8ScaledMMLinearKernel`（`torch._scaled_mm` / hipBLASLt 行缩放 FP8 GEMM），**不是** AITER 预洗牌 FP8。

![Kimi-K2.5 EAGLE3 throughput](../../../../assets/vllm/blog/performance/eagle3-amd/04-figure4.png)

**Figure 4。** Kimi-K2.5 在 MI355X TP=4、1K/1K 的输出 tok/s/GPU。BF16 **1.69–1.90×**，Quark FP8 **1.76–2.00×**，各自对上自己的 no-spec 基线。相对增益在低并发最大。构建和 MML 不同——不是精度对照实验。

### MiniMax-M2.5 BF16 EAGLE3

镜像：`vllm/vllm-openai-rocm:nightly-4eafc729285e459a5fc96efd6f7b313b155cad48`。Target：[MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)。Draft：[thoughtworks/MiniMax-M2.5-Eagle3](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3)，BF16，`num_speculative_tokens=3`，`draft_tensor_parallel_size=1`。1K/1K 随机，TP=4 + expert parallelism，五个 seed（均值）。同一构建的 no-spec 基线。

![MiniMax-M2.5 EAGLE3 throughput](../../../../assets/vllm/blog/performance/eagle3-amd/05-figure5.png)

**Figure 5。** MiniMax-M2.5 在 MI355X TP=4、1K/1K 的输出 tok/s/GPU。相对增益仍是低并发最大。

Sweep 小结：Kimi-K2.5 **1.69–2.00×**，MiniMax-M2.5 **1.38–1.79×**。

## Summary

Instinct 上的 EAGLE3 投机解码，在保住 target 输出语义的前提下把吞吐抬上去：这篇 1K/1K sweep 里 Kimi-K2.5 **1.69×–2.00×**，MiniMax-M2.5 最高 **1.79×**。能端到端落地，靠三件事叠在一起：（1）Quark 给 target 和部分 draft 的 MXFP4/FP8；（2）以 vLLM 为中心的训练环——on-policy 数据、hidden 抽取、按 served acceptance 选盘；（3）ROCm/vLLM 投机 serving。量化工作流已经在放出的 AMD Quark toolkit 里；Instinct 上的 EAGLE3 draft 训练支持写在 **下一版 AMD Quark**。

## Acknowledgements

感谢 AMD Quark 团队、AMD ROCm 与 vLLM 贡献者、InferenceX 维护者和评审、EAGLE3 研究社区。特别感谢：Chang Liu、Xinjun Niu、Wei Luo、Lin Zhao。

## Additional Resources

- [EAGLE3 项目](https://github.com/SafeAILab/EAGLE)
- [EAGLE3 论文](https://arxiv.org/abs/2503.01840)
- [SPEED-Bench](https://arxiv.org/abs/2604.09557)
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
- [AMD Quark](https://github.com/amd/Quark)
- [vLLM](https://github.com/vllm-project/vllm)
