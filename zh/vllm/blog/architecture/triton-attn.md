---
source: https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Triton Attention：一份源码伺候三家卡

英文对照：`en/vllm/blog/architecture/triton-attn.md`  
原文：https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive  
2026-03-04。IBM Research + Red Hat + AMD 的 office hours。kernel：`vllm/v1/attention/ops/triton_unified_attention.py`（大约 800 行；对照 FlashAttention 3 大约 7 万行）。论文：*The Anatomy of a Triton Attention Kernel*。图在原网页。

Attention backend 把注意力从线性层、RMSNorm 后面隔开。CUDA 上有 FlashAttention / FlashInfer，ROCm 有自己的，MLA 还有专用。Triton 后端整份用 Triton 写、跟着 vLLM 走、只依赖 PyTorch+Triton，**总能当 fallback**。AMD ROCm 上默认；Intel XPU 的 float32（FlashAttention 不支持 fp32）走它；ALiBi sqrt、sink token、GPT-OSS、小 head、encoder/decoder、多模态 prefix、batch invariance 也常落到这里。A100 一类 pre-Hopper 同样用得上。

## 为什么是 Triton

为每家 GPU、每种 batch/长度再手写一份 kernel 养不起。Triton 用 tile 表达计算，编译器和 autotuner 再映射到硬件。同一份逻辑，tile 形状可以完全不同。

Paged attention：对 batch 里每个 query token、每个头，穿过分页 KV 算分、乘 V。`tl.dot` 要够大的 tile 才吃得饱；KV 侧 tile 被 page size 卡住，所以从 **Q block** 下手——GQA 把同一 KV head 下的 query head 捆在一起，再把多个 query token 收进一个 work item。Decode 只有一个 query token，Q block 帮不上，于是 **3D kernel**：把 KV 遍历拆到多个实例，再第二发 kernel 做 tiled softmax 归约（Triton 没有全局 barrier）。启发规则决定何时值得付第二次 launch。

CUDA graph 不喜欢随 batch/长度变的 launch grid。线程多于 SM 就分波，第二波往往吃不饱；graph 会把这份浪费原样重放。后来改成 **persistent kernel**：launch 数钉死在计算资源上，实例从 GPU 内存读 metadata 决定干多少——grid 恒定，graph 才能复用。（文中 PR 当时还 pending。）

## 成绩（2025 年末）

Llama 3.1 8B、batch 1、输入 500。H100 上长 decode 达到 FlashAttention 3 的 **100.7%**；MI300 相对更早实现大约 **5.8×**。**同一份源码。** 预览：Helion（更高一层的 Triton / tiled PyTorch）上有简化版实验。

`optimization.md` 说 attention backend 按架构自动选，也可手指定。指定之前先知道：Triton 不是「慢的便携」，在 FA3 缺席或移植成本太高的地方，它就是默认那条路。
