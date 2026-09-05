---
source: https://docs.vllm.ai/en/stable/
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# vLLM 文档入口

稳定版：https://docs.vllm.ai/en/stable/  ·  滚动：https://docs.vllm.ai/en/latest/  
英文对照：[en/vllm/getting-started/index.md](../../../en/vllm/getting-started/index.md)

文档首页。原页 logo 不收。不是官方译本。口号：**Easy, fast, and cheap LLM serving for everyone.**

从 Berkeley Sky Computing Lab 长出来，页上写许多机构与公司、**2000+** 贡献者。按身份进门：

- 跑开源模型 → [Quickstart](quickstart.md)
- 写应用 → 官方 User Guide（本库不镜像）
- 改 vLLM → 官方 Developer Guide（本库不镜像）

另点名：[roadmap.vllm.ai](https://roadmap.vllm.ai)、[GitHub releases](https://github.com/vllm-project/vllm/releases)。

## 快在哪里

- 当时宣称的 serving 吞吐
- **PagedAttention** 管 KV（[立项文](../blog/architecture/paged-attention.md)）
- Continuous batching、chunked prefill、prefix caching
- Piecewise 与 full CUDA/HIP graph
- 量化：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO 等
- Attention kernel：FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton
- GEMM/MoE：CUTLASS、TRTLLM-GEN、CuTeDSL
- Speculative decoding：n-gram、suffix、EAGLE、DFlash
- `torch.compile` 生成 kernel 和改图
- Prefill / Decode / Encode 可拆开

## 活在哪里

- Hugging Face 模型
- Parallel sampling、beam search 等解码
- Tensor / pipeline / data / expert / context parallelism
- 流式输出
- Structured outputs（xgrammar / guidance）
- Tool calling 与 reasoning parser
- OpenAI 兼容 API、Anthropic Messages API、gRPC
- 稠密和 MoE 上的 Multi-LoRA
- NVIDIA / AMD GPU，x86 / ARM / PowerPC CPU，外加插件：Google TPU、Intel Gaudi、IBM Spyre、华为昇腾、Rebellions NPU、Apple Silicon、MetaX GPU 等

页上写 **200+** Hugging Face 架构：decoder-only（Llama、Qwen、Gemma），MoE（Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS），hybrid attention / SSM（Mamba、Qwen3.5），多模态（LLaVA、Qwen-VL、Pixtral），embedding/retrieval（E5-Mistral、GTE、ColBERT），reward/classification（Qwen-Math）。完整名单在官方 supported-models 页。

他们还列：PagedAttention 博客、[vLLM 论文（SOSP 2023）](https://arxiv.org/abs/2309.06180)、Anyscale continuous-batching 文、meetup。

## 本库本地顺序

- 安装 / 离线 / `vllm serve`：[quickstart.md](quickstart.md)
- 服务端性能旗标（不是整页 CLI）：[serve.md](serve.md)
- 调优顺序：[optimization.md](../optimization/optimization.md)
- 客户端尺子：[cli.md](../benchmarking/cli.md) · 网格搜 batch：[auto-tune.md](../benchmarking/auto-tune.md)
- 屋里的钟：[/metrics](../metrics/production-metrics.md) · 怎么算：[design-metrics.md](../metrics/design-metrics.md)
- 功能：[APC](../features/prefix-caching.md) · [记账](../features/prefix-caching-design.md) · [投机解码](../features/speculative-decoding.md) · [V1](../features/v1-guide.md)
