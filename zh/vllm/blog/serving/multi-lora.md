---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Multi-LoRA 进 MoE：一只 GPU 侍候多套适配器

英文对照：`en/vllm/blog/serving/multi-lora.md`  
原文：https://vllm.ai/blog/2026-02-26-multi-lora  
vLLM ≥0.15.0。GPT-OSS 20B 为贯穿例子。图在原网页。SageMaker / Bedrock 上还有额外调参。

基座冻住，请求只换 LoRA。五家客户各吃 10% 卡，不必五张卡。MoE 每只专家 `gate_up` / `down` 各加 shrink+expand，**每专家每适配器四次** 瘦矩阵 GEMM（r 通常 16–64，比 hidden 小 100–300×）。稠密 Multi-LoRA kernel 不懂 expert routing，所以有 `fused_moe_lora`，网格多一维：激活的适配器。

第一版 TTFT 相对基座差约 **10×**：Triton 把与输入长度相关的量当编译期常量，每个 context length 重编译。`do_not_specialize` 之后内核复用。再加 Split-K、CTA swizzle、`EVEN_K` 去多余 mask、expand 里融进基座加法。开源路径报到 144 OTPS / 135 ms TTFT；AWS 调参配置 171 OTPS / 124 ms TTFT（1600/600、rank 32、8 只适配器并行）。相对 0.11.1rc3：OTPS +454%、TTFT −87%。稠密 Qwen3-32B OTPS 也约 +99%。自定义 Triton 配置见 vLLM LoRA Tuning 文档。数字是他们那条负载，不是你的 SLA。
