---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Multi-LoRA 进 MoE：一只 GPU 侍候多套适配器

英文对照：[en/vllm/blog/serving/multi-lora.md](../../../../en/vllm/blog/serving/multi-lora.md)  
原文：https://vllm.ai/blog/2026-02-26-multi-lora  
2026-02-26。署名 **AWS AI Team**（Danielle Maddix Robinson, Florian Saupe, George Novack, Haipeng Li, Mani Kumar Adari, Xiang Song, Yu Gong）。vLLM ≥**0.15.0**。贯穿例子：[gpt-oss](gpt-oss.md) 20B。SageMaker / Bedrock 上还有额外调参。也发在 AWS Blogs。1600/600、rank 32、8 只适配器——**他们**那条负载，不是你的 SLA。

**原文 TL;DR：**

- 基座冻住，请求只换 LoRA。五家客户各吃 10% 卡 → 一张 GPU，不必五张。
- MoE 家族：GPT-OSS、Qwen3-MoE、DeepSeek、Llama MoE。稠密模也受益（Llama3.3 70B、Qwen3 32B）。
- 第一版 TTFT 相对基座差约 **10×**。`do_not_specialize` 之后 Triton 二进制复用。
- 开源路径：**144 OTPS** / **135 ms TTFT**。AWS 调参：**171 OTPS** / **124 ms TTFT**。相对 0.11.1rc3：OTPS **+454%**，TTFT **−87%**。稠密 Qwen3-32B OTPS 约 **+99%**。

## 为什么要 Multi-LoRA

每只定制模吃不满自己的端点，GPU 就闲着。Multi-LoRA：原权重冻住，塞进小的可训适配器，同一张 GPU 上按请求换适配器。

相对 vLLM 0.15.0，Amazon 侧给 GPT-OSS 20B 再加一截：**19%** OTPS，**8%** 更低 TTFT——在 [SageMaker AI](https://aws.amazon.com/sagemaker/ai/) 或 [Bedrock](https://aws.amazon.com/bedrock/) 上。

## 在 vLLM 里给 MoE 做 multi-LoRA 推理

MoE：专门的专家；路由器把每个 token 送给相关的几只；稀疏——总参只有一截被点亮。每只专家是一个小 FFN，两段：

- `gate_up` 把 hidden（例如 **4096**）扩到 intermediate（例如 **11008**）——有空间拆开、再 gate。
- `down` 压回去。瓶颈：只留有用的特征。

vLLM `fused_moe`：这些投影当 Group GEMM——token 分到哪只专家，就做哪次 GEMM。

LoRA：冻住 `W`（例如 `W_gate_up`）；训 `A`（`h_in × r`）和 `B`（`r × h_out`）；`y = xW + xAB`。Rank `r` 通常 **16–64**。Shrink：`z = xA`（`h_in → r`）。Expand：`z B`（`r → h_out`）。

本地图（原文版权仍归原站；学习对照用）：

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

**Figure 1。** MoE-LoRA：hidden 4096，intermediate 11008，LoRA rank `r = 32`。

每只专家有 `gate_up` 和 `down`。每套适配器给**每个**投影都加 shrink+expand → 每专家每适配器每请求 **四次** LoRA kernel。有一维（`r`）比 hidden / intermediate 小 **100–300×**。方阵 GEMM 最怕瘦矩阵。

另外两道题：(1) 稠密 Multi-LoRA kernel 不懂 expert routing；(2) 复合稀疏——expert routing **再加** 适配器选择。解法：`fused_moe` 里的 `fused_moe_lora`。逻辑同 `fused_moe`，网格多一维：激活的适配器。

## 把 multi-LoRA 推理拧快

Nsight Systems：`fused_moe_lora` 是延迟最高的那截。Nsight Compute：剖 `gate_up_shrink`、`gate_up_expand`、`down_shrink`、`down_expand`。然后执行优化、kernel 优化、调参配置。

### Execution optimizations

第一版 Multi-LoRA TTFT 相对公开的 GPT-OSS 20B 基座差约 **10×**。Triton 把和输入长度相关的量当编译期常量 → 每个新的 context length 都把 `fused_moe_lora` 重编译一遍。

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

**Figure 2。** 修之前：每次 `fused_moe_lora` 前都有 `cuModuleLoadData`（新二进制，不是缓存）；大空隙 = 重编译时空转。这段空转把 TTFT 拉成 10×。修法：`do_not_specialize`——编一次，所有 context length 复用。

### Kernel optimizations

**Split-K。** LoRA shrink 是 `xA`，`x` 为 `1×h_in`，`A` 为 `h_in×r`。`r` 个输出每个要加 `h_in` 次乘。标准 GEMM 按输出并行；每个 thread group 仍把 `h_in` 串着走。Split-K 把 K（`h_in`）切给多个 thread group；部分和用 atomic add 合。纯加、没有额外逻辑 → `sem="relaxed"`。

**CTA swizzling** 用在 `lora_shrink`。`A` 的相邻列共享行 / cache line。调度改成邻近列一起跑 → L2 更肯复用。

**EVEN_K。** Triton 按固定块加载；K 除不尽就要每次 load 都 mask。`EVEN_K` 在 K 能整除 `BLOCK_SIZE_K` 时为真——mask 和多余的 dot 都可以省。

**融合** LoRA 加基座加法进 expand kernel——少一次 launch。

做完：GPT-OSS 20B 开源路径 **144 OTPS** / **135 ms TTFT**。

### Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock

Triton 旋钮：`BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`（cache 局部性）、`SPLIT_K`。标准 fused MoE 的默认值没把 LoRA index 那一维网格和适配器稀疏算进去。用户可以指一个文件夹加载自定义配置；见 vLLM LoRA Tuning 文档。四个 op 一起调（它们共享 `BLOCK_SIZE_M`）。SageMaker / Bedrock 自动加载 → GPT-OSS 20B **171 OTPS** / **124 ms TTFT**。

## Results & conclusion

开源了 GPT-OSS、Qwen3 MoE、DeepSeek、Llama MoE 的 Multi-LoRA。相对 vLLM 0.11.1rc3 → 0.15.0：GPT-OSS 20B 上 OTPS **+454%**，TTFT **−87%**。Kernel 调参 + CTA swizzle 也帮了稠密模：Qwen3 32B OTPS **+99%**。本地：≥0.15.0。Amazon 额外：同一只模上相对 0.15.0 再 **19%** OTPS / **8%** TTFT。托管文档：[SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)、[Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/fine-tuning-openai-apis.html)。

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)

**Figure 3。** GPT-OSS 20B Multi-LoRA 的 OTPS 和 TTFT：(1) 最初的 0.11.1rc3；(2) 0.15.0；(3) 0.15.0 + AWS 自定义 kernel 调参。**1600** 输入 / **600** 输出，rank **32**，**8** 只适配器并行。

也发在 [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/)。

## Acknowledgments

vLLM 社区：Jie Li, Chen Wu, Varun Sundar Rabindranath, Simon Mo, Robert Shaw。AWS：Xin Yang, Sadaf Fardeen, Ashish Khetan, George Karypis。
