---
source: https://vllm.ai/blog/2026-02-26-multi-lora
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Multi-LoRA 进 MoE：一只 GPU 侍候多套适配器

英文对照：[en/vllm/blog/serving/multi-lora.md](../../../../en/vllm/blog/serving/multi-lora.md)  
原文：https://vllm.ai/blog/2026-02-26-multi-lora  
2026-02-26。署名 **AWS AI Team**（Danielle Maddix Robinson, Florian Saupe, George Novack, Haipeng Li, Mani Kumar Adari, Xiang Song, Yu Gong）。学习重写，不是官方译本。vLLM ≥**0.15.0**。贯穿例子：[gpt-oss](gpt-oss.md) 20B。SageMaker AI / Bedrock 上还有额外调参。也发在 AWS Blogs。1600/600、rank 32、8 只适配器——**他们**那条负载，不是你的 SLA。

**原文 TL;DR：**

- 基座冻住，请求只换 LoRA。五家客户各吃 10% 卡 → 一张 GPU，不必五张。
- MoE 家族：GPT-OSS、Qwen3-MoE、DeepSeek、Llama MoE。稠密模也受益（Llama3.3 70B、Qwen3 32B）。
- 第一版 TTFT 相对公开基座差约 **10×**。`do_not_specialize` 之后 Triton 二进制复用。
- 开源路径：**144 OTPS** / **135 ms TTFT**。AWS 调参：**171 OTPS** / **124 ms TTFT**。相对 0.11.1rc3：OTPS **+454%**，TTFT **−87%**。稠密 Qwen3-32B OTPS 约 **+99%**。

原文分节：Implementing multi-LoRA inference for MoE models in vLLM → Improving multi-LoRA inference performance in vLLM（Execution optimizations / Kernel optimizations / Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock）→ Results & Conclusion → Acknowledgments。

组织和个人同时跑多套定制模——尤其是新近的 Mixture of Experts（MoE）家族——会碰上同一件事：每只模吃不满自己的端点，GPU 却按端点付钱。AWS 和 vLLM 社区一起做的解法是 Multi-Low-Rank Adaptation（Multi-LoRA）serving，先落在 GPT-OSS、Qwen 这类开源 MoE 上。

Multi-LoRA 是常见的微调路：不重训整份权重，原权重冻住，往层里塞小的可训适配器。推理时，多套定制模共享同一张 GPU，按请求把适配器换进换出。五个客户各只用掉专用 GPU 的 10%，就可以挤到一张卡上——五张闲卡变成一张忙卡。

下文先讲 vLLM 里 MoE 的 multi-LoRA 推理怎么接，再讲 kernel 级优化，最后说你怎么用上。贯穿例子是 GPT-OSS 20B。

本地部署用 **0.15.0** 或更新即可。Multi-LoRA serving 现在覆盖 GPT-OSS、Qwen3-MoE、DeepSeek、Llama MoE。同一套优化也帮了稠密模，例如 Llama3.3 70B、Qwen3 32B。Amazon 侧相对 vLLM 0.15.0 还有额外延迟改进：GPT-OSS 20B 上 OTPS（Output Tokens Per Second，模型往外吐字有多快）高 **19%**，TTFT（Time To First Token，等到第一个非空 token 要等多久）低 **8%**。要吃到这截，把 LoRA 定制模放到 [Amazon SageMaker AI](https://aws.amazon.com/sagemaker/ai/) 或 [Amazon Bedrock](https://aws.amazon.com/bedrock/) 上。

## Implementing multi-LoRA inference for MoE models in vLLM

先把 MoE 和 LoRA 的账摊开，后面那些 kernel 选择才说得通。

MoE 里有多只专门的神经网络，叫专家。路由器把每个输入 token 送给最相关的几只，输出再聚合。稀疏：每个 token 只点亮总参的一截，所以能用更少的算力撑更大的模。Figure 1 是这张图。

每只专家是一个小的 feed-forward network，分两段处理 token 的 hidden state。第一段，`gate_up` 投影把紧凑 hidden（例如 **4096** 维）扩到更大的 intermediate（例如 **11008** 维）。紧凑空间里特征缠在一起，大空间才有地方把它们拆开、变换、再 gate 哪些真正有用。第二段，`down` 投影压回原来的维度：输出才能跟模型其余部分对接，也是瓶颈——只准留下有用的特征。扩完再压，每只专家能做比较丰富的变换，输出尺寸却不变。

vLLM 用 `fused_moe` kernel 把这些投影当成 Group GEMM（Group General Matrix Multiply）来跑——某个 token 分到哪只专家，就做哪一次 GEMM。

Multi-LoRA 微调冻住基座 `W`（例如 `gate_up` 的 `W_gate_up`），另训两块小矩阵 `A` 和 `B` 组成适配器。投影基座形状是 `h_in × h_out`，LoRA 训 `A`（`h_in × r`）和 `B`（`r × h_out`），rank `r` 通常 **16–64**。微调后的输出是 `y = xW + xAB`。每个适配器给一次投影加两步：shrink 算 `z = xA`，把输入从 `h_in` 收到 `r`；expand 再拿这份 `r` 维结果乘 `B`，投回 `h_out`。Figure 1 右侧就是这件事。

本地图（原文版权仍归原站；学习对照用）：

![moe schematic](../../../../assets/vllm/blog/serving/multi-lora/01-moe_schematic.png)

**Figure 1。** MoE-LoRA 怎么工作：hidden 4096，intermediate 11008，LoRA rank `r = 32`。

每只专家有两次权重投影：`gate_up` 和 `down`。一套 LoRA 给**每个**投影都加 shrink + expand。于是每只专家一共要四次 LoRA kernel：`gate_up` 的 shrink/expand，`down` 的 shrink/expand。Multi-LoRA serving 里，多套适配器同时伺候不同用户或任务，系统必须把「每专家、每适配器、每请求」这四次算子管住——这就是 MoE 上的性能瓶颈。

这四次运算的矩阵有一维（LoRA rank `r`）比另一维（hidden / intermediate）小 **100–300×**。标准 GEMM 为大致方阵设计，瘦矩阵上成绩难看，所以后文那些 kernel 优化才必要。

除了瘦矩阵，给 MoE 加 multi-LoRA 还有两道题。第一，vLLM 当时没有在 MoE 层上做 LoRA 的 kernel：现成的稠密 Multi-LoRA kernel 不懂 expert routing。第二，MoE LoRA 叠了两层稀疏：expert routing（token 分到不同专家）再加 adapter selection（请求用不同适配器）。这种复合稀疏要专门的 kernel。

解法是 `fused_moe_lora`：把 LoRA 算子嵌进 `fused_moe`。它给 `gate_up` 和 `down` 做 LoRA shrink / expand GEMM。逻辑跟 `fused_moe` 一样，网格多一维：当时激活的 LoRA 适配器。

## Improving multi-LoRA inference performance in vLLM

初版落地之后，用 NVIDIA Nsight Systems（Nsys）找瓶颈，最高延迟落在 `fused_moe_lora`。再用 NVIDIA Nsight Compute（NCU）剖这四个 op 的算力和显存吞吐：`gate_up_shrink`、`gate_up_expand`、`down_shrink`、`down_expand`。后面三刀就是执行优化、kernel 优化、以及这四个 kernel 的调参配置。

### Execution optimizations

初版 Multi-LoRA 的 TTFT 比基座（公开的 GPT-OSS 20B）高（更差）约 **10×**。剖下来：Triton 编译器把和输入长度相关的量当编译期常量，于是每个新的 context length 都把 `fused_moe_lora` 从零重编译一遍，而不是复用。Figure 2 里看得到：每次 `fused_moe_lora` 执行前都有 `cuModuleLoadData`——GPU 在装一份新编译的二进制，不是缓存里那份；kernel 启动之间的大空隙是重编译时空转。这段空转把相对基座的 TTFT 拉成 10×。修法：给这些变量加 `do_not_specialize` 编译提示，让 Triton 编一次、所有 context length 复用。

![exec opt](../../../../assets/vllm/blog/serving/multi-lora/02-exec_opt.png)

**Figure 2。** 执行优化之前，`fused_moe_lora` 的 profiling。

### Kernel optimizations

**Split-K** 是一种工作切分：瘦矩阵上把负载摊匀。LoRA shrink 算 `xA`，`x` 是 `1×h_in`，`A` 是 `h_in×r`。`r` 个输出每个都要加 `h_in` 次乘。标准 GEMM 把不同 thread group——共享片上快存的一批 GPU 线程——分给不同输出元素，但每个 thread group 仍把那 `h_in` 次求和串着走。`r` 只有几十、`h_in` 有几千：能并行的输出很少，每次求和却很长。Split-K 把 GEMM 内维 `K`（这里 `K = h_in`）切给多个 thread group，各自算部分和，再合并。部分结果要用 atomic add 合成最终和。这里是纯加、没有额外逻辑，于是把 atomic add 的 `sem="relaxed"` 交给 Triton，给编译器留优化空间。

GPU 调度器会把多个 thread group 派给同一个输出元素，同时让不同输出的 thread group 一起跑。对 `lora_shrink`，每个输出要读 `A` 的一列，这一列跨过 `h_in` 行。`h_in` 上千时，每列碰到的 cache line 铺开一大片。邻近列共享同一批行、cache 有重叠，所以打邻近列的 thread group 能复用彼此刚载入的数据。Cooperative Thread Array（CTA）swizzling 改调度顺序：打邻近列的 thread group 同时跑，L2 更肯复用。他们把 CTA swizzling 用在 `lora_shrink` 上。

shrink 和 expand 的 LoRA kernel 还去掉了不必要的 mask 和 dot。Triton 按固定块加载，矩阵维数不一定整除块大小。例如 `BLOCK_SIZE_K` 是 64、K 是 100，第二块会去读 28 个非法地址。Mask 在每次 load 前检查下标是否越界。可这些条件判断每次 load 都跑，即便元素合法也有开销。他们加了 `EVEN_K`：K 能整除 `BLOCK_SIZE_K` 时为真——load 全部合法，mask 整段可省，多余的 dot 也少做。

最后，把 LoRA 权重加回基座权重这件事融进 LoRA expand kernel，少一次 launch。这几刀做完，GPT-OSS 20B 到 **144 OTPS** / **135 ms TTFT**。

### Tuning kernel configurations for Amazon SageMaker AI and Amazon Bedrock

Triton kernel 要拧块大小：`BLOCK_SIZE_M`、`BLOCK_SIZE_N`、`BLOCK_SIZE_K`，决定矩阵计算怎么切给 thread group。更高级的有 `GROUP_SIZE_M`（thread group 排序、管 cache 局部性）和 `SPLIT_K`（沿内维把求和并行化）。

他们发现：MoE LoRA kernel 若沿用标准 fused MoE 的默认配置，在 multi-LoRA serving 上成绩差。那些默认值没把 LoRA index 那一维网格、以及多适配器带来的复合稀疏算进去。于是加了用户可指定文件夹、加载自定义调参配置的路径；细节见 vLLM LoRA Tuning 文档。四个 `fused_moe_lora` op（`gate_up_shrink`、`gate_up_expand`、`down_shrink`、`down_expand`）一起调，因为它们共享 `BLOCK_SIZE_M`。SageMaker AI 和 Bedrock 客户会自动加载这套配置，GPT-OSS 20B 到 **171 OTPS** / **124 ms TTFT**。

## Results & Conclusion

和 vLLM 社区合作之后，他们实现并开源了 GPT-OSS、Qwen3 MoE、DeepSeek、Llama MoE 的 multi-LoRA serving。优化落在 0.15.0 相对 0.11.1rc3：GPT-OSS 20B 上 OTPS **+454%**，TTFT **−87%**。有的优化——尤其 kernel 调参和 CTA swizzling——也帮了稠密模：Qwen3 32B 的 OTPS **+99%**。本地部署用 0.15.0 或更新。Amazon Bedrock 和 SageMaker AI 上的 Amazon 额外优化，再相对 0.15.0 给跨模型的延迟：GPT-OSS 20B 上 OTPS **19%**、TTFT **8%**。起步文档：[Amazon SageMaker AI hosting](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)、[Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/fine-tuning-openai-apis.html)。

![otps](../../../../assets/vllm/blog/serving/multi-lora/03-otps.png)

**Figure 3。** GPT-OSS 20B multi-LoRA 的 OTPS 和 TTFT：(1) 最初落在 0.11.1rc3 的实现；(2) vLLM 0.15.0；(3) 0.15.0 + AWS 自定义 kernel 调参。实验：**1600** 输入 token / **600** 输出 token，LoRA rank **32**，**8** 只适配器并行加载。

也发在 [AWS Blogs](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/)。

## Acknowledgments

vLLM 社区：Jie Li, Chen Wu, Varun Sundar Rabindranath, Simon Mo, Robert Shaw。AWS：Xin Yang, Sadaf Fardeen, Ashish Khetan, George Karypis。
