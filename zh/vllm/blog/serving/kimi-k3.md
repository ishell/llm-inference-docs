---
source: https://vllm.ai/blog/2026-07-27-k3
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Kimi K3 day-0：2.8T hybrid MoE，KDA prefix cache 和 DSpark 一起转

英文对照：[en/vllm/blog/serving/kimi-k3.md](../../../../en/vllm/blog/serving/kimi-k3.md)  
原文：https://vllm.ai/blog/2026-07-27-k3  
2026-07-27。vLLM Team and Inferact。权重 [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3)。DSpark [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark)。菜谱 [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3)。预告 [kimi-k3-preview.md](kimi-k3-preview.md)。模型文 [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)。FlashKDA、Flash-Flash-KDA。因复杂依赖，**当时只有 Docker 能用**；镜像含若干预发布依赖，包括 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。跳过社交预览图和页上 GIF/MP4。本地图版权仍归原站。

上周 [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) 讲生产级集成；今天权重公开，支持上线。最兴奋的挑战：让 KDA、MXFP4 MoE、KV cache、P/D 拆分、speculative decoding、长上下文 recipes 在能跑的 serving 引擎里一起转。Preview 讲 kernel 和 cache，尤其 recurrent state 上的 prefix caching。这篇是实用指南：vLLM 怎么适配架构、数字背后的 kernel、day 0 什么能用。

## Quick start

```bash
# See the linked recipes for the exact Docker command.
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

最省事：8 张 NVIDIA B300 或 8 张 AMD MI355X，上面这条。

Inferact 还训并开源了 [DSpark speculator](https://huggingface.co/Inferact/Kimi-K3-DSpark)。serve 命令加：

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

更多平台 Docker 和部署策略见 [recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3)。复杂依赖，**当时只有 Docker 能用**。

## TL;DR

- **2.8T 多模态 MoE：** 每 token 激活 16/896；上下文到 1M；KDA、AttnRes、LatentMoE、原生 MXFP4。
- **每用户最高 370 tok/s：** 无 speculative 118 tok/s；DSpark 370 tok/s（**3.14×**），16× NVIDIA GB300 NVL72。对着 K3 架构做了大量优化。
- **生产特性：** speculative decoding、P/D 拆分、Mooncake agentic KV、tool calling、reasoning、structured output。Launch 覆盖 NVIDIA Hopper/Blackwell 和 AMD MI355X。
- **开源 DSpark：** block-diffusion speculative decoding；用 vLLM + TorchSpec 训，Inferact 开源。
- **Hybrid prefix caching：** recurrent + full-attention 逼着重设计 hybrid prefix cache。现在每个类似的 hybrid linear 模型都受益。

## Kimi K3 的架构，vLLM 怎么 serve

![Kimi K3 architecture](../../../../assets/vllm/blog/serving/kimi-k3/01-architecture.png)

_架构创新来自 [Moonshot 原文](https://www.kimi.com/blog/kimi-k3)。_

K3 在几处离开标准 Transformer，每一处都改 serving 引擎要做的事。Preview 写内部；这里复述新点，盯 vLLM 怎么适配。

### Kimi Delta Attention: hybrid recurrent + full-attention

**新在哪：** 多数层是 KDA——线性注意力，固定大小 recurrent state，而不是增长的 KV；周期性 full-attention 层保住精确全局回忆。1M 上下文才付得起。

**vLLM 怎么 serve：** 一个 hybrid KV-cache manager 在同一 scheduler 下并排两份内存：full-attention 的 paged KV，KDA 的紧凑 recurrent-state blocks。专用 KDA attention backend：prefill 走 FlashKDA；decode 走 fused CUDA（speculative decoding 时走 Flash-Linear-Attention / Triton）。

最难的是 hybrid cache 上的 prefix caching：full-attention 存 per-token KV；KDA 每 token 更新 recurrent 和 convolution state，却付不起在每个可能前缀边界留快照。vLLM 把大块物理 KDA state 和细粒度 prefix matching 拆开，在块内登记快照，扩展前先拷，长共享 prompt 才能同时复用 KDA state 和 paged KV。这套 hybrid-cache 是 [vLLM core 新基础设施](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)，现在每个类似 K3 的 hybrid 模型都受益。笔记 [kimi-k3-preview.md](kimi-k3-preview.md)。

![hybrid KDA and full-attention cache](../../../../assets/vllm/blog/serving/kimi-k3/02-hybrid-cache.png)

_K3 把 KDA 和周期性 full-attention 交错；vLLM 的 hybrid cache 一起管 recurrent state 和 paged KV。_

### Attention Residuals: 跨深度学着混 residual

**新在哪：** 每个 token，Block AttnRes 用深度方向的 attention 替换普通 residual 累加：每个 Transformer sublayer 用学到的 pseudo-query，给前面 layer blocks 的 RMS-normalized residual states 加权，再把对应加权组合当输入。

**vLLM 怎么 serve：** 优化过的 Triton / CUDA kernel，把深度方向的 attention logits、softmax、hidden-state aggregation 收进一次 fused 操作。支持处把 residual updates 和 output RMSNorm 折进同一 kernel，prefill 和 decode 都少中间流量和 launch。

### Stable LatentMoE: 16/896，quantile-balanced latent experts

**新在哪。** NVIDIA 的 [LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/)：把 dispatch 出去的 token activations 投到更窄的 latent 维做 routed-expert 计算，再投回模型宽度——减 expert-weight 带宽和 all-to-all，才能在相近推理成本下用更多 experts。K3 的 [Stable LatentMoE](https://www.kimi.com/blog/kimi-k3) 扩到 896 experts、每 token 16 active，用 [Quantile Balancing](https://kexue.fm/archives/11619) 从 router-score quantiles 推导 expert 分配，而不是启发式 balancing 更新。

**vLLM 怎么 serve：** Experts 用 expert parallelism 切。两套 MoE backend：TRT-LLM-Gen 对着 TP > 1；MegaMoE 对着拆分 / DEP。可选 Expert-Parallel Load Balancing (EPLB)，让各 rank 算力接近。权重在 MoE 路径上原生 MXFP4。

### Chat template: 渲染程序，不是 Jinja

**新在哪：** K3 的 chat template 要用精确 control tokens 编 system / user / assistant、多模态、tool 定义、tool 结果。常见做法是 [Jinja](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/main/chat_template.jinja) 先渲成文本再 tokenize；K3 用 **Python 程序直接建 prompt token 序列**。输出里 reasoning、answer、tool calls 分区，必须解析进 API。

**vLLM 怎么 serve：** Python 和 Rust 前端都实现 input renderer 和 streaming output parser，保住 control-token 边界，把用户和工具给的文本当普通内容。Tool calls 和 structured outputs 把 K3 格式接到 [XGrammar](https://xgrammar.mlc.ai/)，解码时约束 structured regions，再拆成 reasoning / content / tool-call 字段。K2 的 Jinja 握手坑：[kimi-k2-accuracy.md](kimi-k2-accuracy.md)。

## Built for production

Serve 好 2.8T hybrid MoE：单用户要快，多会话要省，agent 要能扩。vLLM 三头都按 day 0 准备。

### Ultra-low latency: DSpark

2.8T 上要超低延迟又不丢精度，speculative decoding 是自然选择。所以 day 0 就支持 DSpark，并训、放了 [DSpark speculator](https://huggingface.co/Inferact/Kimi-K3-DSpark)。Draft 用 vLLM + [TorchSpec](https://github.com/lightseekorg/TorchSpec) 训，speculator 推理和训练数值对齐。笔记 [dspark-adaptive.md](../features/dspark-adaptive.md)。

DSpark 用 block-diffusion backbone，基于 K3 丰富的中间状态，一次并行出多个 speculative tokens，draft 成本随 block 加深保持平坦。Low-rank Markov head 提供块内依赖；confidence head 预测每条 draft 被接受的可能。Draft 做成 MLA-native，镜像 K3 自己的 attention，draft 和 target 共享相近 KV layout，好兼容高级 KV 管理和 P/D 拆分。

![DSpark positional acceptance](../../../../assets/vllm/blog/serving/kimi-k3/03-dspark-acceptance-rates.png)

_各数据集上的 positional acceptance。_

DSpark：单用户 **3.14×**，118 → 370 tok/s，SPEED Bench。编码和低熵任务大约 **4.73** accepted tokens / step；创意写作等高熵大约 **2.61**。

Confidence-based scheduling 是当时进行中的活。打开后用 DSpark 自带的 confidence head 预测每条 drafted token 被接受的可能，优先强提案、剪弱的，verification 不花在活不下来的 token 上。

Draft 和推理支持随这篇开源。部署见下。

![DSpark draft-and-verify](../../../../assets/vllm/blog/serving/kimi-k3/04-dspark-schematic.png)

_轻量 DSpark draft 提候选，K3 一次并行 verify，加速单流 decode。_

### Sequence parallelism for TEP prefill

![sequence parallelism](../../../../assets/vllm/blog/serving/kimi-k3/05-sequence-parallelism.jpg)

_Sequence parallelism 按 rank 切 token 所有权；attention residual 按 shard 施；一次 all-gather 在下一层 QKV 前重建整 batch。_

Prefill 把 attention tensor parallelism 和 MoE expert parallelism 合在一起（TEP）。相对纯 TP，TEP 减通信、整专家留在各 rank，expert GEMM 形状更有效。

朴素 TEP 每层两次 all-reduce——一次在 attention output projection 后，一次在 MoE 后——每个 rank 都物化整 batch，并对整份冗余施 attention residual。于是做 [sequence parallelism](https://arxiv.org/abs/2205.05198)：`o_proj` 后的 all-reduce 换成 reduce-scatter，每 rank 拥有一 shard tokens；attention residual 按 shard 施；MoE 的 all-to-all 做 dispatch / combine；一次 all-gather 在下一层 QKV 前恢复整 batch。

两个关键好处：

- **减通信：** Reduce-scatter + all-to-all dispatch + combine + all-gather，理论上比两次 all-reduce 便宜。实践里 NCCL 的 reduce-scatter / all-gather 没对着 prefill 消息尺寸优化，**没有加速**。于是自己写 custom reduce-scatter / all-gather，比 NCCL 快 **1.7×–4.5×**，小到中等消息尤其明显。
- **Sharded attention residual：** Residual 整层保持按 rank 切，每 rank 只算、只维护自己那份 tokens。对 K3 尤其要紧：AttnRes 把 residual stream 变成带自己计算和内存脚印的持久跨层状态。

合适时默认开：TP + MegaMoE，或 TP + DP + EP。**不用额外 flag。**

### Large-scale serving: P/D disaggregation

高吞吐：跨节点 EP / DP，P/D 拆分——prefill-heavy 和 decode-heavy 分 replica，各自按瓶颈 sizing。校验过的拓扑之一：TEP8 prefill → DEP16 decode，NIXL 当 KV transfer。笔记 [mooncake.md](../features/mooncake.md)。

Hybrid 模型上 P/D 不留情：recurrent KDA state、full-attention paged KV、block tables 都要到得正确。NIXL connector 把共享 KV-cache page 看成两份逻辑视图：token-level MLA cache，和 request-level KDA state（含 convolution 和 recurrent）。Handshake 交换 MLA/KDA metadata，再为每次 transfer 建分开的 descriptors。

异构 TP 下，hybrid allocator 给 prefill 和 decode 用不同 block size。NIXL connector 跟踪 logical-to-physical block mapping，把没传过去的尾巴 **清零**，免得上一条请求的陈数据从 padding 或 layout 缝里漏出来。

页上 P/D GIF 未收录。

### Reconciling partial block cache hits and KV cache offloading

Preview 里写过：细粒度 prefix hit 可以停在物理 cache block **里面**。给 KV offload 带来细问题：vLLM 可能先在本地 GPU 打中带 partial tail 的 hit，再在 Mooncake 这类外部 store 发现更长的前缀。Full-block hit 时，远程复用可以干净地接到本地前缀之后。Partial tail 却可能和远程结果 **重叠**。

Scheduler 因此比较两层的 **精确** 可复用 token 长度，选更长的前缀。远程赢了，就释放为更短本地尾巴预留的 block，把所有 cache groups 对齐到新前缀长度。

整套机制完全走现有 KV Connector APIs——语义已经够。`MooncakeStoreConnector`、`SimpleCPUOffloadConnector` 和其他 connectors 都能做多层 partial-prefix reuse，不必模型专用路径。笔记 [kv-offload.md](../features/kv-offload.md)。

设计：[RFC #45702](https://github.com/vllm-project/vllm/issues/45702)；实现 [PR #45939](https://github.com/vllm-project/vllm/pull/45939)、[#46384](https://github.com/vllm-project/vllm/pull/46384)、[#49502](https://github.com/vllm-project/vllm/pull/49502)。

### Agentic serving: 更聪明的 cache retention

K3 的线性注意力层只要常量大小的 KDA state，长上下文省内存。单层 KDA state 大约等于几千 token 的 MLA cache。虽大，却不随序列增长——和常规 KV 不同。Agent 负载跨十几万到一百万 token 时，这个区别变大。

同一设计也把 prefix caching 变复杂。KDA state 在 decode 时原地更新，vLLM 必须在选中的前缀边界 **拷** 一份，下一轮 forward 才会覆写。每个 token 位置都 cache 贵得离谱：一份 KDA checkpoint 比一个 token 的 MLA cache 大得多，分布式 cache pool 也会很快耗尽。

为了提高 cache 空间效率又保住有用前缀，vLLM 支持两套互补 retention。

#### Interval-based retention

每个 KDA state 都 cache 浪费；太稀又逼下一请求重算大段后缀。Interval-based 把选中位置当 checkpoints——例如每 32K tokens 一份。

Prompt 边界是更好的 checkpoints。Agent 下一轮通常先回放上一轮 prompt，prompt 末尾的 state 特别容易被复用。vLLM 自动检测并保留这些边界。

周期 checkpoint：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`。设 `0` 关掉周期 checkpoint，只留 prompt-end——多轮对话主导的负载合适。更大 interval 用一点重算换更低 cache。

Interval-based 先为 DeepSeek V4 和 hybrid sliding-window 模型进 [PR #43447](https://github.com/vllm-project/vllm/pull/43447)；K3 和 hybrid linear-attention 的 day-0 在 [PR #45845](https://github.com/vllm-project/vllm/pull/45845)。

![interval-based KDA retention](../../../../assets/vllm/blog/serving/kimi-k3/07-interval-cache-retention.png)

_MLA 每块都 cache KV；KDA state 只在 checkpoints 留：prompt ends（绿）总留，固定 interval（橙）可配。_

#### Marconi-style selective retention

Prompt-end 对会话状态好，但有价值的共享前缀可以出现在别处。System prompt、仓库快照、tool spec 可能被许多请求复用，却对不齐 prompt 边界。

[Marconi-style（MLSys '25）](https://mlsys.org/virtual/2025/poster/3260) 规则简单：**第二次 hit 才 cache。** 第一次证明前缀存在；第二次证明它真被共享。这时 vLLM 才把 cache 容量花在它的 KDA state 上。

Retention 变成按需决策。一次性前缀不挤 cache；反复出现的自动晋升——用户不必预先猜哪些会热。

Selective：[PR #37898](https://github.com/vllm-project/vllm/pull/37898)；K3 day-0 [PR #47782](https://github.com/vllm-project/vllm/pull/47782)。页上 selective GIF 未收录：Request 1 只在自己的 prompt end 留 KDA（过了共享前缀），Request 2 得 KV hit、KDA miss；第二次目击才在前缀边界 cache，Request 3 复用。

两套政策合起来覆盖可预期和涌现的复用：interval 钉结构上重要的边界；Marconi 学哪些别的前缀值得留。

## Performance optimizations

K3 这种体量自带难题。整模型勉强塞进单台 NVIDIA DGX B300；那一代硬件最少要 **16× B200/GB200**。Serving 要在交互性和系统总吞吐之间权衡：TP 对交互性好，但有效 KV 小、总吞吐低；大规模 EP 会因网络带宽卡每用户输出速度。下面这些优化两头都抬，用户按负载选 recipe。许多已在 [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) 写过。

### Attention Residuals

Block AttnRes 最多 attend **八份** 缓存的 block 表示，外加当前块内 residual。每个 token：从 RMS-normalized sources 算 logits，在这些深度方向候选上 softmax，再聚合表示。实现像 FlashAttention 的 online-softmax，但跨的是 **模型深度** 而不是序列位置，最多九个 sources。一次 fused kernel 做混合，输入侧收 residual update，输出可选 RMSNorm。可移植 Triton 走通用路径；专用 CUDA 加速支持的 Blackwell 配置。

### KDA decode

![fused KDA decode](../../../../assets/vllm/blog/serving/kimi-k3/09-kda-decode.png)

_Fused KDA decode 把 causal convolution、recurrent update、RMSNorm 收进一次 launch。_

一层 KDA 操作很多：input projections、causal 1D convolutions、QK norm、gate、KDA recurrent update、output gated RMSNorm。支持的配置上，vLLM 把 post-projection decode——从 causal convolutions 到 gated RMSNorm——融进一个专用 CUDA kernel。Kernel 原地更新 convolution 和 recurrent states，直接写归一化输出，避开中间张量、反复的 state 流量、以及 K3 许多 KDA 层上的 per-operation launch。不支持的配置走可移植 Triton fallback。

### KDA prefill

KDA prefill 成了开源开发的爱例。Moonshot 先放 [FlashKDA](https://github.com/MoonshotAI/FlashKDA)，高性能 CUTLASS。很快接到 vLLM，再啃不那么光鲜的生产细节：更宽 GPU 覆盖、metadata dtypes、tensor layouts、可靠 vendoring。[Shikhar Mishra](https://github.com/Itssshikhar) 再为 H100 优化，发 [Flash-Flash-KDA](https://github.com/Itssshikhar/Flash-Flash-KDA)，改善数据搬移、保住数值正确。一天内在 GB300 NVL72 上校验，收紧 recurrence pipeline 和同步，折进 FlashKDA 集成。不是单向交接：开源 kernel 被 serving 社区扩、独立贡献者改进、很快进生产。

### KDA metadata builder

![KDA metadata before/after](../../../../assets/vllm/blog/serving/kimi-k3/10-kda-metadata-builder.png)

DSpark bring-up 时，KDA metadata 准备成了显著开销。K3 起初复用通用 GDN metadata builder：准备 K3 并不消费的 FLA metadata，再用一串小 eager PyTorch ops 组装、staging GPU metadata。专用 Kimi K3 KDA metadata builder 剪掉不用的路径，把那些序列换成 fused Triton，每段收成一次 launch。Batch size 1：metadata-preparation **96%**，**870 µs → 34 µs**；端到端 DSpark 延迟 **−6%**。

### Low-latency BF16 GEMM

低 batch、延迟敏感时，若干 linear projection 的通用 BF16 GEMM 换成自己的 `skinnyGEMM`。Generic cuBLAS 对着更一般的形状，这里不是最好。Kernel 绕过 shared-memory staging，activations 和 weights 直接进寄存器，用 CUDA Core FMA 做数学。避开为最大吞吐准备的沉重 TMA / Tensor Core setup。Microbenchmarks：kernel 级 **8%–100%**；小 batch 端到端大约 **−10%**。

### Low-latency MoE tail fusion

![LatentMoE tail fusion](../../../../assets/vllm/blog/serving/kimi-k3/11-latent-moe-tail-fusion.png)

_LatentMoE tail：两次 all-reduce、RMSNorm、latent up-projection、elementwise add，换成三个 kernels，减计算、更好重叠通信和计算。_

超低延迟 serving 里，vLLM 用一套新策略压 latent-MoE tail。LatentMoE 末尾，routed experts 收完的 activation 要 RMSNorm、up-project，再加到 shared-expert 输出上。普通 TP：routed 和 shared 两次 all-reduce——或一次 all-reduce + concat——并复制 up-projection。

为避免复制线性投影上的冗余计算：shared experts 走 reduce-scatter；routed experts 仍 all-reduce，因为它们的 activations 要归一化。复制的 routed-expert activation 再按 column-parallel 和 up-projection 做 matmul，elementwise 加到已经 sharded 的 shared-expert 输出上。最后用 broadcast all-gather 到各 rank。这一步大约 **−20%** 延迟；端到端大约 **7%–8%**。

## Quality and Performance Benchmarks

### Accuracy and correctness

vLLM 把精度和速度同等认真。经 OpenAI 兼容 endpoint 端到端校验 K3，精确配置在 recipes，accuracy 干净通过。最大 reasoning-effort：GSM8K **0.976**，GPQA-Diamond **0.939**，OCRBench **0.889**，MMMU Pro Vision **0.818**。

评测 caveat：K3 答之前想很多。低分更常是截断，不是答错。先加大 reasoning effort，把 `max_tokens` 留宽，检查 cut-off，再去 debug 别的。

### Serving performance

![single-user decode](../../../../assets/vllm/blog/serving/kimi-k3/12-serving-performance.png)

_Batch size 1 decode，GB300 NVL72，TP8 / TP16。_

Launch：无 speculative，TP8 **111** tok/s/user，TP16 **118**。DSpark 大约 **3×** 交互性：TP8 **331**，TP16 **370**。

![GB300 NVL72 Pareto](../../../../assets/vllm/blog/serving/kimi-k3/13-pareto-gb300.png)

GB300 NVL72 上的初始 Pareto：从高吞吐 2K+ TPGS 到低延迟 100+ TPS/user。

### Reproduce our benchmark

上面 TP8 + DSpark decode 吞吐的完整 recipes：

```bash
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HEAD_ADDR=127.0.0.1  # Change if vllm-bench runs on another host.

vllm serve moonshotai/Kimi-K3 \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --moe-backend auto \
  --trust-remote-code \
  --load-format fastsafetensors \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-cudagraph-capture-size 256 \
  --kv-cache-dtype fp8 \
  --attention-config '{"mla_prefill_backend":"FLASHINFER","use_prefill_query_quantization":true}' \
  --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'

# Batch size = 1, 8K/1K random (no speculative decoding)
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 0.8 \
  --prompt-token-ids \
  --ignore-eos \
  --sweep-max-concurrency 1 \
  --sweep-num-prompts-factor 10 \
  --seed 42 \
  --percentile-metrics "ttft,tpot,itl,e2el" \
  --metric-percentiles "50,90,99" \
  --save-result

# Batch size = 1, SPEED Bench (speculative decoding)
vllm-bench \
  --backend openai \
  --base-url "http://${HEAD_ADDR}:8000" \
  --model moonshotai/Kimi-K3 \
  --dataset-name speed-bench \
  --speed-bench-config throughput_16k \
  --speed-bench-max-input-len 10240 \
  --speed-bench-category low_entropy \
  --output-len 1536 \
  --num-prompts 10 \
  --no-oversample \
  --max-concurrency 1 \
  --temperature 1.0 \
  --top-p 0.95 \
  --save-result \
  --save-detailed
```

完整 recipes（多节点、EP、vision）在 [Kimi K3 recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3)。

## Important Deployment Tips

1. **Prefix caching：** `--enable-prefix-caching` 打开。vLLM 通常默认开 prefix caching；**K3 当时默认关**，hybrid-cache 设计还在演化。要显式传 flag。
2. **Tool calling：** 先在自己的流量上校验再依赖。偶尔见过 K3 吐出自己 parser 不认的 tool-call 格式，`tool_calls` 空；同一套上干净 probe 却解析完美。跟 prompt 和 run 有关，不是一刀切失败。生产 agent 应对着 schema 校验；`tool_calls` 空时重试或 fallback；考虑 strict / structured tool calling，生成时约束 grammar。
3. **All-to-all：** `--all2all-backend` 管 EP 时 MoE 怎么通信。NVIDIA NVLink 用 `flashinfer_nvlink_one_sided`；RDMA 用 `deepep_v2`。
4. **MoE backend：** 多套。任何 DEP 环境推荐 `deep_gemm_mega_moe`。
5. **Rust frontend：** `VLLM_USE_RUST_FRONTEND=1`。完整支持这个模型。
6. **ViT parallelism：** `--mm-encoder-tp-mode=data` 默认开。K3 vision encoder `head_size=12`，TP=8 切不匀。Vision encoder 不到 1B，backbone 大约 2T，默认开 ViT DP，避开 encoder 的 all-reduce。

## FAQ

### How many GPUs do I need to serve Kimi K3?

至少一台 8× B300（或 GB300 NVL72）；16× B200 也支持。多数生产多节点 EP / DP，RDMA 或 NVLink。

### How do I enable DSpark speculative decoding?

加：

```bash
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

推理和编码负载上，单流 decode 大约三倍。

### Which MoE and all-to-all backend should I use?

拆分或 DEP 用 `deep_gemm_mega_moe`；TP > 1 用 `flashinfer_trtllm`。All-to-all 对齐互联：NVLink `flashinfer_nvlink_one_sided`，RDMA `deepep_v2`。

### Does Kimi K3 support prefix caching, and is it on by default?

支持：full-attention KV **和** recurrent KDA state。**默认关**，要传 `--enable-prefix-caching`。

### Does vLLM support Kimi K3 on AMD GPUs?

支持。Launch 就带 ROCm；更宽调参在 roadmap。

### How is this different from the Kimi K3 preview post?

[Preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) 是架构和 kernel 深潜，含 KDA prefix caching 怎么建。这篇是实用 launch 指南和产物：vLLM 怎么适配、recipes、flags、性能、生产上 K3 准备好了什么。笔记 [kimi-k3-preview.md](kimi-k3-preview.md)。

## Roadmap

- **K3 的 RL：** vLLM rollout 已加。会和 RL 生态项目一起做端到端 RL 训练。
- **持续性能：** day 0 之后继续抬。
- **Decode Context Parallelism (DCP)：** prototype 加速不错，很快上游。早期实验：选定负载上比 TP8 高 **40%** 吞吐。笔记 [dcp.md](../features/dcp.md)。
- **EPLB：** 改善性能。
- **Confidence-based scheduling：** 用 DSpark 的 confidence head 剪要 verify 的 draft tokens。
- **更宽 AMD ROCm 调参。**

## Quick links

- **Model：** [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- **DSpark draft：** [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)
- **Recipes / Docker：** [recipes.vllm.ai/moonshotai/Kimi-K3](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- **Kimi 技术文：** [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)
- **vLLM 设计：** [preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)

## Acknowledgements

感谢 Moonshot 做出 K3、发布前共享架构、共设计 KDA-aware caching；Inferact 端到端集成和部署校验；NVIDIA 的 fused KDA decode / prefill、AttnRes kernels、MXFP4 MoE；AMD 的 ROCm bring-up；inference partners：Alibaba Cloud、Baseten、DigitalOcean、Modal；Shikhar 的 Flash-Flash-KDA；vLLM 社区。为 K3 建的 cache 基础设施，现在属于每个类似架构的 hybrid 模型。页上说等不及看你 serve 什么。
