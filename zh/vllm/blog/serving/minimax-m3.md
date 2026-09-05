---
source: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# MiniMax M3：1M MSA、MXFP8、EAGLE3，day-0 能 serve 才算数

英文对照：[en/vllm/blog/serving/minimax-m3.md](../../../../en/vllm/blog/serving/minimax-m3.md)  
原文：https://vllm.ai/blog/2026-06-12-minimax-m3-vllm  
2026-06-12。vLLM Team。权重 [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3)、[`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8)。EAGLE3 draft [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3)。菜谱 [recipes.vllm.ai/MiniMaxAI/MiniMax-M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)。NVIDIA 校验：H200、GB200、B300。AMD：MI350 / MI300。MSA 源 [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)。vLLM PR [#45381](https://github.com/vllm-project/vllm/pull/45381)。NeMo RL [minimax-m3.md](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md)。H3 生产 serving：[minimax-h3.md](minimax-h3.md)。亲戚：[anatomy.md](../core/anatomy.md)、[spec-decode.md](../features/spec-decode.md)、[kv-offload.md](../features/kv-offload.md)。本地图版权仍归原站。

硬的不是把模型 load 进去。是把 MiniMax Sparse Attention、多模态预处理、MXFP8 MoE、EAGLE3、prefix caching、deployment recipes 放进用户真能跑的 serving 引擎。这篇走模型特性、vLLM 实现、kernel 和 cache，以及 day-0 之后还在落地的优化。

![Figure 1: MiniMax M3 day-0 long-context multimodal sparse-attention serving](../../../../assets/vllm/blog/serving/minimax-m3/01-hero-minimax-m3-vllm.svg)

## TL;DR

- **模型族：** BF16 和 MXFP8。1M 上下文视硬件容量和部署配置。
- **核心架构：** MiniMax Sparse Attention (MSA)。对 128-token KV block 打分，每 query / KV group 选 top blocks，在选中的块上跑 GQA。
- **Serving 栈：** `minimax_m3` tool / reasoning parsers、thinking-mode、text-only 和多模态、TP/EP、prefix caching、chunked prefill、EAGLE3、可用 Docker。
- **Speculative decoding：** day-0 EAGLE3，draft `Inferact/MiniMax-M3-EAGLE3`。
- **RL：** day-0 MiniMax M3 GRPO 在 [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)，vLLM 当 generation backend。
- **性能：** MSA prefill/decode、indexer-score 和 top-k、fused QKNorm + RoPE + KV insert、GemmaNorm 和 quantization-path、MXFP8 MoE backend。
- **Roadmap：** FP8 indexer/KV、TRTLLM-Gen MoE、更宽的拆分 serving recipes、context-parallel 长 prefill、多模态 gateway。

## MiniMax M3 Support Matrix

| 能力 | M3 加了什么 | vLLM |
| --- | --- | --- |
| 1M context | 长文本、代码、agent traces、文档 | `--max-model-len`、block-size 128 recipes、prefix caching、chunked prefill、MSA kernels |
| MSA | 在选中的 128-token KV 上 block-sparse GQA | Hybrid attention backend、indexer-score、top-k、sparse GQA prefill/decode |
| MXFP8 权重 | 大规模 MoE serving | Blackwell DeepGEMM MXFP8；Hopper Marlin MXFP8 |
| 原生多模态 | 图 + 视频 + 文本 | 模型专用预处理和 serving 集成 |
| Tool / reasoning | Agent 和可控 thinking | `minimax_m3` parsers、`thinking_mode` chat-template |
| EAGLE3 | draft 加速 | day-0 recipe + `Inferact/MiniMax-M3-EAGLE3` |

## Quickstart

NVIDIA 上 MSA 走默认 attention backend；vision encoder 走 FlashInfer（`--mm-encoder-attn-backend FLASHINFER`），shared-memory processor cache，data-parallel encoder。

Blackwell 上 MXFP8 起点：

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

BF16：

```bash
vllm serve MiniMaxAI/MiniMax-M3 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

具体 recipe 看加速器、dtype、上下文、流量形状，以及吞吐、延迟、最大上下文谁优先。校验做过 NVIDIA H200、GB200、B300。完整 NVIDIA / AMD launch recipes、策略、旋钮： [vLLM recipe for MiniMax M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)。

### AMD ROCm

Instinct 上能跑。MSA 走 Triton attention，所以加 `--attention-backend TRITON_ATTN`；vision encoder 走 AITER FlashAttention（`--mm-encoder-attn-backend ROCM_AITER_FA`），同样 shm processor cache + data-parallel encoder。

MXFP8：

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --attention-backend TRITON_ATTN \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend ROCM_AITER_FA \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

BF16：

```bash
vllm serve MiniMaxAI/MiniMax-M3 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --attention-backend TRITON_ATTN \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend ROCM_AITER_FA \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data
```

校验：MI350 Series、MI300 Series。

### Deployment Knobs That Matter

M3 有几颗旋钮比平时更要紧。`--block-size 128` 让 vLLM cache block 对齐 MSA 的 sparse 粒度。`--max-model-len` 管对外宣称的上下文和 KV 容量规划。`--tensor-parallel-size` 和 `--enable-expert-parallel` 决定 attention、projections、MoE experts 怎么切。Agent 负载打开 `minimax_m3` tool / reasoning parsers。长上下文 recipe 要写清：prefix caching、chunked prefill、EAGLE3、多模态预处理，这一份目标开没开。

### EAGLE3 Speculative Decoding

Day-0 EAGLE3。Draft：`Inferact/MiniMax-M3-EAGLE3`。流量和 acceptance 对得上时，用 draft 路径压 generation 延迟。

```bash
vllm serve MiniMaxAI/MiniMax-M3-MXFP8 \
  --block-size 128 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

例子用 `num_speculative_tokens=3`，校验用的保守起点。生产要按 acceptance、TPOT、吞吐、目标延迟和流量配比调。

### Thinking Mode

可控 thinking。vLLM 里经 `chat_template_kwargs`：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id

messages = [{"role": "user", "content": "Explain MiniMax Sparse Attention."}]

for mode in ["enabled", "disabled", "adaptive"]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            "chat_template_kwargs": {
                "thinking_mode": mode,
            },
        },
    )
    print(mode, response.choices[0].message.content)
```

## 模型关键特性和新能力

对推理系统，M3 在三个方向上要紧。

### 1M-Token Context with MiniMax Sparse Attention

中心架构变化是 MSA。不是每个 query 对整份 KV 做 dense attend，而是用 index path 给 KV blocks 打分，再给真正的 attention 选最相关的块。默认粒度 128-token KV block；选中的块在一个 GQA group 里共享。

每个 query token 三步：

1. 用小 index head 给候选 KV blocks 打分。
2. 选 top blocks，同时套配置里的 block rules。
3. 只在选中的 KV blocks 上跑 online-softmax attention。

用户期望的长上下文行为还在，每生成 token 的 attention 工作量有上界。实际上，MSA 是让 M3 的 1M 上下文在 vLLM serving 里变得实际的机制。

![Figure 2: MSA local + global context, sparse 128-token blocks from 1M history](../../../../assets/vllm/blog/serving/minimax-m3/02-msa-1m-context.svg)

### MSA Mechanics in More Detail

MSA 拆成两个问题：哪些过去的块值得读，以及怎么在那些块上跑 attention。Index path 答第一问：给固定 128-token KV blocks 打分。Sparse GQA 答第二问：在选中的块上跑 attention。

选中集合不只是学出来的 top-k。M3 config 暴露 `init_blocks` / `sparse_init_block` 和 `local_blocks` / `sparse_local_block`；**当前 recipe 用 `init_blocks=0`、`local_blocks=1`**。实践上，确定性规则是 query 附近的 local-window block；其余来自 indexer-scored top-k。正确性靠小细节：最后一块不完整要 mask；块内因果边界要守；已经排进 top-k 的 local blocks 不能算两次；batched 请求的有效 block range 可以不同。

### Native Multimodality

M3 是多模态模型，不是文本 checkpoint 外挂 sidecar。Serving 路径要接 image / video，预处理成 patch tensors，保住 grid metadata，交给模型时不从 generation 偷 GPU 时间。

Release 工作含模型专用多模态预处理和 parser，用户才能在同一套 serving 表面上跑 text-only、tool-use、reasoning、多模态。

### MXFP8 MoE Weights

MXFP8 checkpoint 对着大规模高效 serving。校验：Blackwell 用 DeepGEMM MXFP8 MoE；Hopper 用 Marlin MXFP8。

## vLLM Implementation

M3 是 hybrid：有的层走 dense attention，sparse 层走 MiniMax MSA backend。vLLM 把这个区别藏在模型和 attention backend 后面，scheduler、cache 分配、batching、prefix caching、serving 从外面仍眼熟。内部背景：[Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)，笔记 [anatomy.md](../core/anatomy.md)。

### MiniMax Sparse Attention Backend

MSA backend 两件职责。

第一，算 sparse metadata。Indexer 给 KV blocks 打分，套 block-selection rules，吐 top-k block IDs。对 M3，选择是 block-based：稀疏单位就是 cache manager 已经认识的那种 page-like 128-token block。

第二，在那些块上算 attention。Prefill 和 decode 形状不同，所以专用 kernel：

- **Prefill indexer-score：** Triton 算 block scores 和 top-k。
- **Prefill sparse GQA：** Triton 和 [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的 CuTe/SM100 路径。CuTe 把 query-to-block 映射翻成 K-major CSR，好复用 KV blocks。
- **Decode indexer-score：** split-style decode kernels 扫候选块、打分、merge top-k。
- **Decode sparse GQA：** GQA decode kernels 吃选中的 block pages，merge 部分 attention 输出。

### Prefill Execution

Prefill 处理 prompt、建 KV cache。对 M3，prompt 长度和 sparse metadata 都要紧。概念上四段：

1. **Build Q/K/V 和 index projections。** Dense projections 给 indexer 和 attention kernels 出表示。
2. **Score blocks。** Index path 给每个候选 KV block 一个分。Reduction 可用 block-level 规则，比如 max 或 log-sum-exp，看模型配置。
3. **Select blocks。** Top-k 把学到的分数和配置规则合在一起，给每个 query / KV group 吐 block IDs。
4. **Run sparse GQA。** Kernel 只读选中的 KV blocks，算的是「dense attention 限制在这个选中集合上」的同一份 online-softmax。

最后一段 sparse GQA 有两种有用的 schedule。Query-major 直白：每个 query 走自己选中的 KV blocks。KV-block-major 对长 prompt 更好——许多 query 选同一块时。那时 vLLM 建 K-to-Q mapping，一块 KV 可以 load 一次、给许多 query 用，再做 output merge。

### Decode Execution

Decode 形状不同。每步通常每条活跃序列一个新 token，但 batch 里可以有许多序列、不同上下文长度。Runtime 更新 cache、给候选块打分、处理 local-window、选 top blocks、跑 sparse GQA decode；kernel 用 split 时还要 merge 部分输出。这发生在 **每个** 生成 token 上，所以 indexer-score 和 top-k 是 **TPOT 的一部分**，不是 setup 开销。

M3 的 sparse-attention config 管：block size、top-k 数、optional init blocks、local-window blocks、index dimension、sparse layer IDs、score type、以及哪些层只用 index attention 做 selection。关键实现规则：每个选中的 block ID 必须映射回 vLLM scheduler 和 cache manager 认识的同一份逻辑 request state。

![Figure 3: dense layers vs MiniMax MSA backend](../../../../assets/vllm/blog/serving/minimax-m3/03-msa-backend-dispatch.svg)

### KV Cache Layout: Standard Storage, Sparse Computation

M3 可以把 KV 存成普通 paged KV，稀疏发生在计算路径。Cache manager 保持简单，kernel 需要的灵活性另加：

- Main attention KV cache 和 indexer K cache **显式**跟踪。
- Prefix caching 和 chunked prefill 在 recipe 的 cache-state 交互校验过后，可以继续用稳定 cache blocks。
- 相关的拆分 serving 和 NIXL 风格 transfer，可以把 cache 当 paged state，attention backend 负责 sparse selection。

### Prefix Caching and Chunked Prefill

Prefix caching 要紧，因为 M3 负载常常复用长 prompt：代码库、文档、多轮 agent traces、多模态上下文。Chunked prefill 要紧，因为 1M-token 请求不该以一整块巨大 prefill 独占引擎。合在一起是 release-readiness 压力测试：index cache、main attention KV、dense attention state、prefix hits、preemption、batching、长上下文 chunk 边界，都要在同一套 block tables 上达成一致，recipe 才能当生产。

### Multimodal and Parser Integration

模型专用的 tool / reasoning / 多模态解析。vLLM 支持：

- `--tool-call-parser minimax_m3`
- `--reasoning-parser minimax_m3`
- Chat template 的 `thinking_mode`
- Image / video 预处理集成

生产上，预处理尽量在 GPU 执行之前做完。目标架构：gateway 下载媒体、解码帧、采样视频、resize / normalize 图、造 patch tensors，把 **ready-to-run tensors** 交给 worker。

这要紧，因为多模态请求在 API 边界上看起来小，预处理之后可以很大。一段视频要帧采样、每帧 resize、patch、metadata packing。CPU 重的媒体活放在上游，GPU 调度更好讲。

Parser 对 agent 流量同样重要。Tool-call 和 reasoning parsers 把模型专用文本约定变成结构化 API。没有对的 parser，模型可以吐出有用的字，应用却难吃。

![Figure 4: CPU-side image/video preprocessing hands ready tensors to the worker](../../../../assets/vllm/blog/serving/minimax-m3/04-multimodal-request-path.svg)

## Performance Optimizations

M3 把瓶颈挪了地方。MSA 减 dense attention，却引入 indexer-score、block selection、sparse metadata、额外小 kernel。Day-0 实现盯着：让这些新零件便宜。

原则简单：**花在「决定读哪些块」上的时间，别超过「不读所有块」省下来的时间。** 三处落地：block-major prefill、瘦的 decode indexer-score kernels、attention 路径周围融合小的 elementwise / cache-write kernels。

### KV-Block-Major Prefill

Prefill 时许多 query tokens 会选同一 KV block。朴素 query-major sparse attention 会反复把同一块从 HBM 搬到片上。[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的 CuTe/SM100 路径建 K-to-Q CSR，跑 block-major sparse attention，再用 log-sum-exp 合并部分输出。长 prompt 和常见长缓存上下文的 agent 流量，算术强度上去。

![Figure 5: KV-block-major prefill reuses selected KV blocks](../../../../assets/vllm/blog/serving/minimax-m3/05-kv-block-major-prefill.svg)

### Decode Indexer-Score Kernels

Decode 里 indexer 在 **每个** 生成 token 的关键路径上。引擎要把 query 侧 index vectors 和候选 key 侧比，把每个 128-token block 收成一个分，处理 local-window，只留给 sparse GQA top blocks。

优化过的 decode 用专用 indexer-score kernels，不当成 padded dense GEMM。避免在 ragged per-request block ranges 周围加活，让 top-k 边界靠近 score 计算。

Decode 还要小心内存流量。选中的 KV blocks 在逻辑序列空间里稀疏，在内存里仍是 page-like。除非复用值得，kernel 不该把 sparse pages 摊成大块临时 dense tensors。

### Speculative Decoding in the Decode Kernels

EAGLE3 还要求 decode kernels 高效做 speculative verification。一条请求一次可以 verify 多个 draft tokens，所以 MSA decode **不能假设** 每条请求恰好一个 query token。

一种 fallback 是用 prefill kernels 做 speculative verification，代价高：prefill kernels 通常对着大得多的 token 数调，小 draft-token batch 上差；通常也不兼容 full CUDA graph——低延迟 decode 的重要优化。

Day-0 更新了 MSA decode indexer、top-k、sparse GQA decode，支持统一的 `decode_query_len`。Kernels 按 request-major 把 speculative verification tokens flatten，再把每个 query token 映射回正确的 request metadata、sequence length、block table、causal position。EAGLE3 verification 走 decode 专用的 split-K，而不是较不对题的 prefill 风格路径；speculative 路径也靠近现有 decode。

同一路径支持 uniform speculative decode batches 的 **full CUDA graph**。Launch grids 形状稳定；选中的 arguments 避免不必要的 Triton specialization；padded request rows 显式处理，captured graphs 才能安全 replay。这些细节要紧：speculative decoding 只有在 draft acceptance 不被额外 kernel launches、recompiles、cache-state 开销抵消时，才改善 TPOT。页上说会继续按不同 draft 长度、并发、流量配比优化。

### Kernel Fusions

若干更小的 kernel 被融合或经 custom ops，减 launch 和 HBM 往返：

- **QKNorm + RoPE + KV insert：** MSA 路径上归一化、位置编码、cache write 合一。
- **GemmaNorm 和 AllReduce + Norm：** 减 TP 执行里归一化周围的开销。
- **Quantization-path cleanup：** 改善 `silu_mul_quant_fp8` 和相关 MXFP8/MoE 输入路径。
- **Router 和 MoE kernels：** 减 sparse expert 路径开销，为更深的 TRTLLM-Gen 集成做准备。

Release 路径有意保守：正确性和稳定 cache 行为，压过 day-0 打开每一个 graph / fusion 旋钮。更激进的融合可以等公开 recipes 成熟再落。

### Quantization and KV Cache Dtype

MXFP8 checkpoint **主要改权重和 MoE 执行**，不是 KV cache 的概念结构。公开 recipes 应把 model dtype、MoE backend、KV-cache 策略 **分开写**：「MXFP8 model」不自动等于每个 cache 和中间张量都是 MXFP8。Roadmap 含 FP8 indexer 和 KV-cache，因为 KV 容量直接决定一份部署能 serve 多少长上下文和 batched 流量。

### CUDA Graphs and Compile Behavior

CUDA graphs 对 decode 有价值，因为 M3 在每 token 周围引入若干小操作。但 graph capture 只有在 captured path 跨 batch shapes、cache states、sparse metadata **稳定** 时才帮得上。Day-0 在需要处用保守 graph 设置，校验成熟后再扩覆盖。

## Validation

公开前，vLLM 团队按日跑 accuracy、吞吐、speculative decoding、容器可用性。

校验环三个目标：

1. **Functional correctness：** 模型能 load、serve、解析 tool / reasoning、处理 text-only 和多模态。
2. **Accuracy parity：** kernel、cache、parser、recipe 改完，benchmark 仍对齐预期模型行为。
3. **Serving readiness：** 容器在目标加速器上按打算的 TP/EP/speculative 设置跑。

最有用的测试把短正确性任务和长输出、长上下文负载合在一起。短任务很快抓住 parser、格式、明显数值问题。长上下文抓住 MSA metadata、prefix caching、chunked prefill、KV-cache layout。Speculative decoding 抓住普通 accuracy 跑不出来的 acceptance 回退。

B300 上的代表快照：

| 维度 | 结果 |
| --- | ---: |
| GSM8K strict / flexible | 91.51% / 91.66% |
| ShareGPT @256 throughput | 8,530 tok/s |
| ShareGPT @256 TPOT | 56.0 ms |
| Speculative Sonnet TPOT，concurrency 1 / 16 / 64 | 4.51 / 9.04 / 14.36 ms |
| Speculative acceptance on Sonnet | ~67%，mean accept length ~3.0 |

工程校验测量，不是官方榜。精确结果随 image、权重、recipe、硬件变。

![Figure 6: release-candidate validation dashboard](../../../../assets/vllm/blog/serving/minimax-m3/06-validation-dashboard.svg)

## Beyond Serving: RL Post-Training with NeMo RL

Day-0 不只是推理 serving。RL 框架把 vLLM 当训练环里出 rollouts 的 generation 引擎，所以撑起 serving 的同一份 M3 工作（[vLLM PR #45381](https://github.com/vllm-project/vllm/pull/45381)）也让 M3 的 post-training 在 day 0 成为可能。

[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) 现在用 vLLM 做 **non-colocated** generation backend 跑 MiniMax M3。短 GRPO（Group Relative Policy Optimization）post-training 已在 BF16 checkpoint 上校验：NeMo AutoModel + expert parallelism + BF16 vLLM generation。长跑收敛和 **超出 expert parallel** 的并行策略仍在校验。早期结果说明一份扎实 serving 路径值什么：serve M3 的引擎，也驱动 RL 训练的 rollout 阶段。参考菜谱：[NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md)。

## Roadmap

Day-0 是起跑线。下一截已经在飞：

- **FP8 indexer 和 KV-cache：** 减 KV 内存压力、抬 batch 容量，同时保住 sparse-attention 精度。
- **TRTLLM-Gen MoE：** Blackwell 上 MXFP8 expert 执行。
- **Context parallelism：** 单节点不够时的超长 prefill 扩展。
- **Disaggregated serving：** 扩 NIXL 和 P/D 拆分 recipes，方向见 [Large-Scale Serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)。
- **Kernel fusion：** 减 MSA 引入的许多小 indexer / top-k / quantization / normalization kernels。
- **Multimodal gateway：** 把图和视频预处理留在 GPU generation 关键路径外。

## FAQ

### Does vLLM support MiniMax M3?

支持。这篇覆盖 BF16 和 MXFP8 的 day-0：MSA、模型专用 parsers、EAGLE3、多模态预处理、TP/EP recipes、可用 Docker。

### What is MiniMax Sparse Attention?

给固定 128-token KV blocks 打分，为每个 query 和 GQA group 选最相关的块，套配置的 local-window，在选中集合上跑 sparse GQA。当前 M3 recipe：`init_blocks=0`、`local_blocks=1`。

### Does MXFP8 mean the KV cache is MXFP8?

**不。** MXFP8 描述模型权重和 MoE 执行。KV-cache dtype 是另一项 serving 决定；当前 sparse-attention 校验把 native KV 存储和量化 KV-cache 当成分开的 roadmap。

### What settings matter most for 1M-token context?

起点：`--block-size 128`、为所选 batch / 上下文形状留够 GPU 内存、recipe 写清 prefix caching / chunked prefill / EAGLE3 开没开。默认 vLLM 从模型 config 读上下文长度，**不必**设 `--max-model-len`。GPU 内存有限或不需要满 1M 窗口时，可以传 `--max-model-len` 往下封，减 KV 压力。

## Acknowledgments

感谢 MiniMax 开源 MiniMax-M3，以及 MiniMax 管理层对 vLLM 的信任和支持。模型支持由 **Inferact Inc.** 牵头——目标是把 vLLM 做成世界的 AI inference engine，让推理更便宜更快。NVIDIA 和 AMD 贡献了硬件支持。

## Related vLLM Reading

- [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — scheduler、KV、prefix caching、分布式。笔记 [anatomy.md](../core/anatomy.md)。
- [Speculative Decoding](https://vllm.ai/blog/2024-10-17-spec-decode)、[P-EAGLE](https://vllm.ai/blog/2026-03-13-p-eagle)。笔记 [spec-decode.md](../features/spec-decode.md)。
- [Large-Scale Serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)、[KV Offloading Connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)、[Moriio KV Connector](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)。笔记 [kv-offload.md](../features/kv-offload.md)。
- [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md)。
