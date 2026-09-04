---
source: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# ROCm 上的七条 attention 路：编排，不是移植

英文对照：[en/vllm/blog/architecture/rocm-attention.md](../../../../en/vllm/blog/architecture/rocm-attention.md)  
原文：https://vllm.ai/blog/2026-02-27-rocm-attention-backend  
2026-02-27。署名 **AMD and Embedded LLM**。快照：`vllm` **0.14.0rc2** / 镜像 `rocm/vllm-dev:nightly_main_20260115`，**ROCm 7.0.0**。页上的 bench，不是你的 SLA。Triton 默认（一只统一 kernel）见 [triton-attn](triton-attn.md)；这篇讲 **AITER FA 三路**，外加其余六条 ROCm 后端。硬件那扇门：[hardware-plugin](hardware-plugin.md)。集群规模上的 Prefill / Decode 拆分：[large-scale](../serving/large-scale.md)。

适用：要在 Instinct 上选 `--attention-backend`、读懂 `ROCM_AITER_FA` 为什么拆三路、对照 Qwen3-235B / DeepSeek-R1 的相对 TPS。不适合：把页上的 **1.2–4.4×** 当成承诺。

`VLLM_ROCM_USE_AITER=1` 会自动选：MHA 走 `ROCM_AITER_FA`，MLA 走 `ROCM_AITER_MLA`。KV 预洗牌相对标准 layout 约 **15–20%** decode TPS。下文插图为学习对照用（原文版权仍归原站）。

## 概览

很久以来，AMD 支持等于 **移植**：先让代码跑起来。原文说那一页翻过去了。CDNA 3（Instinct **MI300X**、**MI325X**、**MI355X**）加上 DeepSeek **MLA** 这种结构，要的是 **架构共设计**：软件编排和硬件原语一起干活。

vLLM 在 AMD ROCm 上给出 **7** 条 attention 后端。这篇把每条的来历、取舍、何时用写开，并给对照 bench。标题句：`ROCM_AITER_FA`（MHA）和 AITER MLA 后端，靠 AITER 原语加 vLLM 的 kernel 编排，相对页上其他 ROCm 选项约 **1.2–4.4×** 系统 TPS。

## 每一批都是混合负载

生产 serving 不会给你干净的 Prefill 批或干净的 Decode 批。每一步推理都在同一批里混着不同请求类型。业界有两条脾气：一只统一 kernel 在内部自己调度；或多路路由、各用特化 kernel。AMD 的 `ROCM_AITER_FA` 走 **显式路由**：负载感知是一等设计，不是 kernel 肚子里的细节。

- **Prefill。** 新 prompt。上千 input token 一次算完 attention。重 GEMM → **compute-bound**。
- **Extend。** 某条请求的 KV 已经长了一截（chunked Prefill、prefix cache、上一轮对话），还要再吃 prompt 侧 token。新 token 既看 **缓存上下文** 又看 **新鲜输入** → 混合。在线调度器把长 prompt 拆开，跟别的在飞请求的 Decode 交错。
- **Decode。** 一次一个输出 token。每步把整份 KV 从内存拉上来 → **memory-bound**。

三种请求随机到达，再被打进同一批。

![continuous batching](../../../../assets/vllm/blog/architecture/rocm-attention/01-continuous-batching.png)

**Figure 1。** 五路并发的在线 serving。Step 4 把 Prefill、Extend、Decode token 打进同一批。

Prefill 要大 tile、把 ALU 吃满。Decode 要合并访存、少取 cache。**为一头调的 kernel，另一头就会把性能留在桌上。** 混合批正是 `ROCM_AITER_FA` 三路要解的：按类型送到特化 kernel，而不是逼一只 kernel 通吃。

## 其余 MHA 后端

先把另外几条 MHA 路看清，再进 `ROCM_AITER_FA`。

### 统一 attention

![ROCm Attention unified attn](../../../../assets/vllm/blog/architecture/rocm-attention/02-ROCm-Attention-unified-attn.png)

**Figure 2。** 统一 attention：Prefill / Extend / Decode 全走一只 kernel。

| Backend | Kernel 来源 | 用途 |
| --- | --- | --- |
| [TRITON_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/triton_attn.py) | [vLLM Triton kernel](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/ops/triton_unified_attention.py) | 默认 fallback |
| [ROCM_AITER_UNIFIED_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/rocm_aiter_unified_attn.py) | [AITER Triton kernel](https://github.com/ROCm/aiter/blob/v0.1.10.post3/aiter/ops/triton/_triton_kernels/attention/unified_attention.py) | 单 kernel 的 AITER 路 |

```python
def forward():
    # Stage 1: Save Key/Value into KV-Cache
    reshape_and_cache_flush(new_key, new_value, ...)
    # Stage 2: Single kernel for all attention
    unified_attention_kernel(new_query, KV-Cache, ...)
```

`TRITON_ATTN` 是 [triton-attn](triton-attn.md) 里那份可移植默认：一份 Triton 源码，总能当 fallback。这篇的 `ROCM_AITER_FA` 是另一注——**三条显式路径**，不是一只万能 kernel。

### `ROCM_ATTN`：旧的两路

[ROCM_ATTN](https://github.com/vllm-project/vllm/blob/v0.14.0rc2/vllm/v1/attention/backends/rocm_attn.py) 按阶段换 kernel：

- **Prefill：** Triton kernel
- **Decode：** HIP paged attention（支持时）

原文点名的两条性格：

1. **旧两路。** Prefill 用 Triton，Decode 用 HIP paged attention。HIP 这条只认一部分 KV head size。不支持的配置（原文点名 **Qwen3-235B**）会退回 Triton Decode，**明显变慢**。
2. **Radeon。** 和 `TRITON_ATTN` 一样能上 **Radeon**——消费卡、没有 AITER 原语的地方用得上。

## `ROCM_AITER_FA`：给 AMD 做的 kernel 编排

不是 kernel 外套。是一层编排：请求分到特化 kernel，vLLM 管上层，AITER 出原语。

![ROCm Attention rocm aiter fa](../../../../assets/vllm/blog/architecture/rocm-attention/03-ROCm-Attention-rocm_aiter_fa.png)

**Figure 3。** `ROCM_AITER_FA` 把 token 分到三条特化路。

### 四件新东西

**1. 三路路由。** 请求动态分成 Decode、Prefill、Extend，各有优化过的 kernel：

- **Prefill 路。** 新序列走 `flash_attn_varlen_func`——CDNA 矩阵核扛计算。
- **Extend 路。** 续写序列走切块 attention + LSE merge——**100K+** 上下文。
- **Decode 路。** 单 token 生成走 AITER 为带宽调过的 kernel。

原文还有一帧短动画：R1（Decode token）进 Decode 路，R2（Prefill token）进 Prefill 路。未复制。

**2. Batch 重排（model runner）。** 会在处理前重排请求的后端不多，`ROCM_AITER_FA` 是其中一个。Model runner 把请求排成 `[decode:extend:prefill]`，好让内存连续。后端用 `reorder_batch_threshold` 报名；`ROCM_AITER_FA` 设成 **1**，于是每一批混合都会先重排，再交给三路。

![batch reordering](../../../../assets/vllm/blog/architecture/rocm-attention/04-batch_reordering.png)

**Figure 4。** 重排之后每条 kernel 路吃到连续 token，少做一遍 KV 取数。

原文第二帧动画（未复制）：重排成 `[decode > extend > prefill]`，再把 R3 送进 Extend 路。

**3. 切块上下文。** 长序列按每 iteration 固定 token 预算切（合计约 **32K**），摊到各条 extend 请求上；LSE merge 保数值稳定。

![chunked context flow](../../../../assets/vllm/blog/architecture/rocm-attention/05-chunked_context_flow.png)

**Figure 5。** 100K+ token 的上下文按 32K 切块，再用 LSE 合。

**4. 给硬件洗过的 KV layout。** AITER kernel 组设计的预洗牌：

```python
k_cache: [num_blocks, num_heads, head_dim // x, block_size, x]
v_cache: [num_blocks, num_heads, block_size // x, head_dim, x]
```

访存跟 CDNA 对齐。Decode 可以直接叫 AITER 的 `pa_fwd_asm`，**零 layout 转换** → 相对标准 KV layout 约 **15–20%** decode TPS。

### 为什么要显式三路？

在软件层分流，而不是指望一只 kernel 包圆：

- **好 debug。** 每条路可以单独 profile、调、优化。
- **可移植。** 同一套路由从 **MI300X → MI325X → MI355X**，不必按卡改逻辑。
- **可加长。** 新负载类型或新 kernel 变体，不必重画骨架。
- **可预期。** 路径确定，性能分析不绕。

Extend 这条在生产里特别要紧：prefix cache 和多轮对话已经是标配。专路加切块上下文，让它们成为一等公民，而不是退路。

### 三路怎么算

**Prefill。** Q/K/V 停在标准 `[num_tokens, num_heads, head_dim]`，对齐 AITER MHA，少一次拷贝。

**Extend。** 最难的一条。新 token 要对已经洗过牌的 KV 做 attention。这份 shuffled layout 跟 AITER 的长上下文 MHA 合不来，于是多一个 gather（`cp_mha_gather_cache`），把上下文 K/V 捞回标准 layout。长上下文再切段：

```python
def extend_forward():
    # Stage 1: Attention for new tokens
    flash_attn_varlen_func()  # calling AITER MHA

    # Stage 2: Context Chunk Loop Processing
    for chunk in context_chunks:
        cp_mha_gather_cache()      # Triton gather kernel
        flash_attn_varlen_func()   # calling AITER MHA
        merge_attn_states()        # LSE-based merge

    # Stage 3: Get the final result
    merge_attn_states()
```

每块吐一份输出和一份 LSE（log-sum-exp）。LSE 是 softmax 的分母，合的时候数值稳定——attention 分高的块自然压过别的。

**Decode。** 直接吃 shuffled layout。定制的 `reshape_and_cache_flush` 保证 cache 一直是洗过的，后端就能零转换叫 `pa_fwd_asm`。

### 请求流（原文七帧动画写成表）

原文有一帧跨 7 个 iteration 的交互动画（未复制）。同一条故事，写在表里：

| Iteration | 关键事件 |
| --- | --- |
| **1** | R1 进场 → tokenization → scheduler 队列 → QKV 投影 → **Prefill 路** → sample 1 token。R2 半途到达，在队列里等。 |
| **2** | R1 + R2 同批。R1 → **Decode 路**，R2 → **Prefill 路**。R3、R4 入队。 |
| **3** | 4 条请求同批。Token budget = **100**，R3 只排 100 token（还剩 180）。R3 输出 = 0（prompt 还没算完）。 |
| **4** | R3 进 **Extend 路** 续算剩下的 prompt。Batch 重排：张量变成 `[decode > extend > prefill]`。 |
| **5** | 继续重排：`[decode > extend]`。R5 做完 Extend，转 Decode。 |
| **6–7** | 全部在 **Decode 路**，生成到 stop。 |

`ROCM_AITER_FA` 按请求状态在 Prefill → Extend → Decode 之间换路，混合批才能吃得下。

## AITER MLA：给 DeepSeek 调的

DeepSeek / Kimi 的 **MLA** 把 KV 压到 **576** 维（标准 MHA 大约 **8K**）——大约 **14×** 省内存。Attention 的脾气变了，MHA 那套菜谱搬不过去。

### 混合做法

两条基于 AITER 的 MLA 后端，Prefill 实现不同：

| Backend | Prefill kernel | Decode kernel |
| --- | --- | --- |
| `TRITON_MLA` | vLLM Triton | vLLM Triton |
| `ROCM_AITER_MLA` | AITER MHA | AITER Assembly |
| `ROCM_AITER_TRITON_MLA` | AITER Triton MHA | AITER Assembly |

底盘 `TRITON_MLA` 两阶段都用 vLLM 默认 Triton。AITER 两条把 Decode 换成手调汇编（`mla_decode_fwd`）——**增益多半在这里**。两条 AITER 后端的唯一差别是 Prefill：`ROCM_AITER_MLA` 叫 `aiter.flash_attn_varlen_func`（AITER MHA 自动派到 CK 或汇编），`ROCM_AITER_TRITON_MLA` 叫 `aiter.ops.triton.mha.flash_attn_varlen_func`（AITER Triton MHA）。

### Absorbed 与 non-absorbed

所有 MLA 后端共用一份菜谱：

- **Prefill / Extend（non-absorbed）。** 在未压缩表示上跑标准 MHA kernel。
- **Decode（absorbed）。** 专用 MLA kernel，直接在压缩后的 **576 维** latent 上算。

```python
def _forward_prefill():
    # Stage 1: Attention for new tokens (non-absorbed)
    _run_prefill_new_tokens()

    # Stage 2: For extend path, context chunk loop
    for chunk in context_chunks:
        gather_and_maybe_dequant_cache()
        _run_prefill_context_chunk()
        merge_attn_states()

    # Stage 3: Final merge
    merge_attn_states()
```

Decode 仍然 **memory-bound**——一次一个 token，KV 虽压缩，瓶颈还是 HBM3 带宽。AITER 汇编 `mla_decode_fwd` 就是为了把每个字节吃干净；泛用 Triton Decode 会输。

### 为什么汇编 Decode 要紧

`ROCM_AITER_MLA` 和 `ROCM_AITER_TRITON_MLA` **共用** 同一只汇编 Decode（`mla_decode_fwd`）：

| 阶段 | AITER MLA 后端 | vLLM `TRITON_MLA` 基线 |
| --- | --- | --- |
| **Prefill** | AITER MHA 或 Triton（看哪条） | Triton flash attention |
| **Decode** | 汇编 `mla_decode_fwd` | Triton `decode_attention_fwd` |

**1.2–1.6×** 加速主要来自这只共用的汇编 Decode。TPOT 偏 Decode（OSL=1K 就是 **1K** 次 iteration），所以优化 Decode 对吞吐贡献最大。两条 AITER 后端的 Prefill 差别，端到端几乎看不见。

它们还继承 FlashMLABackend 的整套能力：`FULL_AND_PIECEWISE` CUDA graph、MTP。几乎任意 KV cache **block size** 成绩都差不多——可以把每个 token 都当 prefix cache，不必怕细粒度缓存通常要付的税。

## 性能对照

**方法。** `rocm/vllm-dev:nightly_main_20260115`，**ROCm 7.0.0**。2026-01-15 从 `vllm` main 打的 nightly。先用请求把 kernel 热身；**去掉第一轮**（JIT）。

### 原文的 server CLI

**MHA（Qwen3-235B）：**

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

# Choose backend: TRITON_ATTN, ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN
ATTN_BACKEND="ROCM_AITER_FA"

model_path=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --attention-backend ${ATTN_BACKEND} \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234
```

**MLA（DeepSeek-R1）：**

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

# Choose backend: TRITON_MLA, ROCM_AITER_MLA, ROCM_AITER_TRITON_MLA
ATTN_BACKEND="ROCM_AITER_MLA"

model_path=deepseek-ai/DeepSeek-R1-0528
vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --attention-backend ${ATTN_BACKEND} \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --async-scheduling \
    --port 1234
```

### MHA 成绩

**模型：** [Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)，attention TP8 + MoE EP8。**负载：** ISL=10K，OSL=1K，**64** 与 **128** 并发。

![mha tpot comparison](../../../../assets/vllm/blog/architecture/rocm-attention/06-mha_tpot_comparison.png)

**Figure 6。** `ROCM_AITER_FA` 相对旧 `ROCM_ATTN`，TPOT 快 **2.8–4.6×**（MI300X / MI325X / MI355X）。

![mha ttft comparison](../../../../assets/vllm/blog/architecture/rocm-attention/07-mha_ttft_comparison.png)

**Figure 7。** TTFT：64 与 128 并发下，`ROCM_AITER_FA` 和 `ROCM_AITER_UNIFIED_ATTN` 领跑 Prefill。

![mha tps comparison](../../../../assets/vllm/blog/architecture/rocm-attention/08-mha_tps_comparison.png)

**Figure 8。** 输出 TPS 跟 TPOT 同向——`ROCM_AITER_FA` 相对旧 `ROCM_ATTN` 高 **2.7–4.4×**。

**相对 `ROCM_AITER_FA`，TPS 慢几倍（64 并发）：**

| Hardware | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| --- | ---: | ---: | ---: | ---: |
| MI300X | **1.00×** | 1.05× | 1.30× | 3.82× |
| MI325X | **1.00×** | 1.02× | 1.19× | 4.36× |
| MI355X | **1.00×** | 0.95× | 1.08× | 3.61× |

**相对 `ROCM_AITER_FA`，TPS 慢几倍（128 并发）：**

| Hardware | ROCM_AITER_FA | ROCM_AITER_UNIFIED_ATTN | TRITON_ATTN | ROCM_ATTN |
| --- | ---: | ---: | ---: | ---: |
| MI300X | **1.00×** | 1.05× | 1.36× | 2.65× |
| MI325X | **1.00×** | 1.00× | 1.28× | 3.12× |
| MI355X | **1.00×** | 1.01× | 1.23× | 2.88× |

相对名次跨代卡稳定。`ROCM_AITER_UNIFIED_ATTN`（单 kernel）在这份 **均匀** 负载上离 `ROCM_AITER_FA`（三路）不超过 **5%**——三路的好处，更该在混合流量、prefix cache hit 上显现。

这里的 `ROCM_ATTN` TPS 慢 **2.7–4.4×**，是因为 Qwen3-235B 的 KV head size HIP paged attention 不认，退回了 Triton Decode。原文另写：head size 支持时，`ROCM_ATTN` **比 `TRITON_ATTN` 快**。

### MLA 成绩

**模型：** [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)，TP8，`block_size=16`。**负载：** ISL=10K，OSL=1K，**64** 与 **128** 并发。

![mla tpot comparison](../../../../assets/vllm/blog/architecture/rocm-attention/09-mla_tpot_comparison.png)

**Figure 9。** AITER MLA 相对 `TRITON_MLA`，TPOT 快 **1.2–1.6×**（三款卡），功劳在共用的汇编 Decode。

![mla ttft comparison](../../../../assets/vllm/blog/architecture/rocm-attention/10-mla_ttft_comparison.png)

**Figure 10。** TTFT：128 并发、**MI355X** 上 `ROCM_AITER_MLA` 最好。

![mla tps comparison](../../../../assets/vllm/blog/architecture/rocm-attention/11-mla_tps_comparison.png)

**Figure 11。** 输出 TPS：AITER MLA 相对 `TRITON_MLA` 最多约 **1.5×**。

**相对 `ROCM_AITER_MLA`，TPS 慢几倍（64 并发）：**

| Hardware | ROCM_AITER_MLA | ROCM_AITER_TRITON_MLA | TRITON_MLA |
| --- | ---: | ---: | ---: |
| MI300X | **1.00×** | 0.98× | 1.33× |
| MI325X | **1.00×** | 0.98× | 1.41× |
| MI355X | **1.00×** | 1.03× | 1.52× |

**相对 `ROCM_AITER_MLA`，TPS 慢几倍（128 并发）：**

| Hardware | ROCM_AITER_MLA | ROCM_AITER_TRITON_MLA | TRITON_MLA |
| --- | ---: | ---: | ---: |
| MI300X | **1.00×** | 0.97× | 1.24× |
| MI325X | **1.00×** | 0.97× | 1.24× |
| MI355X | **1.00×** | 1.01× | 1.35× |

两条 AITER MLA 整体接近。**gfx942**（MI300X / MI325X）上 `ROCM_AITER_TRITON_MLA` TPS 高 **2–3%**。**gfx950**（MI355X）上 `ROCM_AITER_MLA` 持平或更好，因为它走 AITER 汇编 MHA Prefill。MI355X 的最佳 TTFT 也是 `ROCM_AITER_MLA`。原文推荐：所有负载都用自动选出的 `ROCM_AITER_MLA`。

这些 bench 是 **均匀** 请求长度。生产里的 prefix cache、长短混杂、花样请求，才会更充分地锻炼三路。

## 协作：vLLM + AITER

增益不是某一记优化。是 vLLM 的编排层和 AMD 的 AITER 原语叠在一起。「只移植」之所以不够，原因在这里。

![system stack](../../../../assets/vllm/blog/architecture/rocm-attention/12-system_stack.png)

**Figure 12。** 整栈：用户请求 → vLLM 编排 → AITER 原语 → AMD 硬件。

### 创新记在谁头上

![innovation attribution](../../../../assets/vllm/blog/architecture/rocm-attention/13-innovation_attribution.png)

**Figure 13。** vLLM 编排管路由和切块；AITER 提供为硬件调过的原语。

AITER：为 CDNA 专门做的 attention 原语。vLLM：负载感知路由和切块处理，把最后一档性能解锁。**单独哪一层都到不了最优。**

## 上手

### 自动选

```bash
# Recommended: Let vLLM auto-select optimized backends
export VLLM_ROCM_USE_AITER=1
vllm serve <your-model> --tensor-parallel-size <tp>
```

`VLLM_ROCM_USE_AITER=1` 时，vLLM 会选：

- MHA 模型（Llama、Qwen、Mistral）→ `ROCM_AITER_FA`
- MLA 模型（DeepSeek、Kimi）→ `ROCM_AITER_MLA`

### 显式 `--attention-backend`

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --tensor-parallel-size 8 \
    --attention-backend ROCM_AITER_TRITON_MLA
```

两条 AITER MLA 共用汇编 Decode，整体数字接近。Prefill 随架构略有差别；Decode 占主菜，差距就小。多数人用自动选出的 `ROCM_AITER_MLA` 即可。

### 硬件支持

| GPU | Memory | Architecture |
| --- | --- | --- |
| MI300X | 192GB HBM3 | gfx942 |
| MI325X | 256GB HBM3e | gfx942 |
| MI355X | 288GB HBM3e | gfx950 |

### 完整后端对照

AMD ROCm 上的七条 attention 路：

| Category | Backend | 怎么开 | 备注 |
| --- | --- | --- | --- |
| MHA | TRITON_ATTN | `--attention-backend TRITON_ATTN` | 基线，支持 Radeon |
| MHA | ROCM_AITER_UNIFIED_ATTN | `--attention-backend ROCM_AITER_UNIFIED_ATTN` | AITER 统一 kernel |
| MHA | ROCM_ATTN | `--attention-backend ROCM_ATTN` | 旧两路，支持 Radeon |
| MHA | **ROCM_AITER_FA** | `--attention-backend ROCM_AITER_FA` + `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | **推荐**，开 AITER 时自动选 |
| MLA | TRITON_MLA | `--attention-backend TRITON_MLA` | 基线，支持 Radeon |
| MLA | **ROCM_AITER_MLA** | `--attention-backend ROCM_AITER_MLA` | **推荐**，开 AITER 时自动选 |
| MLA | ROCM_AITER_TRITON_MLA | `--attention-backend ROCM_AITER_TRITON_MLA` | 另一条 AITER MLA |

## 收束

「只移植」那一页翻过去了。七条 ROCm attention 后端都有对照 bench。

**要点（ISL=10K，OSL=1K）：**

- `ROCM_AITER_FA`：MHA 上相对 `ROCM_ATTN`，TPS 高 **2.7–4.4×**
- `ROCM_AITER_MLA`：DeepSeek MLA 上相对 `TRITON_MLA`，靠汇编 Decode，TPS 高 **1.2–1.5×**
- 名次从 **MI300X → MI325X → MI355X** 站得住

**原文建议：** `export VLLM_ROCM_USE_AITER=1`，让 vLLM 自己选。默认（MHA → `ROCM_AITER_FA`，MLA → `ROCM_AITER_MLA`）在测过的负载上就是赢家。

这才像原生 AMD 优化：不是搬过来的，是为这块硅做的。三路是故意的——软件层把负载切开，每条路去叫 AITER 原语。好 debug，跨代卡可移植，对着生产里的混合批。

## 致谢

**AMD：** Hattie Wu、Yi Gan、Zejun Chen、Carlus Huang、Lingpeng Jin、Peng Sun，以及 AITER 团队。

**Embedded LLM：** Pin Siang Tan、Tun Jian Tan、Jun Kang Chow，以及 Embedded LLM 团队。

## 资料

- [AITER Library (AMD)](https://github.com/ROCm/aiter)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-235B Model](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
- [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

## 免责声明

AMD AI Framework 团队于 **2026-01-29** 在 Instinct MI300X、MI325X、MI355X 上测推理 TPS。

**硬件配置**

- **MI300X：** AMD EPYC 9654 96-Core Processor 服务器，8× AMD Instinct MI300X（192GB，750W），Supermicro AS-8125GS-TNMR2，NPS1（每 socket 1 个 NUMA），2.2TiB（24 DIMM，4800 mts，96 GiB/DIMM），BIOS 3.2
- **MI325X：** AMD EPYC 9575F 64-Core Processor 服务器，8× AMD Instinct MI325X（256GB，1000W），Supermicro AS-8125GS-TNMR2，NPS1，2.2TiB（24 DIMM，4800 mts，96 GiB/DIMM），BIOS 3.2
- **MI355X：** AMD EPYC 9575F 64-Core Processor 服务器，8× AMD Instinct MI355X（288GB，1400W），Supermicro AS-8125GS-TNMR2，NPS1，2.2TiB（24 DIMM，4800 mts，96 GiB/DIMM），BIOS 3.2

**软件配置**

Ubuntu 22.04 LTS，Linux kernel **5.15.0-116-generic**，**ROCm 7.0**，PyTorch **2.9.0a0**，vLLM **0.14.0rc2**（2026-01-15）。

服务器厂商配置可能不同。成绩会随配置、软件、vLLM 版本、驱动和优化代际变化。学习笔记，不是 SLA。
