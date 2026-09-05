---
source: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# MiniMax M3：1M 上下文靠 MSA 选 128-token 块，不是满 attention

英文对照：[en/vllm/blog/serving/minimax-m3.md](../../../../en/vllm/blog/serving/minimax-m3.md)  
原文：https://vllm.ai/blog/2026-06-12-minimax-m3-vllm  
2026-06-12。署名 **vLLM Team**。权重：[`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3)（BF16）、[`MiniMaxAI/MiniMax-M3-MXFP8`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8)。验证过 H200 / GB200 / B300；AMD MI350 / MI300。更早的 Lightning Attention 亲戚：[minimax-m1.md](minimax-m1.md)。后来 Omni 栈：[minimax-h3.md](minimax-h3.md)。投机路径：[spec-decode.md](../performance/spec-decode.md)、[p-eagle.md](../performance/p-eagle.md)。缓存 / P/D：[large-scale.md](large-scale.md)、[kv-offload.md](kv-offload.md)、[mooncake.md](mooncake.md)、[shm-ipc.md](shm-ipc.md)。菜谱在 [recipes.vllm.ai](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)。**引擎骨架没换**；换的是 MSA backend、parser、MXFP8 MoE、EAGLE3 配方。

MiniMax M3 对着已经变日常的负载：百万 token 上下文、原生多模态推理、编码和 agent、工具、可控 thinking。难的不是把权重载进来，是让 MiniMax Sparse Attention、多模态预处理、MXFP8 MoE、EAGLE3、前缀缓存、部署配方在同一台引擎里一起活。

本地图（原文版权仍归原站；学习对照用）：

![hero minimax m3 vllm](../../../../assets/vllm/blog/serving/minimax-m3/01-hero-minimax-m3-vllm.svg)

![msa 1m context](../../../../assets/vllm/blog/serving/minimax-m3/02-msa-1m-context.svg)

![msa backend dispatch](../../../../assets/vllm/blog/serving/minimax-m3/03-msa-backend-dispatch.svg)

![multimodal request path](../../../../assets/vllm/blog/serving/minimax-m3/04-multimodal-request-path.svg)

![kv block major prefill](../../../../assets/vllm/blog/serving/minimax-m3/05-kv-block-major-prefill.svg)

![validation dashboard](../../../../assets/vllm/blog/serving/minimax-m3/06-validation-dashboard.svg)

**Figure 1。** Day-0：长上下文、多模态、稀疏 attention 进 vLLM。

## TL;DR

- **模型族：** BF16 和 MXFP8 MiniMax M3；1M 上下文看硬件和配方。
- **架构核心：** MiniMax Sparse Attention（MSA）——稠密 / 稀疏杂交。给 128-token KV 块打分，每 query、每 KV group 选 top，再在选中的块上做 GQA。
- **Serving 栈：** `minimax_m3` 工具与 reasoning parser、thinking-mode、纯文本和多模态、TP/EP、前缀缓存、chunked Prefill、EAGLE3、Docker。
- **投机解码：** Day-0 EAGLE3，draft [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3)。
- **RL 后训练：** Day-0 MiniMax M3 GRPO 在 [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL)，生成后端是 vLLM。
- **性能工作：** MSA Prefill/Decode kernel、indexer-score 与 top-k、fused QKNorm + RoPE + KV insert、GemmaNorm 和量化路径、MXFP8 MoE backend。
- **Roadmap：** FP8 indexer/KV-cache、TRTLLM-Gen MoE、更宽的分离配方、context-parallel 长 Prefill、多模态网关。

## MiniMax M3 Support Matrix

| 能力 | MiniMax M3 加了什么 | vLLM 怎么接 |
| --- | --- | --- |
| 1M-token 上下文 | 长文本、代码、agent trace、文档 | `--max-model-len`、block-size 128 配方、前缀缓存、chunked Prefill、MSA kernel |
| MiniMax Sparse Attention | 在选中的 128-token KV 块上做 block-sparse GQA | hybrid attention backend、indexer-score、top-k、稀疏 GQA Prefill/Decode |
| MXFP8 权重 | 大规模 MoE serving | Blackwell 类 DeepGEMM MXFP8 MoE；Hopper 类 Marlin MXFP8 |
| 原生多模态 | 图、视频和文本一起 | 模型专用多模态预处理 |
| 工具与 reasoning 输出 | agent、可控 thinking | `minimax_m3` tool / reasoning parser，`thinking_mode` |
| EAGLE3 投机解码 | draft 加速 | Day-0 EAGLE3 配方 |

## 快速开始

NVIDIA 上 MSA 走 **默认** attention backend。视觉编码器：`--mm-encoder-attn-backend FLASHINFER`，共享内存 processor cache，数据并行 encoder。

Blackwell 类节点上的 MXFP8：

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

BF16：同样的旗，模型换成 `MiniMaxAI/MiniMax-M3`。精确配方跟加速器、dtype、上下文、流量形状、以及你要吞吐、延迟还是最大上下文走。完整 NVIDIA / AMD 菜谱：[vLLM recipe for MiniMax M3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)。

### AMD ROCm

MSA 走 Triton：`--attention-backend TRITON_ATTN`。视觉：`--mm-encoder-attn-backend ROCM_AITER_FA`，shm processor cache，数据并行 encoder。MI350 Series / MI300 Series 验过。

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

BF16：同样的旗，`MiniMaxAI/MiniMax-M3`。

### 真正要紧的旋钮

`--block-size 128` 必须对齐 MSA 的稀疏粒度。`--max-model-len` 是对外宣称的上下文，也是 KV 规划。`--tensor-parallel-size` 和 `--enable-expert-parallel` 切开 attention、投影、MoE 专家。agent 流量请打开 `minimax_m3` parser。长上下文配方要写清楚：前缀缓存、chunked Prefill、EAGLE3、多模态预处理，这一套开没开。

### EAGLE3 投机解码

Draft：[`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3)。加上：

```bash
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}'
```

`num_speculative_tokens=3` 是保守起点。生产要拿 acceptance rate、TPOT、吞吐、目标延迟对着流量再拧。

### Thinking Mode

`thinking_mode` 经 `chat_template_kwargs` 传：`"enabled"`、`"disabled"`、`"adaptive"`。

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model = client.models.list().data[0].id
messages = [{"role": "user", "content": "Explain MiniMax Sparse Attention."}]

for mode in ["enabled", "disabled", "adaptive"]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"chat_template_kwargs": {"thinking_mode": mode}},
    )
    print(mode, response.choices[0].message.content)
```

## 模型新在哪

### 1M 上下文靠 MiniMax Sparse Attention

不是每个 query 对整份 KV 做稠密 attention。MSA 用一条 index 路径给 KV 块打分，挑最值得读的。默认粒度：**128-token** KV 块。选中的块在一个 GQA group 里共享。

每个 query token 三步：

1. 小 index head 给候选 KV 块打分。
2. 选 top，叠上配置里的块规则。
3. 只在选中的 KV 块上做 online-softmax attention。

1M 上下文能伺候，靠的是这件事。

**Figure 2。** MSA 把局部和全局都留着，从 1M 历史里挑稀疏的 128-token KV 块。

### MSA 再细一点

两件事：过去哪些块值得读，读完怎么做 attention。Index 路径答第一问（固定 128-token 块）。稀疏 GQA 答第二问。

选中集合不只是学来的 top-k。配置里有 `init_blocks` / `sparse_init_block` 和 `local_blocks` / `sparse_local_block`。当前配方：**`init_blocks=0`**，**`local_blocks=1`**。硬规则：query 附近的 local-window 块；剩下的名额给 indexer 打分的 top-k。正确性细节：末尾残块要 mask；块内因果边界要守；已经在 top-k 里的 local 块不要算两遍；一个 batch 里各请求的合法块范围可以不同。

### 原生多模态

不是纯文本权重外挂一个 sidecar。图、视频变成 patch tensor 和 grid 元数据，交给模型，别从生成里偷 GPU 时间。纯文本、工具、reasoning、多模态走同一张 serving 面。

### MXFP8 MoE 权重

验证：Blackwell 类 DeepGEMM MXFP8 MoE；Hopper 类 Marlin MXFP8。

## vLLM 怎么接

Hybrid：有的层走稠密 attention，稀疏层走 MiniMax MSA backend。分界藏在模型和 attention backend 后面——调度、cache 分配、batch、前缀缓存、serving 从外面看还是熟的。背景：[Anatomy of vLLM](../architecture/anatomy.md)。

### MiniMax Sparse Attention Backend

两份活。

先造稀疏元数据：indexer 给 KV 块打分，套选择规则，吐 top-k block ID。M3 的稀疏单位，就是 cache manager 已经认识的那种 128-token 页。

再在这些块上做 attention。Prefill 和 Decode 形状不同：

- **Prefill indexer-score：** Triton 打分和 top-k。
- **Prefill 稀疏 GQA：** Triton 和 [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的 CuTe/SM100。CuTe 把 query-to-block 翻成 K-major CSR，KV 块好复用。
- **Decode indexer-score：** split 风格扫候选、打分、合并 top-k。
- **Decode 稀疏 GQA：** GQA Decode kernel 读选中的页，合并 partial。

### Prefill

四个概念阶段：(1) 造 Q、K、V 和 index 投影；(2) 给块打分（按配置 max 或 log-sum-exp）；(3) top-k 加配置规则；(4) 只在选中的 KV 上做稀疏 GQA。

最后一步两种日程。Query-major：每个 query 走自己的选中块。KV-block-major：长 prompt 里很多 query 点同一块时更好——建 K-to-Q 映射，一块 KV 载上来给许多 query 用，再合并。

### Decode

通常每个活跃序列一步一个新 token；batch 里上下文长度可以很乱。更新 cache、打分、local-window、选 top、稀疏 GQA Decode、合并 split。Indexer-score 和 top-k 坐在 **TPOT** 上，不是启动开销。

配置管：块大小、top-k、可选 init 块、local-window 块、index 维、稀疏层 ID、分数类型、以及哪些层只用 index attention 做选择。每个选中的 block ID 必须映回调度器和 cache manager 认识的那份逻辑请求状态。

**Figure 3。** 稠密层走标准 attention；稀疏层走 MiniMax MSA backend。

### KV 布局：存还是普通页，算才稀疏

KV 可以当普通 paged KV 存；稀疏发生在计算路径。Cache manager 保持简单：

- 主 attention KV 和 indexer K cache 分开记账。
- 前缀缓存和 chunked Prefill 在 cache 状态交互验过之后，继续用稳定的 cache 块。
- 分离 / NIXL 式搬运可以把 cache 当 paged 状态；稀疏选择归 attention backend。

### 前缀缓存和 Chunked Prefill

M3 流量常复用长 prompt：代码库、文档、多轮 agent、多模态上下文。1M 请求不该以一整块 Prefill 独占引擎。发布就绪的压力测试：index cache、主 attention KV、稠密 attention 状态、前缀命中、抢占、batch、长上下文 chunk 边界，必须同意同一套 block table。

### 多模态和 Parser

- `--tool-call-parser minimax_m3`
- `--reasoning-parser minimax_m3`
- Chat template 的 `thinking_mode`
- 图、视频预处理

生产：能在进 GPU 之前预处理，就不要拖到 GPU。目标架构是网关：下媒体、解帧、采样视频、缩放归一化、造 patch tensor，把现成 tensor 交给 worker。一条视频在 API 上看着很小，采样和 patch 之后会很大。CPU 重的媒体活放上游，GPU 调度才好讲。

Parser 把模型自己的文本约定变成结构化 API。parser 不对，生成再有用，应用也吃不进去。

**Figure 4。** CPU 侧图/视频预处理应把现成 tensor 交给 worker，GPU 时间留给推理。

## 性能优化

MSA 少了稠密 attention 的活，又多了 indexer-score、选块、稀疏元数据、一串小 kernel。原则：决定读哪几块花的时间，别超过不读全部省下来的。三处：block-major Prefill、瘦的 Decode indexer-score、attention 周围小 elementwise / cache-write 的融合。

### KV-Block-Major Prefill

很多 query token 会点同一块 KV。Query-major 会把同一块从 HBM 搬到片上搬很多遍。CuTe/SM100：K-to-Q CSR，block-major 稀疏 attention，log-sum-exp 合并 partial。长 prompt、带长缓存上下文的 agent 流量，算术强度更好。

**Figure 5。** KV-block-major Prefill 让选中的 KV 块在 query 之间复用，最后再 LSE 归约。

### Decode Indexer-Score Kernel

Indexer 在每个生成 token 的关键路径上：query 侧 index 向量对候选 key 侧向量，每个 128-token 块收成一个分数，local-window，留下 top。专用 kernel，不当成补齐的稠密 GEMM。选中的 KV 在逻辑序列上稀疏，内存里仍像页——除非复用划得来，别把稀疏页摊成大临时稠密张量。

### Decode Kernel 里的投机解码

EAGLE3 核验：一次请求可以核验多个 draft token，MSA Decode 不能假定每请求恰好一个 query token。

核验回退到 Prefill kernel 很贵：Prefill kernel 对着大得多的 token 数调，通常也 **不** 兼容 full CUDA Graph。

Day-0 给 MSA Decode 的 indexer、top-k、稀疏 GQA Decode 加上均匀的 `decode_query_len`。投机核验 token 按 request-major 摊平；每个 query token 映回请求元数据、序列长度、block table、因果位置。EAGLE3 核验留在 Decode 专用的 split-K 路径。同一条路径给均匀投机 Decode batch 做 full CUDA Graph：launch grid 形状稳定，少做 Triton 特化，padding 行显式处理。投机解码只有在 acceptance 没被额外 launch、重编译、cache 状态开销吃掉时，才会改善 TPOT。

### Kernel 融合

- **QKNorm + RoPE + KV insert**，MSA 路径。
- **GemmaNorm 和 AllReduce + Norm**，tensor-parallel 周围。
- **量化路径清理：** `silu_mul_quant_fp8` 和相关 MXFP8/MoE 输入。
- **Router 和 MoE kernel**，给更深的 TRTLLM-Gen 铺路。

Day-0 故意保守：正确性和稳定 cache，压过把每颗 graph / fusion 旋钮都拧开。

### 量化和 KV Cache Dtype

MXFP8 改的是权重和 MoE 执行，不是 KV 的概念结构。「MXFP8 模型」**不等于** 每一份 cache 和中间量都是 MXFP8。Roadmap 里有 FP8 indexer 和 KV-cache，因为 KV 容量直接决定这台机器能伺候多少长上下文和 batched 流量。

### CUDA Graph 和编译

CUDA Graph 对 Decode 值钱，因为 M3 每步多了好几颗小 op。只有路径在 batch 形状、cache 状态、稀疏元数据之间稳定，capture 才有用。先保守，验证熟了再扩覆盖。

## 验证

公开发布前每天转：准确率、吞吐、投机解码、容器能不能用。

1. **功能正确：** 能载、能伺候、能解析工具和 reasoning、纯文本加多模态。
2. **准确率对齐：** kernel、cache、parser、配方改完，benchmark 还跟预期在一块。
3. **Serving 就绪：** 容器带着打算用的 TP/EP/投机解码设置，在目标加速器上能跑。

短任务抓 parser、格式、明显数值问题。长上下文抓 MSA 元数据、前缀缓存、chunked Prefill、KV 布局。投机测试抓普通准确率跑看不到的 acceptance 回退。

B300 上的一份代表快照（工程验证，不是排行；镜像、权重、配方、硬件都会漂）：

| 维度 | 结果 |
| --- | ---: |
| GSM8K strict / flexible | 91.51% / 91.66% |
| ShareGPT @256 吞吐 | 8,530 tok/s |
| ShareGPT @256 TPOT | 56.0 ms |
| Speculative Sonnet TPOT，concurrency 1 / 16 / 64 | 4.51 / 9.04 / 14.36 ms |
| Sonnet 上投机 acceptance | ~67%，mean accept length ~3.0 |

**Figure 6。** 发布候选：准确率、吞吐、投机解码。

## Serving 之外：NeMo RL 后训练

伺候 M3 的同一摊活，[vLLM PR #45381](https://github.com/vllm-project/vllm/pull/45381)，也让 Day-0 后训练成为可能。

[NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) 用 vLLM 当 **非同驻** generation backend 跑 MiniMax M3。短 GRPO 在 BF16 checkpoint 上验过：NeMo AutoModel + expert parallelism + BF16 vLLM 生成。长跑收敛、EP 以外的并行还在验。参考：[NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md)。

## Roadmap

- **FP8 indexer 和 KV-cache** — KV 内存压力、batch 容量、稀疏 attention 准确率。
- **TRTLLM-Gen MoE** — Blackwell 上的 MXFP8 expert。
- **Context parallelism** — 单节点不够时的超长 Prefill。
- **分离 serving** — NIXL 和 Prefill/Decode 配方；见 [Large-Scale Serving](large-scale.md)。
- **Kernel fusion** — MSA 引进的 indexer、top-k、量化、归一化小 kernel。
- **多模态网关** — 图/视频预处理离开 GPU 生成环。

## MiniMax M3 vLLM FAQ

### vLLM 支持 MiniMax M3 吗？

支持。Day-0 覆盖 BF16 和 MXFP8：MSA、模型专用 parser、EAGLE3、多模态预处理、TP/EP 配方、Docker。

### MiniMax Sparse Attention 是什么？

给固定 128-token KV 块打分，按 query 和 GQA group 选最相关的块，叠配置里的 local-window 规则，在这集合上做稀疏 GQA。当前配方：`init_blocks=0`，`local_blocks=1`。

### MXFP8 是不是说 KV cache 也是 MXFP8？

不是。MXFP8 是权重和 MoE 路径。KV-cache dtype 是另一项 serving 决定。原生 KV 存储 vs 量化 KV-cache，是 roadmap。

### 1M 上下文最要紧的设置？

`--block-size 128`，选中的 batch / 上下文形状要有足够显存，配方写清前缀缓存、chunked Prefill、EAGLE3 开没开。默认 vLLM 从模型配置读上下文长度——不必设 `--max-model-len`。显存紧、或不需要整段 1M，可以把它压低。

## 致谢

MiniMax 开源 MiniMax-M3；MiniMax 管理层信任 vLLM。模型支持由 Inferact Inc. 牵头。NVIDIA 和 AMD 出了硬件支持。

## 相关阅读

- [Anatomy of vLLM](../architecture/anatomy.md) — 调度、KV cache、前缀缓存、分布式。
- [投机解码](../performance/spec-decode.md) 和 [P-EAGLE](../performance/p-eagle.md) — draft 路径。
- [大规模 Serving](large-scale.md)、[KV Offload](kv-offload.md)、[Moriio](moriio.md) — 前缀复用、KV 搬运、分离 serving。
- [NeMo RL MiniMax M3 guide](https://github.com/NVIDIA-NeMo/RL/blob/minimax-m3/docs/guides/minimax-m3.md) — GRPO，生成走 vLLM。
