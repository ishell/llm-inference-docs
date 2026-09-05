---
source: https://vllm.ai/blog/2026-04-24-deepseek-v4
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# DeepSeek V4 在 vLLM 里怎么侍候

英文对照：[en/vllm/blog/architecture/deepseek-v4.md](../../../../en/vllm/blog/architecture/deepseek-v4.md)  
原文：https://vllm.ai/blog/2026-04-24-deepseek-v4  
2026-04-24。署名 **vLLM Team**。学习重写，不是官方译本。两兄弟：[`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)（1.6T）、[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)（285B），都声称能撑到 **一百万** token。镜像 `vllm/vllm-openai:deepseekv4-cu130`。这是第一版模型支持，页上写着优化还在路上。

前一截稀疏注意力：[deepseek-v32](deepseek-v32.md)。FP8 KV / attention：[fp8-kvcache](../performance/fp8-kvcache.md)。Wide-EP 走廊：[large-scale](../serving/large-scale.md)。GB200 成绩单：[gb200-wideep](../serving/gb200-wideep.md)。插件门：[plugin-system](plugin-system.md) / [hardware-plugin](hardware-plugin.md)。

**原文 TL;DR：**

- 同一套 1M 上下文注意力实现。可选：FP4 indexer、MTP。`--block-size 256` 是分配器的逻辑尺。
- 四件结构活：K/V 共享（约 **2×**，正确性靠注意力输出上的 **inverse RoPE**）；跨 token 压 KV（`c4a` / `c128a`，约 **4× 到 128×**）；DSA 只看 top-$k$ 压缩位；短滑窗 **128** 作用在未压缩 token 上。
- bf16 KV、1M 上下文：V4 每条序列 **9.62 GiB**，相对 61 层 V3.2 风格栈的 **83.9 GiB** 大约 **8.7×**。线上 indexer **fp4**、attention cache **fp8**，相对这份 bf16 估计再砍大约 **2×**。算术在附录。
- vLLM：逻辑 block 一律 256 个原生位置；压缩机残差当滑窗 KV；五种 cache 收成三个 page-size 桶；kernel fusion + 多 stream。实现 [vllm#40760](https://github.com/vllm-project/vllm/pull/40760)。

原文分节：Running DeepSeek V4 on vLLM（DeepSeek-V4-Pro / DeepSeek-V4-Flash）→ DeepSeek V4's Attention Mechanism Explained → vLLM's Implementation of DeepSeek V4（Keeping the KV Cache Memory Packed：(1) A single logical block size / (2) Compressor state as a sliding window / (3) Unifying page sizes；Keeping the GPU Busy：(1) Kernel Fusion / (2) Multi-stream）→ Planned Work → Acknowledgments → Appendix: The Math behind DeepSeek V4's Attention Mechanism（inverse RoPE / position ranges / $k$ in c4a and c128a / short sliding window / 8.7× savings arithmetic）。

新注意力初读会显得绕，按第一性原理摊开之后，原则并不绕。下文三块：怎么起；注意力从第一性原理怎么长出来；vLLM 侧怎么把 hybrid KV、kernel fusion、disaggregated serving 接住。希望开源社区既能看懂机制，也能看懂当时那些实现选择为什么这样选。

## Running DeepSeek V4 on vLLM

两个模型，同一套能撑到 1M 的注意力实现。页上点名的可选优化：FP4 indexer、MTP。下面的 docker 是单机试跑，不是集群配方。P/D 分离、别的卡型，看 [V4-Pro recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro) 和 [V4-Flash recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)。

### DeepSeek-V4-Pro

**8×B200** 或 **8×B300**：

```bash
docker run --gpus all \
  --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

### DeepSeek-V4-Flash

旗一样，DP 更小。**4×B200** 或 **4×B300**：

```bash
docker run --gpus all \
  --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:deepseekv4-cu130 deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

`--block-size 256` 是下面分配器的逻辑尺，不是随手填的 page。

## DeepSeek V4's Attention Mechanism Explained

长上下文推理两堵墙：

- **KV cache memory growth。** Cache 仍随上下文线性涨。[MLA](https://arxiv.org/abs/2405.04434) 已经比 MHA / MQA 省很多；一百万 token 仍挤不进 GPU 显存。
- **Attention computation cost。** 即便有 [DSA](http://arxiv.org/abs/2512.02556)，matmul 仍是瓶颈。

V4 同时压房子、压算力：

1. **Share key and value vectors**（约 **2×** 显存）。正确性靠注意力输出上的 **inverse RoPE**——代数在附录。
2. **Compress the KV cache across multiple tokens**（约 **4× 到 128×**）。两条：
   - **`c4a`：** 大约压到 1/4。一条压缩 token 是 **8** 条未压缩 token 的加权和，**stride 4**。
   - **`c128a`：** 大约压到 1/128。加权和跨 **128** 条未压缩 token，**stride 128**。
3. **DSA，把计算封顶。** `c4a` 之后，1M 序列仍有 **250k** 条压缩 token。DSA 只看 top-$k$ 压缩位。
4. **Preserving locality: short sliding window。** 窗口 **128**，作用在 **未压缩** token 上，query 在撞上压缩边界之前还能看见身边。

原文用 **13** 个 token 动画 `c4a`（页上还有可悬停的 [interactive](https://vllm.ai/assets/interactive_pages/c4a.html)；本笔记只收静态对照，不嵌 HTML widget）。`c128a` 同一张图，步子更粗。

![c4a animation](../../../../assets/vllm/blog/architecture/deepseek-v4/01-c4a_animation.gif)

**Figure。** `c4a` 注意力动画（学习对照；版权仍归原站）。

**bf16** KV 时，1M 上下文每条序列 **9.62 GiB**——相对 **61** 层 V3.2 风格栈的 **83.9 GiB** 大约 **8.7×**。线上 indexer 用 **fp4**、attention cache 用 **fp8**，相对这份 bf16 估计再砍大约 **2×**。算术和数学解释在附录。

![kv cache comparison](../../../../assets/vllm/blog/architecture/deepseek-v4/02-kv-cache-comparison.svg)

**Figure。** 逐层 KV：DeepSeek V3.2 对 DeepSeek V4。

## vLLM's Implementation of DeepSeek V4

结构上省下来的是真的；接到 serving 路径仍是系统活：

- 跟 V3.2 一样：注意力 kernel 在 Prefill 走 **bfloat16** KV，Decode 走部分 **token-wise fp8**。
- `c4a` 和 `c128a` 混用，还有些层是 **纯滑动窗口、不压缩**。异构类型把 KV 管理撕复杂。
- 一个 batch 里的序列，相对压缩边界可能停在 **不同状态**。
- 原生 **fp4 MoE** 权重要专门接。

点名却略过的架构改动：[Manifold-Constrained Hyper-Connections](http://arxiv.org/abs/2512.24880)，以及 MoE 模块上的几处。原文说那些比注意力好接。

vLLM 两头拧：显存怎么打包，kernel 怎么不饿 GPU。

### Keeping the KV Cache Memory Packed

分配器还要跟 prefix caching、Prefill/Decode 分离、CUDA graphs、整条 serving 路径共事。三招。

#### (1) A single logical block size

层的压缩比是 1/4（`c4a`）、1/128（`c128a`）或 1/1（SWA）。按「凑整的压缩条数」给每层定 block，每层就有自己的 page 布局，分配器要分别算账。

改成：压缩层的逻辑 block 一律 **256 个原生 token 位置**。`c4a` 物理上装 `256 / 4 = 64` 条压缩项，`c128a` 装 `256 / 128 = 2`。申请一块，永远是这条请求上下文里下一段 256 个原生位置，不管哪一层。slot map、调度记账、prefix-hit 都用这把尺，不必按 `compress_ratio` 分叉。

#### (2) Compressor state as a sliding window

每层压缩机给每条请求留一小段滚动残差：C4 是 **8 token**（有重叠）的部分和，C128 是 **128 token**。单独放 side buffer 在隔离环境能跑，一旦碰上栈的其余部分就别扭。

Side buffer 的代价：prefix cache 要在每个可缓存边界快照残差，跟 prefix hash 一起当钥匙，hit 时再还原；disaggregated Prefill 还要第二条传输路径，专门把残差从 Prefill worker 搬到 Decode worker，跟 KV block 并列。单独看每一条都做得完，合在一起又多一条要跨特性维护的状态路径。

vLLM 把压缩机状态当成滑动窗口 KV。不变量一样：每请求固定大小，Decode 往前推，窗外的丢掉或走缓存。登记进 sliding-window KV cache spec，`sliding_window = coff * compress_ratio`（C4 是 **8**，C128 是 **128**），放进同一套 hybrid KV manager 的 SWA 风格 block。

于是几件事共用抽象：

- **Prefix caching。** hit 落在 KV block 边界（上面那把 256 尺）上，边界处的压缩机状态已经是交接姿态。
- **Disaggregated Prefill。** 压缩机状态按 SWA 搬：只传窗内的 block，不必另开残差通道，传输量仍省。
- **CUDA graphs** 和 **MTP** 跟 SWA 同一套接入；元数据仍是压缩机自己的。

#### (3) Unifying page sizes

上面两招还不够。C4 indexer block、`c128a` KV block、`c4a` 压缩机状态 block，**每块字节数**仍不同。各开各的池，跨池碎片又回来了。

page size = `block_size * compress_ratio * per_entry_size`，三个因子都在手里。选好了，五种 cache 收成 **三个** page-size 桶。每个池加载时定一次大小，分配就是查桶。运行时不再切分、不按种类记账、种类之间不互相打碎。

- *Largest bucket：* `c4a` 主 KV、SWA KV、`c4a` 压缩机状态、`c128a` 压缩机状态。
- *Middle bucket：* C4 indexer KV、C4 indexer 压缩机状态。
- *Smallest bucket：* `c128a` 主 KV。

源稿里有一段注释掉的细节：61 层、标准 C4/C128 配比，三种每块大小 **1,728 B**、**8,640 B**、**37,440 B**，都是 FlashMLA **576 B** 对齐的倍数。公开正文没有把「哪种 cache 进哪一档」的表写完。

### Keeping the GPU Busy

FlashMLA、FlashInfer 管注意力和 MoE。这模型仍有许多小的、多半 memory-bound 的 kernel。多一次 launch、多一趟 HBM，Decode 就会饿。

![decode path](../../../../assets/vllm/blog/architecture/deepseek-v4/03-decode-path.svg)

**Figure。** `c4a` Decode 路径：算子图上的融合（彩色框）和多 stream 划分（默认 stream 蓝带，indexer stream 琥珀带）。

#### (1) Kernel Fusion

三处融合，就是图里那些彩色框：

- **Compressor + RMSNorm + RoPE + cache insertion。** 压缩完的 K 立刻 RMSNorm、RoPE、插入后续注意力的 KV cache（主注意力或 indexer）。几乎全是 elementwise，融成一个 kernel。indexer K 与主注意力 K 仍分两个 kernel，好按 head dim 调并行。相对未融合基线大约 **~1.4–3×**。
- **Inverse RoPE + fp8 quant。** 主注意力输出先 inverse RoPE，再进 `o_lora` 的 fp8 batched matmul。融掉背对背的 HBM 往返，算术强度上去，大约 **~2–3×**。
- **Fused Q norm + KV RoPE + K insert。** 主注意力之前，压缩路径和滑窗路径都要插 KV。压缩路径已被第一刀吃掉，剩下是 query 和未压缩 SWA key 上的 elementwise。水平融进一个 kernel，静态 `warpID` 分派：每个 warp 独立盯一个 Q head 或 K head，不必跨 warp 通信。相对朴素未融合 **10–20×**。

V3.2 那几刀也接着用：Q RoPE + quant + weight multiply，以及注意力开头 QK 投影之后的 QK norm 水平融合。

#### (2) Multi-stream

主注意力之前的活能拆成三截：indexer、主 KV 压缩、SWA token 插入。初始投影之后几乎独立，于是叠到不同 CUDA stream。蓝带默认 stream，琥珀带 indexer stream。

- **`c128a` 层**（没有 indexer）：主 KV 压缩和 SWA 插入并行。
- **`c4a` 层：** 整条 indexer 管线自己一条 stream，与主 KV 压缩、SWA 插入并行（后两者彼此仍串行）。

低 batch 时端到端延迟大约少 **5–6%**——Decode 路径少在饿 GPU。Decode 上仍用 CUDA graphs 砍 launch，跟别的模型一样。

实现：[vllm#40760](https://github.com/vllm-project/vllm/pull/40760)。

## Planned Work

页上还在做：

- DeepGEMM MegaMoE kernel
- Paged Prefill kernel

当前实现对着 NVIDIA **Hopper** 和 **Blackwell**；配方在 [recipe 站](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)。硬件厂商可以走插件门自己接。点名已经独立支持 V4 的：[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[vllm-mlu](https://github.com/Cambricon/vllm-mlu)。

## Acknowledgments

感谢 DeepSeek 开源 V4，也感谢 DeepSeek 管理层对 vLLM 的信任。模型支持记在 [Inferact Inc.](https://inferact.ai/) 名下。

## Appendix: The Math behind DeepSeek V4's Attention Mechanism

### Why inverse RoPE is needed when key and value are shared

位置 $i$ 的 query，[RoPE](http://arxiv.org/abs/2104.09864) 之后是 $<q_i, i> = R(i)q_i$，$R(i)$ 是位置 $i$ 的旋转矩阵（转角由位置参数化）。常用性质：

- $R(i)R(j) = R(i+j)$
- $R(i)^{-1} = R(i)^T = R(-i)$
- $R(i)$ 正交：$R(i)R(i)^T = I$

位置 $j_1, j_2, j_p, \ldots, j_n$ 的 key，RoPE 之后：$<k_{j_1}, j_1> = R(j_1)k_{j_1}$，$<k_{j_2}, j_2> = R(j_2)k_{j_2}$，$\ldots$，$<k_{j_p}, j_p> = R(j_p)k_{j_p}$，$\ldots$，$<k_{j_n}, j_n> = R(j_n)k_{j_n}$。

value 通常 **不做** RoPE：$<v_{j_1}, j_1> = v_{j_1}$，$\ldots$，$<v_{j_p}, j_p> = v_{j_p}$，$\ldots$，$<v_{j_n}, j_n> = v_{j_n}$。

注意力输出（略去缩放等细节）：

$$
a_i = \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <v_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p - i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r - i)k_{j_r})} v_{j_p}
$$

平移不变：位置相关的因子 $R(j_p -i)$、$R(j_r -i)$ 只看 query 与 key 的相对位置。query 和 key 一起挪同一段，输出不变。

K/V **共享** 之后：

$$
a_i = \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <k_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p -i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r -i)k_{j_r})} R(j_p) k_{j_p}
$$

$R(j_p)$ 把 **绝对位置** 漏进输出。这不是想要的。修法：对注意力输出做 inverse RoPE：

$$
R(-i) a_i = R(-i) \sum_{p=1}^n \frac{\exp(<q_i, i>^T <k_{j_p}, j_p>)}{\sum_{r=1}^n \exp(<q_i, i>^T <k_{j_r}, j_r>)} <k_{j_p}, j_p> = \sum_{p=1}^n \frac{\exp(q_i^T R(j_p -i)k_{j_p})}{\sum_{r=1}^n \exp(q_i^T R(j_r -i)k_{j_r})} R(j_p -i) k_{j_p}
$$

只剩 $R(j_p -i)$，平移不变回来。相关讨论：https://kexue.fm/archives/10862。

### Implementation details: exact position ranges and causality conditions

处理压缩 KV 时要小心。每个压缩下标 $j$：先把固定邻域里的原始 token 合成一条，再用这条压缩 token 的 **锚点位置** 做一次 RoPE，然后写入 KV cache。

- **`c4a`：** 第 $j$ 条压缩 token 是 $[4j - 4, 4j + 3]$ 的加权和（$j$ 从 0 起；负下标当值为 0 的 token）。RoPE 位置：$4j$。
- **`c128a`：** 加权和落在 $[128j, 128j + 127]$（$j$ 从 0 起）。RoPE 位置：$128j$。

因果：位置 $i$ 的 query 只能看见 $[0, i]$ 里的信息。所以 query $i$ 对压缩下标 $j$：$i \ge 4j + 3$（`c4a`）或 $i \ge 128j + 127$（`c128a`）。

### Implementation details: The exact value of $k$ in c4a and c128a

V4 默认：`c4a` 的 $k = 512$，`c128a` 的 $k = 8192$。（V3.2 默认 $k = 2048$。）

`c128a` 压得更狠：1M 上下文最多 **8k** 条压缩 token。这点量够做压缩 token 上的 **全** 注意力。实现上仍把它写成 top-$k$ = 8192 的稀疏注意力。

### Implementation details: why the short sliding window is needed

`c128a` 下，位置 **100** 的 query 碰不到任何压缩 token：第一条压缩 token 装着 $0$–$127$，因果又不允许看见 100 之后。短滑窗让它去看 $[0, 100]$ 的未压缩 token，局部信息还在。

### Arithmetic behind the estimates for the 8.7× savings

序列 1M（`1,048,576` token）。

DeepSeek V3.2、bf16 KV：

- 每层每 token 的 MLA cache：$(512 + 64) \times 2 = 1152$ bytes。
- 每层每 token 的 indexer cache：$128 \times 2 = 256$ bytes。
- 合计：$1152 + 256 = 1408$ bytes。
- 1,048,576 token：$1{,}048{,}576 \times 1408 \approx 1.375$ GiB / 层。
- **61** 层：约 **83.9 GiB**。

DeepSeek V4，61 层，bf16 KV：

- 每条共享 KV：$512 \times 2 = 1024$ bytes。
- 每条 `c4a` indexer：$128 \times 2 = 256$ bytes。
- `c4a` 层：共享 KV $(128 + 1{,}048{,}576 / 4) \times 1024$，加上 indexer $(1{,}048{,}576 / 4) \times 256$，约 **320.1 MiB**。
- `c128a` 层：$(128 + 1{,}048{,}576 / 128) \times 1024 \approx 8.1$ MiB。
- **30** 层 `c4a` + **31** 层 `c128a`：约 **9.62 GiB**。
