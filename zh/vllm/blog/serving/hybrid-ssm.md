---
source: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Hybrid SSM 的 P/D 分离：两种记忆，同一根 RDMA 管子

英文对照：[en/vllm/blog/serving/hybrid-ssm.md](../../../../en/vllm/blog/serving/hybrid-ssm.md)  
原文：https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg  
2026-04-21。Nicolò Lucchesi、Zhanqiu Hu（Red Hat）与 vLLM 团队。`vllm>=0.20.0`。学习译文，不是官方译本。

Nemotron-H 一类架构：Mamba 式 SSM 层和满注意力（FA）层插花。线性时间的状态空间，碰上注意力那点表达力。vLLM 给标准 Transformer 已经备好 [NIXL KV connector](https://blog.vllm.ai/2025/01/27/v0-disagg-prefill.html)：Prefill 实例算完 KV 块，Decode 实例用 RDMA 把它们拉走，不必再算一遍。混合模型却不是那套制服——FA 和 SSM 存的根本不是同一种状态，布局不同、尺寸不同，而 block manager 和 NIXL connector 当初只认一种均匀的 KV 格式。

这篇讲他们怎样把 NIXL connector 扩到 hybrid SSM-FA 的分离模式。三件关键想法：

- **双描述符（dual descriptor views）**：同一片物理内存，两套 NIXL 块描述符，偏移和长度各写各的——一套给 FA，一套给 SSM。
- **物理块 vs 逻辑块（physical/logical block bridging）**：block manager 看见的逻辑块，对不上 attention kernel 要的物理块。
- **Conv 的三个描述符（3-descriptor conv transfer）**：把 Mamba conv 状态拆开，异构 tensor-parallel 也能零拷贝，发送端不必重排。

没有 SSM 层时，旧的 Transformer 路径一步都不改。纯加法，只在模型里真有 SSM 时才亮。底座是 [HMA 对 NIXL 的接口](https://github.com/vllm-project/vllm/pull/35758)，落地跨几份 PR：

- [#36687](https://github.com/vllm-project/vllm/pull/36687) — 双描述符，以及同构 TP 下的 hybrid SSM-FA
- [#37416](https://github.com/vllm-project/vllm/pull/37416) — Mamba kernel 的 DS conv 布局
- [#37635](https://github.com/vllm-project/vllm/pull/37635) — 异构 TP 的 3-descriptor conv 搬运
- [#37310](https://github.com/vllm-project/vllm/pull/37310) — Mamba P/D 的 N-1 Prefill

本地图（原文版权仍归原站；学习对照用）：

![transfer volume vs isl](../../../../assets/vllm/blog/serving/hybrid-ssm/01-transfer-volume-vs-isl.png)

![disagg vs colocated](../../../../assets/vllm/blog/serving/hybrid-ssm/02-disagg-vs-colocated.png)

## 背景：稠密 Transformer 的 NIXL KV 搬运

先把标准 Transformer 的 NIXL 分离 P/D 走一遍。四段：

1. **登记内存 region。** 每个 worker 把自己的 KV 张量登记给 NIXL，好走 RDMA。
2. **按块做描述符。** 每个已登记 region 上，为每一块写 `(address, length, device_id)`。搬运单位是块，不是整片 region。
3. **握手。** Decode（D）第一次要向 Prefill（P）拉数据时，两边交换 metadata：agent handle、块数、块长度。每一对 P–D **只做一次**。
4. **搬运。** 调度器告诉 D 该从 P 拉哪些块。D 把 `block_id → descriptor_id`，发 RDMA READ，轮询完成。

标准模型：`M` 个已登记 region × `N` 块，名单长这样：

```text
+----------------------------------+
| Region 0: desc_0 ... desc_{N-1}  |
| Region 1: desc_0 ... desc_{N-1}  |
| ...                              |
| Region M: desc_0 ... desc_{N-1}  |
+----------------------------------+
```

Region `r` 的块 `b` → 描述符下标 `r * N + b`。

混合模型把这份统一方案撕开：FA 和 SSM 要的描述符大小不同，块数也不同。

## 难点：FA 和 SSM 根本不是同一种记忆

稠密 Transformer 里，每一层 KV 形状一样：`[num_blocks, 2, block_size, num_kv_heads, head_dim]`（或某种 layout 变体）。块大小、页大小、块数，层与层共用。

Mamba 存的是整段历史的**塌缩**：一份 conv 状态，一份时间维 SSM 状态——没有 token 轴：

```text
Conv state:  (conv_dim, state_len)              例如 (3072, 3)      -- bf16
SSM state:   (num_heads, head_dim, state_size)  例如 (32, 64, 128)  -- fp32
```

这些状态是整段序列的固定大小摘要。对 SSM，`block_size` 等于 **1**：每一块是一份完整快照，不是一捆 per-token 向量。记住：**块仍是唯一的搬运单位。**

### HMA 的共享张量

vLLM 的 Hybrid Memory Allocator（HMA）按类型分组：FA 一层堆、SSM 一层堆。再让各组**同一位置的层共用一块物理张量**。块可以互换，省。代价是：同一张量，FA 看成 K/V，Mamba 看成 conv + SSM + padding。Nemotron-H 一类长这样：

```text
                KV Cache Tensor (shared via HMA pooling)
                 /                        \
                /                          \
     Attention (FA) View              Mamba View
              |                            |
    +-----------------------+    +-----------------------+
    | Block 0               |    | Block 0               |
    |   Key     |  Value    |    |  Conv |    SSM  |[pad]|
    | Block 1               |    | Block 1               |
    |   Key     |  Value    |    |  Conv |    SSM  |[pad]|
    |  ...                  |    |  ...                  |
    +-----------------------+    +-----------------------+
```

页的字节数对不上。FA 页大约 `block_size * num_kv_heads * head_dim`（再 `*2` 给 K/V）；SSM 页是 `conv_state_bytes + ssm_state_bytes`。HMA 把 FA 的 `block_size` **抬到不小于 Mamba**，再给 Mamba 行垫 `[pad]`，让两边页的字节数对齐，共享张量才站得住。

**NIXL 的麻烦：** 一份均匀的 `(address, length)` 名单无法同时索引两种视图。异构 TP（D 的 TP ≠ P 的 TP）还要把 K/V（以及 Conv/SSM）登记成**分开的**描述符，才能按 head 切。

FA 的块 `b`：`base + b * page_size`，长度 `fa_block_len`。同一块的 Mamba：同一基址，长度却是 `conv_size` 或 `ssm_size`。两套长度，不是同一把尺子。

## 双描述符

办法：同一片物理内存登记**两套**描述符名单，拼在一个 NIXL transfer handle 下面：

```text
+------------------------------------------------------+
|  FA descriptors (M regions x N_phys blocks)          |
|                                                      |
|  Region 0                                            |
|    FA_desc_K[0], FA_desc_K[1], ... FA_desc_K[N-1]    |
|    FA_desc_V[0], FA_desc_V[1], ... FA_desc_V[N-1]    |
|  Region 1                                            |
|    ...                                               |
|  Region M                                            |
|    ...                                               |
|                                                      |   ^
|  --------------------------------------------------- |   | num_descs
|                                                      |   v
|  Mamba descriptors (M regions x N_log blocks)        |
|                                                      |
|  Region 0                                            |
|    Mamba_desc_x[0]   ... Mamba_desc_x[N-1]           |
|    Mamba_desc_B[0]   ... Mamba_desc_B[N-1]           |
|    Mamba_desc_C[0]   ... Mamba_desc_C[N-1]           |
|    Mamba_desc_SSM[0] ... Mamba_desc_SSM[N-1]         |
|  Region 1                                            |
|    ...                                               |
|  Region M                                            |
|    ...                                               |
+------------------------------------------------------+
```

`N_phys` / `N_log` 分别是物理块和逻辑块。可以先假定 `N_phys = N_log = N`，下一节才是它们不同的时候。

上图 Mamba 段已经按 x、B、C 子投影拆开（见「Conv 的三个描述符」）。同构 TP 时，这些子 region 收成两份：Conv、SSM。

FA 占名单前 `num_descs = M * N_phys` 个槽；Mamba 紧接其后。块 ID 映射：

```python
if is_fa_group:
    desc_id = region_id * N_phys + block_id
else:  # mamba group
    desc_id = mamba_region_id * N_log + block_id + num_descs
```

## 物理块 vs 逻辑块

第二件麻烦来自 attention kernel。FlashInfer 一类要固定**物理**块（例如 **16 token**），和用户设的、或 HMA 算出来的**逻辑**块可能对不上。

标准模型用一个比例就够：

```text
physical_blocks = logical_blocks * ratio
ratio = logical_block_size / kernel_block_size
```

混合模型里，这个比例**只作用在 FA 上**。SSM 没有 token 维可拆，始终直接用 `logical_blocks`。于是描述符名单两段的块数不同：

```text
FA section:    M regions * N_phys blocks    (N_phys = N_logical * ratio)
Mamba section: M regions * N_logical blocks
```

记在 `_physical_blocks_per_logical`，**按引擎**算——P 和 D 的 TP 不同时，两边比例可以不同。`_get_block_descs_ids` 按组选 stride：FA 一组，Mamba 另一组。

## Conv 的三个描述符

同构 TP（P 和 D 的 `--tensor-parallel-size` 相同）：每个 D rank 从对应的 P rank 整块读 conv + SSM。直接。

异构就难。例子 `P_TP=1, D_TP=4`：四个 D worker 各自要自己那一瓣。SSM 时间状态按 **heads**（第一轴）切，好切。Conv 却是：

```text
Conv state = [x | B | C]     其中 x、B、C 是子投影
              ^   ^   ^
              |   |   |
     intermediate_size / TP   groups_ss / TP   groups_ss / TP
```

标准 **SD** 布局 `(state_len, dim)` 把这些子投影**交错**排在内存里。某个 D 只要自己那份 `x`，就得捞非连续字节——零拷贝 RDMA 做不到。

### DS 布局

他们要求 conv 走 **DS** 布局 `(dim, state_len)`，环境变量 `VLLM_SSM_CONV_STATE_LAYOUT=DS`。这一布局里，每个子投影自己连续：

```text
DS layout within one page:

|--- x (x_bytes) ---|--- B (b_bytes) ---|--- C (b_bytes) ---|--- SSM ---|
```

每个 D rank 对 `x`、`B`、`C` 做三次连续 RDMA 读——所以叫「3-descriptor transfer」。NIXL READ 仍只发**一次**。

异构 TP 下，`remote_conv_offsets` 按 TP 比算出 D 的切片落在 P 页的哪一段。于是每层 Mamba **4** 份描述符 region（x、B、C、SSM），同构时是 **2** 份（Conv、SSM）。名单更长；RDMA 本身仍是连续读。

**两边 GPU 都不另开 in-memory staging buffer。**  
**两边都不必重排数据。**

同构混跑里，他们还没量到 DS 布局对 kernel 有可察觉的退步；以后可能会让 DS 成为默认。

### 零额外开销：不 Staging，不 Permute

更简单的路是：把整份 conv 寄到每一个 D，再在本地 permute / slice。Mamba 他们故意不走：

- **没有 staging buffer。** 在 D 上 permute，就要为每个 D worker 分配一块和 **P 的整份 conv** 一样大的临时缓冲。Nemotron-H 上，每块 conv 已经是 `3 * 3072 * 2` 字节 bf16。再乘上千块、所有 Mamba 层——那是从 KV 房间里抠走的。
- **没有事后重排。** DS 布局下，每个 D 只读自己要的字节，直接落进 KV 的最终位置。没有事后 permute kernel。搬运结束，状态立刻能用。
- **只搬自己那一份。** 每个 D 只搬 `1/TP` 的 conv，不是整份。`D_TP=4` 就是每 rank 少 4 倍。
- **跳过 HMA padding。** HMA 给 SSM 页垫过，好对齐 FA 页。Mamba 描述符按真正的 `conv_bytes + ssm_bytes` 计，不是垫过的页。线上从不搬 padding。FA 页比裸 SSM 大很多时，每块都能少搬一截。

Figure 1 在 Nemotron Super **120B**、TP=4、FA `block_size=4224`（HMA 定的）上核对这件事。每个 KV dtype（bf16 和 fp8）对比两条解析基线：**Naive** = 把垫过的 Mamba 整页都传；**Optimal** = 只传真正的 conv + SSM，跳过 HMA padding 和辅助缓冲。NIXL 报的实测字节贴着 Optimal。

**fp8：** FA 页每元素 1 字节对 2 字节，这种配置下 padding 可以忽略。  
**bf16：** 大约少搬 **50 MB** 不该搬的——**每条请求**。

Mamba 状态是**每条请求一份固定快照**，传输量跟着 FA 块数走；ISL 变长，曲线跟着 FA 走。

**图注（原文）。** Figure 1：Nemotron Super 120B（TP=4，FA `block_size=4224`）的 P→D 传输量对 input sequence length。Naive / Optimal 按模型的页大小和块数解析算出来；Measured 是分离 P/D serving 时 NIXL 报的实测字节。他们的 Optimal 路径消掉 HMA padding，实测贴着这条线。

## 合在一起：Nemotron-H 走一遍

具体例子：侍候 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`，P/D 分离，**TP=2**。

**模型结构。** 一共 **52** 层，Mamba / FA 交替。HMA 收成 **5** 组（**4** 组 Mamba，**1** 组 FA）。pooling 之后 **6** 块共享 KV 张量。

**KV 布局：**

```text
FA layers:    [num_blocks, 2, block_size=400, 4, 128]   # K/V，HMA 抬过的 block_size
SSM layers:   [num_blocks, 3, 3072]  (conv)  +  [num_blocks, 48, 64, 128]  (ssm)
```

HMA 把两边页的字节数垫齐。Kernel（FlashInfer / FlashAttention）还可能再切 FA 块，于是出现物理 / 逻辑比例。

**描述符登记：**

1. 6 块共享张量照旧登记成 NIXL memory region（和稠密模型一样）。
2. FA 描述符覆盖全部 6 个 region × `N_phys` 块，K 和 V 分开索引。
3. 后面接 Mamba：6 个 region × `N_logical` 块，每层 4 个子 region（x、B、C、SSM），给 3-descriptor 搬运。

**搬运流程：**

1. P 做完 Prefill。调度器按组交出块 ID：`[[fa_block_ids], [mamba_block_ids_g0], [mamba_block_ids_g1], ...]`。
2. D 收到块 ID，映射成描述符下标：FA 用标准的 `region * N + block_id`；Mamba 加 `num_descs` 偏移，stride 用 `N_logical`。
3. D 发一次 `make_prepped_xfer` READ，FA 和 Mamba 描述符一起带上，然后轮询。
4. 完成后 D 通知 P，P 可以放块。

从 D 的角度看，整次搬运是**一次**异步操作。没有中间缓冲，没有重排。

## 成绩

他们在 **8× H200**、NVLink 上对比分离 P/D 和混跑。模型：`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`——120B LatentMoE，Mamba2 与满注意力插花。

- **混跑基线：** 单实例，TP=8，8 张卡全用。
- **分离 P/D：** 1 个 Prefill（TP=4，4 卡）+ 1 个 Decode（TP=4，4 卡），总卡数相同。

并发从 **8** 扫到 **256**。纵轴是每 GPU 的 output throughput，横轴是每用户的 output token rate（他们叫 *Interactivity*）。负载用 ShareGPT。

所有 run 用很高的 warmup，好让 KV「搅乱」——刚启动时块碰巧连续，会有一段虚高；长跑不是那样。也可以核对：扫完整份数据时，metrics 里的描述符数量应保持常数。Prefix-caching **关**。

**图注（原文）。** Figure 2：hybrid SSM 模型上，分离 P/D 对混跑。跨并发的吞吐–延迟 Pareto。Prefix-caching disabled。

高 batch 时，分离 Pareto 压过混跑——和稠密 Transformer 的 P/D 是同一句。Decode 不再被 Prefill 打断，batch 能更大，高并发下每 GPU 的 output tok/s 明显更高。

## 怎么开

hybrid SSM 模型跑分离 P/D：

```bash
# Prefill instance
VLLM_SSM_CONV_STATE_LAYOUT=DS vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --max-model-len 8192 \
    --block-size 128 \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

`VLLM_SSM_CONV_STATE_LAYOUT=DS` **异构 TP 必须**；同构不是必须。

## 当时的边界与以后

- **Mamba1：** 三描述符 conv **只支持 Mamba2**。Mamba1 时间形状 `(intermediate_size // tp, state_size)` 还原不出 conv 分解需要的 `intermediate_size`。**GDN**（Qwen3.5+）写在分离 [roadmap](https://github.com/vllm-project/vllm/issues/33702) 上。
- **投机解码：** SSM 状态搬运和投机解码当时还没广泛验证。
- **HMA 下 P/D 块大小不同：** `block_size_ratio > 1` **当时还不支持**（HMA 开着的时候）。

## 致谢

Thomas Parnell（IBM Research）、Roi Koren（NVIDIA）。

Router 那篇的 P/D 假定记忆长得一样。混合模型把「一块」拆成两种方言——管子还是 NIXL，词典要两本。邻居：[Router](router.md)、[Mooncake](mooncake.md)、[MORI-IO](moriio.md)、[大规模 serving](large-scale.md)。
