---
source: https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Distributed Layerwise Offload：把 124 GB 的 DiT 挤进 64 GB HBM，再估一条 200B+ 的路

英文对照：[en/vllm/blog/serving/omni-layerwise-offload.md](../../../../en/vllm/blog/serving/omni-layerwise-offload.md)  
原文：https://vllm.ai/blog/2026-08-17-distributed-layerwise-offload  
2026-08-17。署名 **vLLM-Omni Diffusion Team**。DLO 把 DiT 权重切碎、流进来，测过的 **124 GB** Cosmos3-Super 能在 **64 GB** HBM 上跑；往 200B+ 走的是按内存模型外推，**没有**真跑过那档模型。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)、[minimax-h3.md](minimax-h3.md)、[omni-diffusion-cache.md](omni-diffusion-cache.md)。页上的测量是他们的实验合同，不是你的 SLA。

## TL;DR

开箱 DLO + AllGather 快路径：vLLM `0.27.0` 配 Omni `v0.27.0rc1`。meta 设备初始化 + mmap：Cosmos3-Nano DP4 冷启动 cgroup 可见峰值 **178 GB → 47 GB**（−73%）。每 rank 只留 `1/dp_size`，AllGather 在专用流上跟计算重叠。设备上固定 **双缓冲两层**，跟总层数无关。测到的 720p 10s `dist_offload+SP`：峰值 HBM **23.1 → 28.1 GB**（约 22%），空闲 HBM **11.5 → 14.6 GB**（约 27%）。DP 多并发相对单请求 HSDP **3.3×**，大约是理想 4× 的 **83%**。CUDA/NCCL 和昇腾 CANN/HCCL 走 Omni 的平台层。8× B300 上 MiniMax-H3：AllGather 赢 DP1×SP8 和 DP4×SP2；rank-local 赢 DP8×SP1，**183.78 videos/h**、**43.97 Wh/video**。

本地图（原文版权仍归原站；学习对照用）：

![dlo problem overview](../../../../assets/vllm/blog/serving/omni-layerwise-offload/01-dlo-problem-overview.svg)

![mmap loading memory](../../../../assets/vllm/blog/serving/omni-layerwise-offload/02-mmap-loading-memory.svg)

![weight sharding allgather](../../../../assets/vllm/blog/serving/omni-layerwise-offload/03-weight-sharding-allgather.svg)

![dlo pipeline last frame](../../../../assets/vllm/blog/serving/omni-layerwise-offload/04-dlo_pipeline_last_frame.png)

![dlo pipeline](../../../../assets/vllm/blog/serving/omni-layerwise-offload/05-dlo_pipeline.gif)

![hbm nano vs super](../../../../assets/vllm/blog/serving/omni-layerwise-offload/06-hbm-nano-vs-super.svg)

![dp multi concurrency](../../../../assets/vllm/blog/serving/omni-layerwise-offload/07-dp-multi-concurrency.svg)

![ascend memory accounting](../../../../assets/vllm/blog/serving/omni-layerwise-offload/08-ascend-memory-accounting.svg)

![minimax h3 topology policy](../../../../assets/vllm/blog/serving/omni-layerwise-offload/09-minimax-h3-topology-policy.svg)

![minimax h3 multimodal frontiers](../../../../assets/vllm/blog/serving/omni-layerwise-offload/10-minimax-h3-multimodal-frontiers.png)

## Quickstart

两条 AllGather 命令要 Omni `v0.27.0rc1` 以上、vLLM `0.27.0`。`v0.26.0` 上 Cosmos3 的 DLO+DP 会把请求全拒：引擎要求 `supports_request_batch=True` 才允许多请求入队，`Cosmos3OmniDiffusersPipeline` 没声明（[#5953](https://github.com/vllm-project/vllm-omni/issues/5953)）。[#5864](https://github.com/vllm-project/vllm-omni/pull/5864) 给 DLO+AllGather+DP 绕开这条：每个 DP rank 走 pipeline 的单请求前向，引擎从 per-rank 队列收结果。no-AllGather 的 DP 命令 **不**在 #5864 覆盖里；`--dlo-no-use-allgather` 的独立派发当时还开着 [#5911](https://github.com/vllm-project/vllm-omni/pull/5911)。#5864 不改 DLO 切分或 offload 内存机制。下面每一段测量自己报环境。

```bash
# 4× NPU or GPU — Cosmos3-Nano with DP=4
vllm serve /path/to/Cosmos3-Nano --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 4

# 2× devices — Cosmos3-Super (124 GB) with DP=2
vllm serve /path/to/Cosmos3-Super --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 2

# Disable AllGather (each rank loads full weights, no sharding)
vllm serve /path/to/Cosmos3-Nano --omni \
    --enable-distributed-layerwise-offload \
    --data-parallel-size 4 \
    --dlo-no-use-allgather
```

`--dlo-use-allgather` / `--dlo-no-use-allgather` 管切不切（默认切）。关掉时，每 rank 装标准 loader 的 rank-local 张量——纯 DP 就是整模一份，已有的 TP 分片本来就是 rank-local，直接复用。AllGather 同步税盖过省内存时才关。

## 问题：大扩散模型对上 HBM 和主机内存

Cosmos3-Super（**64B**，BF16 **124 GB**）塞不进单张 **64 GB** HBM。旧路两家：**offloader** 从主机流权重，**parallelism** 把常驻活摊到多卡。各有坑。

**Figure 1。** Cosmos3-Super 的对照。HSDP 大约 **31 GB** 权重再加约 **25 GB** 激活和通信缓冲（合计约 **56 GB**），只剩 **8 GB** 余量；DLO 设备上只留两层，主机权重再切。

| Approach | Device HBM | Host memory per rank | Limitation |
|---|:---:|:---:|---|
| HSDP (FSDP2) | model / N | 0 | HBM 填满：64B → 56 GB/card（8 GB 余量） |
| Layerwise offload（纯 DP） | 只两层 | 整模 | N × model_size 主机内存（4 × 124 GB = 496 GB） |
| Tensor Parallel | model / N | 0 | 激活能摊，通信税在 |
| Dist. Layerwise（本文） | 只两层 | model / N | 要 AllGather 同步 |

纯 DP 的主机内存才是杀手：传统 layerwise offload 每 rank 一份整模。4 卡就是 **4 × 124 GB = 496 GB**。TP 部署可能已经 rank-local，按比例少一点。

更糟的是加载：每 rank 各自 `param.data.copy_(loaded_weight)`，RSS 里造 `dp_size` 份私有拷贝。峰值 RSS 按 `O(dp_size × model_size)`，200B、`dp_size=4` 能到 **2 TB**。

## 方案总览

四件一起做：

| Technique | 对着的问题 | 主要收益 |
|---|---|---|
| Meta device + mmap | 加载时 O(dp_size × model) RSS | 冷启动 cgroup 可见峰值 −73% |
| Weight sharding + AllGather | N × model_size 主机内存 | 总共 1× model_size（共享 page cache） |
| Double-buffer prefetch | 权重全上设备 | 任意时刻 HBM 上只有 2 层 |
| DP multi-concurrency | 请求串行 | N 路并行，吞吐 3.3× |

前三件让大模型在内存上站得住。DP 多并发是吞吐优化，吃的是第 2 件已经付过的 AllGather 同步。

## 1. Meta device + mmap 加载

**为什么。** 旧路径每 rank 先 `load_model(load_device="cpu")` 再 `offload_backend.enable()`。`param.data.copy_(loaded_weight)` 造 `dp_size` 份私有拷贝。Cosmos3-Nano DP4 的 cgroup 可见峰值 **178 GB**，模型才 **33 GB**。

**为什么能成。** offloader 把已经建好的 DiT 模块 `to_empty(device="meta")`，再把 meta 参数换成 `safe_open().get_tensor()` 的 mmap 视图，指进 OS page cache。

```python
# distributed_layerwise_backend.py — release existing DiT parameter storage
dit_module.to_empty(device="meta")

# Resolve an HF repo ID, then replace meta parameters with mmap views
model_path = download_weights_from_hf(...)
tensor = safe_open(file_path, framework="pt", device="cpu").get_tensor(ckpt_key)
parent._parameters[name] = Parameter(tensor)  # points to shared page cache
```

所有 rank mmap 同一批 safetensors，OS 只留一份 page cache。Hugging Face repo ID（不是本地路径）先走 `download_weights_from_hf()`，跟 DiffusersPipelineLoader 同一套。

**你得到什么。** Cosmos3-Nano DP4 冷启动 cgroup 可见峰值 **178 GB → 47 GB**（−73%）。178 GB 基线：**132 GB** 私有拷贝 + **33 GB** 共享 page cache + 约 **13 GB** 框架/瞬时。mmap page cache（1× model_size）只读共享，内存紧时 OS 可以部分收回。

**Figure 2。** 那两根柱：四份私有拷贝换成一份共享 mmap。

## 2. 切分权重，AllGather 再拼回来

**为什么。** mmap 完了，layerwise offload 还是会把整模拷进每 rank 的 pinned CPU，给 H2D 用。纯 DP 基线：4 卡 × 33 GB = **132 GB** pinned，跟卡数线性。

**为什么能成。** 每 rank 只存 `1/dp_size`。运行时在专用通信流上 `all_gather_into_tensor` 重建当前层。

```python
# _shard_and_pin: each rank stores only its 1/dp_size shard
shard_size = (total_numel + dp_size - 1) // dp_size  # ceil division
shard = torch.zeros(shard_size, dtype=dtype, device="cpu")
# Copy only the portion within [rank * shard_size, (rank+1) * shard_size)
shard[dst_slice].copy_(mmap_view.flatten()[src_slice])
shard = shard.pin_memory()  # DMA buffer for fast H2D
```

向上取整加零填充，分片等长——`all_gather_into_tensor` 的硬条件。切完之后，原来的 mmap 视图换成零元素占位，放开 page cache 引用。

**你得到什么。** 总 pinned 从 `dp_size × model_size` 掉到 `model_size`（各 rank 加起来）。Cosmos3-Super DP4：**4 × 124 GB → 124 GB** 总共，每 rank **31 GB**。

**Figure 3。** 主机上从「每 rank 一整模」变成「每 rank 一片」；设备上 AllGather 只拼当前这一层。

## 3. 双缓冲预取：H2D 和 AllGather 叠在计算上

**为什么。** 内存问题解决了，算的时候层还是要完整权重在设备上。一层层全装上，HBM 又满。同步 H2D → 等 → AllGather → 等 → 计算，GPU 在搬数据时闲着。

**为什么能成。** 设备上正好两个 buffer，每个按模型**最大 block** 定容。计算流在 slot 0 跑第 N 层时，后台流把第 N+1 层备进 slot 1。

**Figure 4 / GIF。** 三流时间线：Compute（蓝）、H2D（橙）、AllGather（绿）。红虚线是 event 同步——计算要等 AllGather 完才换槽。

两段准备分两条流：

1. **H2D**（`copy_stream`）：从 pinned CPU 装 `1/dp_size` 分片
2. **AllGather**（`comm_stream`）：把各 rank 分片收进完整权重 buffer

跟计算流用 event 重叠。AllGather 完，参数按缓存的 metadata 重新指到输出 buffer 的切片。buffer 按最大 block 一次性分配，所有层共用。权重 HBM 上界是 `2 × max_block_size`，跟层数无关。

昇腾上 `pin_memory()` 走 `/dev/davinci_manager` 分配能 DMA 的内存，落在 CPU 内核空间，**cgroup 看不见**——所以 cgroup 峰值比直觉低得多。

**你得到什么。** HBM 上只留两层权重（Nano 约 **2 GB**，Super 约 **3 GB**），跟总层数无关。buffer 容量仍随最大 block 长；总 HBM 还要加工作负载相关的激活和通信缓冲。测到的 `dist_offload+SP` 720p 10s：峰值 HBM 从 Nano 到 Super 大约 **22%**（**23.1 → 28.1 GB**），空闲 HBM 大约 **27%**（**11.5 → 14.6 GB**）。模型大 **3.8×**，两边都远低于 64 GB。

**Figure 6**（原文 HBM 图标成 Figure 4）。同一条 720p 10s。HSDP+SP 在 Super 上到 **56.3 GB**。

## 4. DP 多并发：N 个请求一起算

**为什么。** AllGather 只收**权重**分片，跟请求无关。各 rank 在 AllGather 处对齐，激活却可以各算各的。不吃这点，AllGather 之间 rank 闲着，吞吐锁在 1 个请求。

**为什么能成。** 开 `dp_concurrent`，调度器最多攒 `dp_size` 个请求。executor 一次 broadcast RPC 发出去。

**Figure 5。** 一次 broadcast 带请求列表；各 DP rank 算不同请求，同步的 AllGather 换的是跟请求无关的权重片。

```python
# Executor: send all requests at once
reqs_list = [nr.req for nr in new_reqs]
results = collective_rpc("execute_model", args=(reqs_list, ...),
                         unique_reply_rank=None, exec_all_ranks=True)
```

worker 按 **DP rank** 挑请求（不是 global rank，才罩得住 SP/TP）：

```python
dp_rank = get_data_parallel_rank()
req = reqs_list[dp_rank % len(reqs_list)]
```

每个 DP replica 里只有主 rank（SP=0, TP=0, CFG=0, PP=0）回，带着 `dp_rank`。executor 轮询再按 `dp_rank` 排序对上请求。

校验会拒掉 batch-compatibility key 不同的并发请求。key 覆盖空间/时间形状（`height`、`width`、`num_frames`、`fps`），CFG/guidance（`guidance_scale`、`true_cfg_scale`、`cfg_normalize`），`num_inference_steps`，LoRA 身份（`lora_int_id`、`lora_scale`），输出个数，quality mode，以及 pipeline 自己的 `extra_args`——AllGather 是 collective，这些共享字段对不上，一个 rank 跑偏、其余挂死。`extra_args` 必须 JSON 完全一样。seed 和 generator 可以各 rank 不同。#5864 之后 pipeline 不必声明 `supports_request_batch=True`。不兼容或空 prompt 的 wave 在派给 worker 之前就拒；部分 wave 超时 **fail closed**，不让 collective 死锁。

**你得到什么。** 4 个并发请求：**3.22** generated video frames/s，相对 HSDP 单请求基线 **3.3×**，大约理想 4× 的 **83%**。固定 AllGather 开销约 **150 ms/step**，摊到 4 路计算上。

## 昇腾内存账：cgroup 看见的，和物理 RAM

直觉会以为主机要 2× model_size：page cache 一份，shard buffer 一份。昇腾上 `pin_memory()` 经 `/dev/davinci_manager` 把分片放进 CPU 内核 DMA，**cgroup 看不见**。物理 RAM ≈ page cache + pinned 分片 + 框架开销。

**Figure 6。** Cosmos3-Nano DP2：cgroup 看见共享 page cache 和框架 RSS；经 `/dev/davinci_manager` 的 pinned 分片是驱动管的 CPU DMA，不在 NPU HBM。

干净测量（Cosmos3-Nano DP2，新 cgroup）：

```
cgroup usage_in_bytes = 49 GB = cache(31) + rss(18)  ← exact match, no extra
cgroup kmem           = 0 GB
davinci_manager RSS   = 0 kB  (in /proc/<pid>/smaps)
NPU HBM per card      = 10 GB  (< 14.5 GB shard → shard NOT in HBM)
Slab                  = 3.3 GB  (too small for 29 GB shard)
```

| Component | Location | Size | cgroup 管不管 |
|---|---|---|:---:|
| Safetensors page cache | 系统 RAM（用户态，共享） | 1× model_size | 管（cache） |
| Framework（Python/torch/HCCL） | 系统 RAM（用户态，per-rank） | ~3.5 GB × dp_size | 管（rss） |
| Shard（pinned） | CPU 内核 DMA（`/dev/davinci_manager`） | 每 rank model_size / dp_size | 不管 |
| Prefetch buffers | NPU HBM | 每 rank 2 × block_size | 不管 |

cgroup 可见内存按 `O(model_size + dp_size × constant)` 长，不是 `O(dp_size × model_size)`——但物理 RAM 还要加上 cgroup 看不见的 pinned DMA。200B、`dp_size=4`：约 **423 GB** cgroup + 约 **400 GB** 内核 DMA = 约 **823 GB** 物理（2 TB 里装得下），没有 mmap 则是 **2000 GB**。

## 验证结果

全在昇腾 910B3（每卡 **64 GB** HBM，**2 TB** 系统内存）上，Cosmos3-Nano（**33 GB**）和 Cosmos3-Super（**124 GB**）。

### Correctness

| Model | Config | Requests | HTTP | Frames | Video |
|---|---|:---:|:---:|:---:|:---:|
| Nano (33 GB) | DP2 | 2 concurrent，35 steps | 2/2 × 200 | 29/29 | OK |
| Nano (33 GB) | DP4 | 4 concurrent，35 steps | 4/4 × 200 | 29/29 | OK |
| Super (124 GB) | DP2 | 1 request，5 steps | 200 | 29 | OK |
| Super (124 GB) | DP4 | 1 request，5 steps | 200 | 29 | OK |

### 主机内存（cgroup 峰值）

| Model | Config | cgroup Peak | Page Cache | RSS | Per-worker HWM | vs. Baseline |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Nano (33 GB) | DP4 (mmap) | 47 GB | 31 GB | 14 GB | 12.1 GB | — |
| Nano (33 GB) | DP4 (no mmap) | 178 GB | — | — | 36 GB | −73% |
| Super (124 GB) | DP2 | 157 GB | 149 GB | 7 GB | 65.2 GB | — |
| Super (124 GB) | DP4 | 172 GB | 149 GB | 14 GB | 35.5 GB | — |

### NPU HBM

| Model | Config | HBM/card（idle） | HBM/card（inference） | 64 GB Headroom |
|---|---|:---:|:---:|:---:|
| Nano (33 GB) | DP2 | 9.9 GB | 10.4 GB | 55 GB |
| Nano (33 GB) | DP4 | 9.4 GB | 10.2 GB | 55 GB |
| Super (124 GB) | DP2 | ~15 GB | — | ~49 GB |
| Super (124 GB) | DP4 | ~10 GB | — | ~54 GB |

这两行 Super 的 inference HBM **没报**。他们给出的 Super HBM 证据是上面那条 720p 10s `dist_offload+SP` 的峰值/空闲。

### Performance

昇腾：Cosmos3-Nano，**832×480**，**29** 帧，**35** 个去噪步。**Generated frames/s** 是墙钟每秒产出的视频帧（`29 frames × outputs per wave / wave latency`），**不是**播放帧率。

| Strategy | Per-step (ms) | Generated frames/s | CPU/rank | HBM/card | vs. HSDP |
|---|:---:|:---:|:---:|:---:|:---:|
| HSDP+SP（baseline） | 870 | 0.967 | 0 GB | 20.3 GB | — |
| dist_offload+AG (DP4, 1 req) | 1,020 | 0.806 | 3.5 GB | 12.4 GB | −17% |
| dist_offload+AG (DP4, 4 req) | 1,020 | 3.22 | 3.5 GB | 12.4 GB | 3.3× |
| dist_offload no-AG | 1,877 | 0.439 | 28.3 GB | 14.1 GB | −55% |

AllGather 开销 **150 ms/step**（**72 ms** 切流 + **10 ms** HCCL + **68 ms** Python dispatch），Cosmos3-Nano DP4。通信量随层尺寸、参与人数、拓扑变。4 个并发请求把这笔固定成本摊 4 倍。

### NVIDIA B300 结果

同一套 DLO 上 NVIDIA B300 SXM6。Cosmos3-Super BF16（**124 GB**），4× B300（物理 GPU **1,5,6,7**），Python 3.12.3，PyTorch 2.11.0+cu130，CUDA 13.0，vLLM `0.23.0`，Omni commit [`9772bb32`](https://github.com/vllm-project/vllm-omni/commit/9772bb321f558a28c0dca1cb53b44aaf10e4ab69)——PR [#5397](https://github.com/vllm-project/vllm-omni/pull/5397) 的 **合入前**快照。合入后的 head 还有后来的 loader 闸门和 TP/mmap 校验，**不在**这次基准里。下面 MiniMax-H3 自己的版本、`enforce_eager=True`、本地 pipeline 补丁，**不要**套到 Cosmos3。

正确性：各策略输出哈希逐字节相同。T2I seed 42：SHA256 `6e7d2a8c63b88391...` 覆盖 DLO+AG、no-AG、DLO+USP4、legacy layerwise+USP4、HSDP+USP4。T2V 832×480×29f seed 17：同一份 **666,029** 字节（SHA256 `c5d38f5d21ca619e...`）。

CUDA 进程树 PSS 含共享 page cache、pinned CPU 分片和框架内存。昇腾 cgroup **不含** `/dev/davinci_manager` 撑起来的 pinned 分片——GPU PSS 和昇腾 cgroup **不能直接比**。

#### 1024×1024 T2I，50 steps

| Strategy | Concurrency | Wave latency | Throughput | Process-tree PSS | Peak HBM/card |
|---|:---:|:---:|:---:|:---:|:---:|
| DLO+AG DP4 | 4 | 43.69s（median） | 0.0915 outputs/s | 198–202 GiB | 12.62 GiB |
| DLO no-AG DP4 | 4 | 112.96s | 0.0354 outputs/s | 532 GiB | 11.43 GiB |
| HSDP+USP4 | 1 | 15.19s | 0.0658 outputs/s | 483 GiB | 42.00 GiB |
| legacy layerwise+USP4 | 1 | 105.22s | 0.0095 outputs/s | 533 GiB | 13.99 GiB |

DLO+AG DP4 四个并发：吞吐是 HSDP+USP4 的 **1.39×**，HBM 只有 **30%**（**12.6 GiB** vs **42.0 GiB**）。

#### 832×480 T2V，29 帧，35 steps

| Strategy | Outputs/wave | Wave latency | Throughput | Output SHA |
|---|:---:|:---:|:---:|:---|
| DLO+AG DP4 | 4 | 38.79s | 0.1033 outputs/s | c5d38f5d... |
| HSDP+USP4 | 1 | 15.38s | 0.0653 outputs/s | c5d38f5d... |
| DLO+AG+USP4 | 1 | 30.79s | 0.0326 outputs/s | c5d38f5d... |
| legacy layerwise+USP4 | 1 | 81.46s | 0.0123 outputs/s | c5d38f5d... |

#### 工作负载延迟和 HBM（35 steps，DLO+AG DP4 vs HSDP+USP4）

| Workload | DLO strategy | DLO outputs/wave | DLO wave latency | DLO peak HBM/card | HSDP outputs/wave | HSDP wave latency | HSDP peak HBM/card |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 480p, 29f | DLO+AG DP4 | 4 | 38.79s | 14.55 GiB | 1 | 15.38s | 43.77 GiB |
| 480p, ~5s (121f) | DLO+AG DP4 | 4 | 102.58s | 15.88 GiB | 1 | 41.36s (125f) | 53.73–62.65 GiB |
| 480p, ~10s (241f) | DLO+AG DP4 | 4 | 226.70s | 17.33 GiB | 1 | 82.47s (245f) | 53.74 GiB |
| 720p, 5s (121f) | DLO+AG DP4 | 4 | 288.29s | 24.95 GiB | 1 | 87.47s | 52.19 GiB |
| 720p, 10s (241f) | DLO+AG+USP4 | 1 | 214.53s | 24.99 GiB | 1 | 210.05s | 53.73 GiB |

720p 10s（241f）：DLO+AG+USP4 **214.53s**，离 HSDP 的 **210.05s** 只差 **2.13%**，输出逐字节相同（SHA256 `08cb679322996ea6...`），HBM 只有 HSDP 的 **47%**（**24.99 GiB** vs **53.73 GiB**）。

#### MiniMax-H3 在 8× B300：DLO 模式看拓扑

另一次 [MiniMax-H3 B300 研究](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_b300_dlo_industrial_report)，Shunyang Li。跟 Cosmos3 不同，这里出视频**也出音频**：**768×1344**，**124** 视频帧，立体声音频，BF16，每个 replica batch size 1，请求 **50** 步（调度器 **49** 次去噪更新）。`environment.json.txt`：vLLM `0.24.0`，Omni `0.26.0rc2.dev11+g6607f4a7f`（源码 [`9e73ee1`](https://github.com/vllm-project/vllm-omni/commit/9e73ee1a50ce247c638052011914d8027d717f28)）；runner 设 `enforce_eager=True`（图编译**关掉**），还给 `pipeline_minimax_h3.py` 打了 [本地 subgroup-broadcast 补丁](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_b300_dlo_industrial_report)。**不是**未改的发版，也不是默认编译图路径。每条入选 T2VA 路线：两个引擎生命周期各一次完整 warmup 之后，共 **20** 个测到的 wave。吞吐 = 输出个数 / wave 时间；能量把八卡板功耗按输出积分，**不减**空闲基线；外部 `nvidia-smi` 中位间隔 **0.758s**。

**Figure 7。** 三条测过路线上的服务前沿。加大 DP 是用单波延迟换并发产能；到 DP8×SP1，偏好的 DLO 模式从 AllGather 换成 rank-local。

| Service objective | Topology / DLO mode | Wave P50 | Wave P95 | Sustained throughput | Measured peak/GPU | Board energy/video |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Lowest latency | DP1×SP8 / AllGather | 34.55s | 35.25s | 103.84 videos/h | 26.37 GiB | 68.08 Wh |
| Balanced knee | DP4×SP2 / AllGather | 94.73s | 95.31s | 151.89 videos/h | 25.11 GiB | 51.76 Wh |
| Highest throughput / lowest energy | DP8×SP1 / rank-local | 156.74s | 157.03s | 183.78 videos/h | 20.05 GiB | 43.97 Wh |

配对的五波模式对照：没有全局唯一的 DLO 政策。DP1×SP8：AllGather 走 SP 组，吞吐 **+129.4%**，P50 延迟 **−56.6%**。DP4×SP2：吞吐收益只剩 **2.2%**。DP8×SP1：AllGather 吞吐 **−4.1%**，P50 延迟 **+3.8%**，测到的每 GPU 峰值从 **20.03 升到 94.03 GiB**——这时该用 rank-local。FL2VA 首帧和 Ref2VA 图+音频保住同一套延迟–吞吐排序。

**Figure 8。** 三条路线（每条 n=5 个 wave）：FL2VA 首帧 I2VA 和 Ref2VA 图+音频改绝对延迟和吞吐，但 DP1×SP8 → DP4×SP2 → DP8×SP1 的前沿顺序不变。

页上写死的边界：这是拓扑研究，**不是**通用生产主张。DP2×SP4 **没测**；一台节点、一套输入、一个分辨率和帧数；做的是**形状校验**，不是观感质量；源码 `9e73ee1` 外加一份记过的本地 subgroup-broadcast 补丁；运行时警告 Omni 和 vLLM 版本 **没有**发版对齐。档案：[PDF、CSV、105 个 wave 样本、环境哈希、本地 diff、runner](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_b300_dlo_industrial_report)。

### 外推到 400 GB

按下述内存模型做的主机容量外推。**没有真跑过 200B 档模型。** 那个尺度上的最大 block、HBM 余量、带宽、延迟、输出质量都还没验。

| Model | dp_size | cgroup Peak（est.） | Total RAM（est.） | Fits 2 TB? |
|---|:---:|:---:|:---:|:---:|
| 33 GB | 4 | 47 GB | ~80 GB | 是 |
| 124 GB | 4 | 172 GB | ~296 GB | 是 |
| 185 GB | 4 | ~220 GB | ~405 GB | 是 |
| 400 GB | 4 | ~423 GB | ~823 GB | 是 |
| 400 GB | 8 | ~443 GB | ~843 GB | 是 |

## 致谢

vLLM-Omni 贡献者，点名 @hsliuustc0106、@yuanheng-zhao 的评审；Shunyang Li（[@lishunyang12](https://github.com/lishunyang12)）的 MiniMax-H3 B300 拓扑研究和可复现制品；昇腾 NPU 团队提供硬件。

## 参考文献

**源码：** `distributed_layerwise_backend.py`（backend、meta 转换、mmap）；`base.py`（OffloadConfig 和策略）；`multiproc_executor.py`（多队列 executor）；`diffusion_worker.py`（DP 多并发 worker）；`test_distributed_layerwise_backend.py`。

**RFC 和 PR：** Issue #5396；实现 [vllm-omni#5397](https://github.com/vllm-project/vllm-omni/pull/5397)；DLO DP 并发修复 [vllm-omni#5864](https://github.com/vllm-project/vllm-omni/pull/5864)；rank-local DLO DP 的独立请求 [vllm-omni#5911](https://github.com/vllm-project/vllm-omni/pull/5911)。

**模型：** Cosmos3-Nano 33 GB safetensors（17B，72 blocks）；Cosmos3-Super 124 GB（64B，128 blocks）；MiniMax-H3 [B300 DLO 制品](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_b300_dlo_industrial_report)。
