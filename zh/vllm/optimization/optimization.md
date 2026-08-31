---
source: https://docs.vllm.ai/en/stable/configuration/optimization/
lang: zh
fetched: 2026-08-30
---

# 优化与调优 — vLLM V1

英文原文：`en/vllm/docs/optimization.md`  
来源：https://docs.vllm.ai/en/stable/configuration/optimization/

显存不够时，另见官方 memory conservation 指南。

## 优化等级

vLLM 提供 4 档（启动时间 vs 性能）：

- `-O0`：无优化，启动最快，性能最低
- `-O1`：快优化。简单编译 + 快 fusion + PIECEWISE cudagraph
- `-O2`：默认。更多编译范围、更多 fusion、FULL_AND_PIECEWISE cudagraph
- `-O3`：激进。目前等于 `-O2`，以后可能加更耗时或实验性优化

## 加快重复启动

同一套（模型、配置、硬件）反复启动时：

- **复用 compile cache。** `torch.compile` 产物在 `VLLM_CACHE_ROOT`（默认 `~/.cache/vllm`），可在机器间拷贝或打进镜像。`VLLM_FORCE_AOT_LOAD=1` 在 cache miss 时直接失败而不是默默重编译。模型、配置、相关 `VLLM_*` 环境变量、torch 版本、GPU 型号变了都会让 cache 失效。
- **`--kv-cache-memory` 跳过 memory profiling。** 启动日志会打出当前分配对应的精确值，下次原样传入可跳过 profiling 和 CUDA graph 显存估算。给小了会限制并发/吞吐；给大了会在分配时失败。只在同一块 GPU、相近空闲显存下有效。OOM 就去掉该参数重新 profile。
- **`--enforce-eager` 不用 CUDA graph。** 跳过编译和 capture，启动最快，稳态 decode 更慢。适合开发循环，或测量启动时间里 compile/capture 占多少。

## Preemption（抢占）

Transformer 自回归导致 KV cache 不够同时扛住当前 batch 时，vLLM 会抢占部分请求、腾出 KV，空间够了再重算。日志类似：

```
Sequence group 0 is preempted by PreemptionMode.RECOMPUTE ... Increase gpu_memory_utilization or tensor_parallel_size
```

抢占能保证不崩，但伤害端到端延迟。频繁抢占时：

- 提高 `gpu_memory_utilization`，给 KV 更多空间
- 降低 `max_num_seqs` 或 `max_num_batched_tokens`，减小并发
- 增大 `tensor_parallel_size`：权重切开，每卡给 KV 更多空间，但同步开销可能过大
- 增大 `pipeline_parallel_size`：层切开，间接给 KV 腾空间，可能增加延迟

可用 Prometheus 指标或 `disable_log_stats=False` 看累计抢占次数。V1 默认抢占模式是 `RECOMPUTE` 而不是 `SWAP`，V1 里重算开销更低。

## Chunked Prefill

把大 prefill 切成小块，和 decode 请求组在同一个 batch 里，平衡 compute-bound（prefill）和 memory-bound（decode）。

V1 能开就默认开。调度策略：**先排所有 pending decode，再用 `max_num_batched_tokens` 预算去排 prefill**；单条 prefill 塞不下就自动切块。

好处：

- decode 优先，ITL 更好
- 同一 batch 里既有 prefill 又有 decode，GPU 利用率更好

### 用 `max_num_batched_tokens` 调

- 小值（如 2048）：ITL 更好，因为 decode 较少被大 prefill 拖慢
- 大值：TTFT 更好，一个 batch 能处理更多 prefill token
- **追求吞吐：建议 `max_num_batched_tokens > 8192`**，尤其是大卡上的小模型
- 若它等于 `max_model_len`，几乎相当于 V0 默认调度（仍会优先 decode）

关掉 chunked prefill 时，`max_num_batched_tokens` 必须大于 `max_model_len`，否则启动可能直接崩。

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

相关论文：https://arxiv.org/pdf/2401.08671 、 https://arxiv.org/pdf/2308.16369

## 并行策略

可组合使用。

### Tensor Parallelism (TP)

层内切参数。单机大模型最常见。

何时用：单卡装不下；或想减轻每卡权重压力、给 KV 腾空间提高吞吐。

```python
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

70B 这类单卡装不下的模型，TP 是刚需。

### Pipeline Parallelism (PP)

按层切到不同 GPU，顺序流水。

何时用：TP 已经用到头还要继续切，或跨节点；很深很窄、切层比切张量更合适。

可与 TP 组合：`tensor_parallel_size=4, pipeline_parallel_size=2`。

### Expert Parallelism (EP)

MoE 专用：不同 expert 分到不同 GPU。`enable_expert_parallel=True` 后，MoE 层用 EP 替代 TP，并行度与 TP size 相同。适用于 DeepSeekV3、Qwen3MoE、Llama-4 等。

### Data Parallelism (DP)

整模复制多份，并行处理不同 batch。GPU 够、要扩吞吐而不是扩模型尺寸时用。`data_parallel_size=N`。MoE 层会按 `TP × DP` 来切。

### 多路 NUMA 绑定

多 socket GPU 服务器上，worker 的 CPU/内存若漂到离 GPU 远的 NUMA 节点会掉速。`--numa-bind` 在 Python 子进程起来前用 `numactl` 钉住。默认自动检测 GPU→NUMA。自定义 CPU：`--numa-bind-cpus`。只作用于 GPU 执行进程，不作用于 CPU backend 的线程亲和，也不绑 API server / DP coordinator。容器里可能需要 `--cap-add SYS_NICE`。Python API 启用时要设 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。

### CPU backend 线程亲和

CPU backend 不走 `--numa-bind`，而是 `VLLM_CPU_OMP_THREADS_BIND` 等环境变量。默认 `auto`。

### 多模态 Encoder 的 batch-level DP

默认 encoder 也用 TP 切权重。Encoder 很小，TP 收益小、每层 all-reduce 开销大。改用 TP 切 **batch 数据**（本质是 batch-level DP）在 `TP=8` 上吞吐/TTFT 大约 +10%；未优化的 Conv3D vision encoder 相对普通 TP 还可再 +40%。权重会在每个 TP rank 复制，勉强能装下的模型可能 OOM。

`mm_encoder_tp_mode="data"`。这不是请求级 DP（请求级由 `data_parallel_size` 控制）。需要模型实现 `supports_encoder_tp_data = True`。已知支持：Qwen2-VL 及以上、Llama4、InternVL、Kimi-VL、GLM-4.1V+、MiniCPM-V-2.5+ 等。

## 输入处理

### fastokens

默认用 Hugging Face `tokenizers`。BPE tokenizer（Qwen、Llama、DeepSeek、GPT-OSS 等）可切到 Rust 的 fastokens：`VLLM_USE_FASTOKENS=1`（v0.23.0+）。需安装 `fastokens>=0.2.0`。对长共享前缀、短 prompt 突发、批量 detokenize 最明显。瓶颈在 GPU prefill/decode 时，换 tokenizer 端到端几乎看不出。

### API server 横向扩展

输入处理（tokenize 等）在 API server，模型执行在 engine core。输入处理成瓶颈且 CPU 有余量时：

```
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4 -dp 2
```

只对在线 serving 有效。每个 API server 默认 8 个线程加载媒体，扩进程时考虑调 `VLLM_MEDIA_LOADING_THREAD_COUNT`，避免 CPU 打满。API scale-out 会关掉多模态 IPC cache（它要求 API 与 engine 一对一），不影响 processor cache。

## 多模态缓存

避免多轮对话里同一张图反复传、反复处理。

- Processor cache：默认开，避免重复处理同一多模态输入
- IPC cache：API 与 engine 一对一时默认开，避免 P0↔P1 反复传
- 默认 key-replicated：key 在 P0 和 P1，数据只在 P1
- TP>1 时共享内存更高效：`mm_processor_cache_type="shm"`
- 大小：`mm_processor_cache_gb`（默认 4 GiB）；设 0 全关

## GPU 部署的 CPU 资源

V1 多进程。CPU 核不够是常见掉速原因，虚拟机里更明显。

N 张 GPU 至少：

- 1 个 API server（HTTP、tokenize、输入处理）
- 1 个 engine core（调度、协调 worker）
- N 个 GPU worker

至少 `2+N` 个进程抢 CPU。**物理核少于进程数会明显伤吞吐和延迟。** engine core 是 busy loop，对 CPU 饥饿特别敏感。

实践中核要更多：OS、PyTorch 后台线程也要 CPU。有超线程时 1 vCPU = 半个物理核，最少需要 `2×(2+N)` 个 vCPU。

DP 或多 API server：

```
最少物理核 = A + DP + N + (DP>1 则再 +1)
```

`A` 为 API server 数（默认等于 DP）。例如 `DP=4, TP=2` 共 8 GPU：4 API + 4 engine + 8 worker + 1 DP coordinator = 17 进程。

CPU 不够时优先伤：输入处理吞吐、调度延迟、流式 detokenize/网络。GPU 利用率低于预期时，先查是不是 CPU 在抢。

## Attention backend

按 GPU 架构、模型、配置自动选。也可手动指定。详见官方 Attention Backend Feature Support。
