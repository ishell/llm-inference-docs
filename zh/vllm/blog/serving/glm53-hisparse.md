---
source: https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading
lang: zh
voice: book-zh
fetched: 2026-09-11
---

# GLM 5.3 优化（一）：Hybrid HiSparse，压力来了才把冷 KV 卸到 CPU

英文对照：[en/vllm/blog/serving/glm53-hisparse.md](../../../../en/vllm/blog/serving/glm53-hisparse.md)  
原文：https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading  
2026-09-08。署名 **vLLM Team**。学习译文，不是官方译本。两篇系列的第一篇（单机 8× H200 聚合部署）。第二篇当时已预告、尚未进 CATALOG：大尺度 P/D 上把 PCP、[DCP](../performance/dcp.md)、[自适应验收](../performance/dspark-adaptive.md) 和 Hybrid HiSparse 拼在一起。GLM-5.2 SLA 邻居：[glm52-b300.md](glm52-b300.md)。CPU / 分层 offload：[kv-offload.md](kv-offload.md)、[tiered-kv-offload.md](tiered-kv-offload.md)。论文：[HiSparse](https://arxiv.org/abs/2608.07009)、[IndexShare](https://arxiv.org/abs/2603.12201)。页上的 occupancy 计算器是 HTML 控件，不收。客户端复现脚本仍在原站。

**页上的 TL;DR。** Hybrid HiSparse 是一层按压力驱动的内存策略，和 Hybrid Memory Allocator、KV offloading 叠在一起。单台 8× H200 对 GLM 5.3 本来偏紧，它让完整 **100 万** 上下文长度跑得起来——这块硬件上以前做不到——并在多种上下文长度上抬高并发。

## 需要的时候才用稀疏

Agent 负载：许多并发请求，每条上下文又长又在长。GPU 块池是固定的，KV cache 迟早给新块腾不出地方。

先前主要两条路，各有代价：

- **抢占**挑一条请求，丢掉它的 KV，回头再 Prefill。每次被挤走都要再付一遍完整 TTFT。
- **Offload** 把块搬到 host，但 dense attention 要求每个 token 都住在 GPU 上，并发仍被 GPU 内存卡住。

稀疏 MLA 的 KV 不一样：indexer 选出 top-K token，attention 只打这些。[HiSparse](https://arxiv.org/abs/2608.07009) 利用这一点，把除选中 token 以外的 KV 都卸到 CPU，于是每条请求在 GPU 上需要的内存有了上界。Indexer 的 KV 仍住在 GPU，并且仍随上下文增长，但整体小得多。GLM 5.3 的 [IndexShare](https://arxiv.org/abs/2603.12201) 是 **每四层稀疏 MLA 共用一层 indexer**。

**Hybrid HiSparse** 在 GPU 还有容量时把 KV 留在 GPU。只有 KV 吃紧时才启用上面那套 HiSparse offload。Hot buffer 的页按 token 建索引；一页可以装来自许多不同 CPU 块的 token，所以压缩能跨很长一段上下文。CPU–GPU 传输只在系统处于 KV 压力（更高并发）时才付。

![two requests](../../../../assets/vllm/blog/serving/glm53-hisparse/01-hisparse-two-requests.svg)

**图注（页上两条请求的示意）。** 抢占：B 的槽位被腾空，KV 没了。常规 offload：B 的 KV 还在 host 上，不必再 Prefill，但要等它全部再装回 GPU 才能跑，于是只有 A 在 Decode。Hybrid 稀疏 offload：每条请求就地把最冷的页交出去，同一批槽位再租给新的尾巴和 hot page，两条都继续 Decode。

只有 Hybrid HiSparse 让两条请求都继续 Decode。Hot page 从和 KV 页**同一块**池子租，并且住在**同一份** KV-cache tensor 里，对稀疏 MLA kernel 看起来就是普通页。Hybrid 稀疏独有的一点：一部分 token 可以在 hot buffer 里，另一部分仍在 GPU 常驻页里，CPU reload 因此变少。

## 它怎么工作

![residency](../../../../assets/vllm/blog/serving/glm53-hisparse/02-hisparse-residency.svg)

**图注（页上 residency 分面）。** 每一面圈出的六个 top-K token 相同，变的只是住在哪。实线箭头：miss，把一行拷进 hot page。虚线箭头：hot hit，不必再拷。

Residency 按页记账。压力上来、下去时，一条请求在三种状态之间走：

- **完全常驻：** 全部稀疏 MLA 的 KV 仍在 GPU，已经完成的前缀页会主动在 host 上落一份。
- **混合常驻：** 尾巴留在 GPU，更早的页只住在 CPU，indexer 要从那些页里拿的行坐在 hot buffer 里。块表里真块和空占位并排；尾巴从不被挤走。一个融合 kernel 解析 top-K：常驻 token 就地读，hot token 读完刷新 LRU，miss 则从 pinned host 内存拷一行进 LRU 槽。Decode 路径上没有任何一步在等 CPU 做决定，所以仍能捕获 CUDA graph。
- **没有常驻：** 新请求复用一份只存在 CPU 上的前缀时，从占位和一张 hot page 开始。Indexer 选到哪些行，哪些行才过来，我们为模型真正 attend 的部分付钱，而不是整段历史。

三种状态能成立，是因为 hot buffer **不是**另开一份分配。一张 hot buffer 页就是一块普通 KV-cache 块，经 Hybrid Memory Allocator 从和常驻页同一块池子租来，请求第一次需要时拿走，不需要时还回去。无论行在常驻页还是 hot buffer，resolver 交出 HMA 行 ID，HMA 用同一套步长 gather。一条请求腾出的块，可以变成另一条请求的 hot-buffer 容量。

HiSparse 在压力到来之前就做准备。可缓存的前缀页一完成，它就排队拷到 CPU，同时仍从 GPU 提供服务。之后 GPU cache 装满，这页可以交出 GPU 槽位，不必再拷一次。即便压力先打到更新的一页，拷贝一入队，它的 GPU 槽位就可以复用；CPU 那份拷贝传完之后，才能给前缀复用。

`hisparse-glm` 分支把这条路径做得很轻：前向之后一次 launch 把所有稀疏 MLA 层一起拷。拷贝排在模型的 GPU stream 上，同步简单、也安全。

## 和 vLLM 其余部分怎么叠

Hybrid HiSparse 是共享 HMA 池上的 residency 政策，也是和其它 KV 机制并排的 connector。其它 cache 组仍走普通的 prefix caching、传输和 offload。Indexer 的 KV **HiSparse 不碰**：标准 OffloadingConnector 可以按块粒度单独卸它。P/D 分离进来的 import，当前缀装不进常驻时可以落在 host 侧。投机解码走逐步可回放的 resolver 计划，共享这条请求的 hot 状态。

Hot buffer 默认每条请求 **2× top-K** 行。MLA 的 KV 在各 TP rank 上相同，所以 pinned host 池按 **每个 DP replica** 分配，并在本地 TP rank 之间共享。TP rank 0 写这份共享拷贝，每个 rank 都能读，CUDA event 保住 stream 顺序。

## 数字

GLM 5.3，**8× H200**，OpenHands 多轮 agent 负载（[来源](https://www.lmsys.org/blog/2026-07-13-glm52-optimization)）：13 轮对话，首轮 **74160** token，后续轮 **753** token，输出固定 **220** token。两套 TP8 部署都用 MTP3、FP8 KV cache、**142K** 准入上限、`max_num_batched_tokens=32768`、`max_num_seqs=256`、`gpu_memory_utilization=0.92`。Offload 基线：**512 GiB** offload 池。Hybrid HiSparse 把同一份 host 预算拆成 **384 GiB** HiSparse 池 + **128 GiB** offload。

![OpenHands Pareto](../../../../assets/vllm/blog/serving/glm53-hisparse/03-openhands-pareto-occupancy.svg)

**图注（页上）。** 上：interactivity–吞吐扫描。Interactivity 是 1000 除以 mean TPOT；逻辑总 token 吞吐含 prefix-cached 的 prompt token，再除以八张 GPU。下：每个测点期间非零 `vllm:num_requests_running` 样本的均值。Hybrid HiSparse：`e8ef1e07bd`。Offload 基线：`80cb71c9ff`。

他们计划在 **vLLM v0.30** 把 Hybrid HiSparse 做宽。精确启动命令在文末附录。

## 只在需要的地方 offload

KV 从 GPU 开始，有空就留在那里，池子不够再一页页交出常驻。Hot buffer 和常驻页共享池和 tensor，压力下的请求以部分常驻继续 Decode，不必等槽位空出来，也不必再 Prefill 自己一遍。

## 估算收益（页上计算器——不收）

原页嵌了一份并发计算器（`/assets/interactive_pages/hisparse_concurrency_calculator.html`）。那是规划估算，**不是**保证的 serving 上限：运行时 workspace、请求长度偏斜、调度行为都会把实际并发压低。

MTP 还会进一步限制并发，因为它的 hot buffer 必须一次装下所有验收 token。写这篇时，每个 hot buffer 要按 `(num_speculative_tokens + 2) × top-K` 来定。他们还在把 buffer 做小，这条可能变。计算器当时**还没**把这项算进去。

## 第二篇

Hybrid HiSparse 在 P/D 部署的 **Decode** 侧最要紧：上下文最长，KV 压力最高。第二篇（已预告）会在大规模上把几块拼起来：Prefill Context Parallelism（PCP）、[DCP](../performance/dcp.md)、[自适应验收](../performance/dspark-adaptive.md)、以及 Hybrid HiSparse。

## 致谢

实现：Matthew Bonanni（Red Hat）、Lucas Wilkinson（Red Hat）、Fares Obeid（Prime Intellect）。设计协作：Chao Lei（Ant Group）、Nicolò Lucchesi（Mistral）。性能评估和这篇博客：Simon Veitner（Red Hat）。感谢 [HiSparse](https://arxiv.org/abs/2608.07009) 作者提供这篇工作用到的稀疏 offload 概念。

## 附录：复现页上的结果

结果用 [vLLM `e8ef1e07bd`](https://github.com/neuralmagic/vllm/commit/e8ef1e07bd2f174bebfe34c3a3e35e952931efb1)（neuralmagic fork 上的钉死提交）。v0.30 之前按这份 checkout 编。一台 8× H200 上的 Hybrid HiSparse：

```bash
vllm serve zai-org/GLM-5.3 \
  --served-model-name glm-agentx \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 142000 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --attention-config '{"hisparse_config":{"host_pool_gib":384}}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":137438953472}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

`host_pool_gib` 按每个 DP replica，并取整到整块 host block。**128 GiB** 的 offload 池（`cpu_bytes_to_use` **137438953472**）存 HiSparse 不管的 cache 组，包括 indexer KV。不要 MTP 的 HiSparse：去掉 `--speculative-config`。图上的无 HiSparse MTP3 基线：留下 `--speculative-config`，去掉 `--attention-config`，把 `cpu_bytes_to_use` 改成 **549755813888**（512 GiB）。HiSparse 和 `--speculative-config` 都去掉，就是无 MTP 基线。HiSparse 当时只实现在 **NVIDIA GPU** 上。

### 复现 padded OpenHands 扫描

博客自带客户端材料（这里不抄文件）：`build_openhands_padded_dataset.py`、`install_evalscope_deps.sh`、`evalscope-all-nodeps.txt`。EvalScope 钉在 `acd09b44384d53174768bb1063f675420f76fae9`。先做出确定的 128 路对话数据集，再跑 c1/c8/c16/c24/c32，每个点都用新对话：

```bash
python3.12 -m venv client-venv
source client-venv/bin/activate
bash install_evalscope_deps.sh
pip install 'modelscope[datasets]==1.34.0' 'lxml==6.0.2'
pip install 'evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9'

python build_openhands_padded_dataset.py \
  --model zai-org/GLM-5.3 \
  --pad-source openscience \
  --first-turn-length 74160 \
  --subsequent-turn-length 753 \
  --num-turns 13 \
  --number 128 \
  --output-path openhand-zai-org-GLM-5.3.json

evalscope perf \
  --model glm-agentx \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset swe_smith \
  --dataset-path openhand-zai-org-GLM-5.3.json \
  --dataset-offset 52 \
  --max-tokens 220 \
  --multi-turn \
  --number 4 16 32 48 64 \
  --parallel 1 8 16 24 32 \
  --extra-args '{"ignore_eos":true}' \
  --name tp8-hisparse384-native128 \
  --outputs-dir results \
  --no-timestamp
```

图上：interactivity 是 `1000 / mean_TPOT_ms`；每 GPU 的逻辑总 token 吞吐是 EvalScope 的总 token 吞吐除以八。他们每个点期间每 **30** 秒刮一次 `/metrics`。请求占用是非零 `vllm:num_requests_running` 样本的均值。MTP 接受长度是 `1 + Δ(vllm:spec_decode_num_accepted_tokens_total) / Δ(vllm:spec_decode_num_drafts_total)`。
