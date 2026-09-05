---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Encoder 分离（EPD）：别让一张图堵住整列车

英文对照：[en/vllm/blog/serving/epd.md](../../../../en/vllm/blog/serving/epd.md)  
原文：https://vllm.ai/blog/2025-12-15-vllm-epd  
2025-12-15。署名 **Multimodality Workstream @ vLLM**。标题里的 EPD 是 **Encoder / Prefill-Decode 分离**，不是 [Router](router.md) 那篇的文本 P/D——两件事常被缩写成「分离」。原生实现 [PR #25233](https://github.com/vllm-project/vllm/pull/25233)，2025 年 11 月初合入，**v0.11.1** 起。NVIDIA Dynamo 先做过 EPD 风格的拆（文档当时很薄）。单机表亲：`mm_encoder_tp_mode="data"`（ViT DP + LM TP）。

本地图（原文版权仍归原站；学习对照用）。

## Motivation: Why Disaggregate the Encoder in LMM Serving?

现代 Large Multimodal Models（LMM）在 serving 时多出一道瓶颈：**开口之前，所有图像必须先过视觉编码器（例如 ViT）**。这一段的计算剖面，和文本 Prefill、Decode 很不一样。今天的常见做法——encoder + Prefill + Decode 绑在**同一张** GPU 上——会造出结构性的浪费。

### Problems With Colocating Encoder and Text Generation

**1. Encoder–Prefill–Decode Interference**

同卡流水线（E+PD 在同一张 GPU）：

```
[E PD] -> [E PD] -> [E PD]
```

每个请求两段都走完，下一辆才能过。编码器不能和别人的 Prefill / Decode 重叠。

后果：

- 编码器又慢又跳（分辨率、图数、复杂度）。
- 混进纯文本，一张 LMM 输入能让整批停住。
- Prefill 和流式 Decode 的尾巴变得不可预测。
- Compute-bound 的编码器和 memory-bound 的 Decode 共用硬件、共用一套并行策略——对两边都不合适。

**2. Coupled and Inefficient Resource Allocation**

三种剖面要三种最优：

- **Encoder：** 一次性、compute-bound、高并行。
- **Prefill：** 高带宽、大 GEMM。
- **Decode：** 极度 memory-bound、活得久、顺序吐字。

同卡意味着一套并行计划、一套资源比例焊死三阶段：

- 不能只给编码器加卡，而不给文本生成集群买多余的 GPU。
- 偶尔几张图，成本却按高峰来付。

## Solutions: Encoder Disaggregation

把视觉编码器拆成可独立扩缩的服务。

### 1. Pipelined Execution and Elimination of Interference

拆开以后：

```
E → P D   (Request 1)
......E → P D   (Request 2)
..........E → P D   (Request 3)
```

- 请求 N 的编码器可以在 N–1 已经 Prefill / Decode 时跑。
- 纯文本**绕过**编码器，不必在图后面排队。
- 由编码器引起的排队消失。
- 系统变成 pipeline-parallel：吞吐上去，延迟更稳。

### 2. Independent, Fine-Grained Scaling

每一段跟自己的需求曲线走：

- **Encoder GPU** 跟多模态图像量走。
- **Prefill/Decode GPU** 跟总请求率和输出长度走。

浪费也跟着消失：不必为了偶发的图去买一整排 Decode 卡。每池用对的硬件和并行。

### 3. Encoder Output Caching and Reuse

集中的编码器服务天然能跨请求缓存 embedding：

- 常出现的图（logo、示意图、产品图）算一次，用户之间复用。
- 命中时编码器代价为零，直接降 TTFT。
- 命中率涨，编码器负载跟着掉。

## Design

![EPD Architecture](../../../../assets/vllm/blog/serving/epd/01-image.png)

**图注（原文）。** EPD Architecture。

### Components

**Proxy & Router**

- 编排请求流。
- 把多模态（MM）输入送给编码器实例。
- 等编码器写完，再把原请求（embedding 已在远端存储里）转给 Prefill / Decode（PD）实例。

**Data Transfer Layer**

- 编码器产出的多模态 embedding（Encoder Cache，简称 EC）的远程存储。
- 编码器 worker 和 PD worker 之间的共享走廊。

**EC Connectors**

- 把 worker / scheduler 接到那一层。
- 负责存、取 encoder cache。

角色：

- **Scheduler-side connector：** 这一拍调度该 load 还是 save 哪些多媒体 embedding；给下游 worker 做 metadata。
- **Worker-side connector：** 真去读写远端；管每张卡上 embedding 的搬运。

## Workflow

### Dataflow Graph

![EPD Dataflow Graph](../../../../assets/vllm/blog/serving/epd/02-workflow.png)

**图注（原文）。** EPD Dataflow Graph。

### Request Lifecycle

1. **Proxy receives request。** 从原请求抽出多模态输入。拆出 **N 个编码器任务**（每个 MM 输入一个），派到编码器实例。
2. **Encoder scheduling。** 编码器 scheduler 跑这些任务，算出 embedding，经 EC connector 写入远端。
3. **Encoder completion。** 编码器 worker 通知 proxy：全部 embedding 都存好了。
4. **Proxy forwards request to PD instance。** 原请求只带 **image hash，不带像素**，送到 Prefill / Decode 节点。
5. **PD execution。** PD 用 EC connector 从远端把 MM embedding 灌进 model runner cache，Prefill / Decode 照常。

## Implementation

### Core Components

#### 1. `ECConnectorRole`

connector 实例住在哪：

```python
class ECConnectorRole(enum.Enum):
    SCHEDULER = 0   # in scheduler process
    WORKER = 1      # in worker process
```

#### 2. `ECConnectorMetadata`

scheduler 侧与 worker 侧共享的抽象同步 / 状态对象：

```python
class ECConnectorMetadata(ABC):
    pass
```

#### 3. `ECConnectorBase`

所有 connector 的抽象接口。

字段：`role`（scheduler 或 worker）、`config`（connector 自己的配置）、`metadata`（`ECConnectorMetadata`）。

方法：

- `has_caches(request)`：远端是否已有 embedding
- `build_connector_meta(sched_output)`：worker 必须 load 哪些 cache
- `update_state_after_alloc(request, item)`：命中 / 未命中之后更新分配
- `save_caches(encoder_cache)`：把编码器输出推到远端
- `start_load_caches(metadata)`：PD 侧在 Prefill / Decode 之前加载

和文本 KV 的 **KVConnector** 是表亲：算过的中间状态，不要隔着机器再算一遍。

## Scheduler-Side Behavior

### 1. Connector Initialization

Scheduler：

```python
if self.vllm_config.ec_transfer_config is not None:
    self.ec_connector = ECConnectorFactory.create_connector(
        config=self.vllm_config,
        role=ECConnectorRole.SCHEDULER,
    )
```

Worker：

```python
def ensure_ec_transfer_initialized(vllm_config):
    global _EC_CONNECTOR_AGENT
    if vllm_config.ec_transfer_config is None:
        return
    if vllm_config.ec_transfer_config.is_ec_transfer_instance and _EC_CONNECTOR_AGENT is None:
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
            config=vllm_config,
            role=ECConnectorRole.WORKER,
        )
```

### 2. Remote Cache Check

调度媒体时：

```python
remote_cache_has_item = self.ec_connector.has_caches(request)
```

### 3. Cache State Updates

调度之后：

```python
for i in external_load_encoder_input:
    self.encoder_cache_manager.allocate(request, i)
    if self.ec_connector:
        self.ec_connector.update_state_after_alloc(request, i)
```

### 4. Metadata Construction

调度迭代末尾：

```python
ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
scheduler_output.ec_connector_metadata = ec_meta
```

## Worker-Side Behavior

Workers 用 `ECConnectorModelRunnerMixin` 把 connector 手术折进 GPU model runner。

## Execution Integration

### Encoder Side (Saving to Remote Storage)

算出 embedding 之后：

```python
for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
    self.encoder_cache[mm_hash] = scatter_mm_placeholders(...)
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

### Prefill/Decode Side (Loading Remote Embeddings)

媒体编码器路径包一层 loader：缓存的 embedding 在本地编码器跑之前灌进去。

```python
with self.maybe_get_ec_connector_output(
        scheduler_output,
        encoder_cache=self.encoder_cache,
    ) as ec_connector_output:

    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

## Performance Results

**Environment：** 4×A100 80G  
**Dataset：** `vllm bench serve --dataset-name random-mm`  
**Inputs：** 400 / 2000 文本 token；每请求 1–4 张图（640×640 → 大约 **400** 个视觉 token）  
**Outputs：** 150 token  
**QPS range：** 4–24  
**Model：** Qwen3-VL-4B-Instruct  
**Baseline：** 1 Encoder + 3 PD（**1E3PD**）vs Data Parallel（`--data-parallel-size 4`）

生产级 LMM serving 要盯尾延迟——通常是 **P99 TTFT** 和 **P99 TPOT**。文中 **goodput** = 两条 SLO 同时满足时的最大可持续请求率（评测里：TTFT **20000 ms**，TPOT **100 ms**）。

## Short-Text Workloads (~400 tokens)

![Short-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/03-plot_len400_epd_vs_non_epd.png)

**图注（原文）。** Short-Text Workloads Performance。

短文本上，每请求图越多，EPD 越值钱。

- **Single-image：** goodput 轻微改善（23 → 24 QPS）。
- **Four-image：** goodput **翻倍**（6 → 12 QPS）。

尾延迟也明显好：P99 TTFT / TPOT 常常比非 EPD 基线低 **20–50%**。

吞吐–速率曲线：

- 没有 EPD 时，多图大约在 **12–14 QPS** 失稳：P99 TPOT 暴涨 **30–50%**，SLO 破掉。
- EPD 把失稳点往后推，延迟曲线长得更慢——编码器与 Decode 不再抢队；纯文本绕过视觉。

## Long-Text Workloads (~2000 tokens)

![Long-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/04-plot_len2000_epd_vs_non_epd.png)

**图注（原文）。** Long-Text Workloads Performance。

输入变长，图像编码变成小头，系统进入 Decode 为主的区间。即便如此，EPD 仍有实质收益。

基线在 P99 违约前能撑的 QPS：

- 1 图：**8 QPS**
- 3–4 图：**4 QPS**

EPD 维持：

- **18 / 11 / 9 / 8 QPS** —— goodput 大约 **2× 到 2.5×**。

另外：

- 有效 Decode 吞吐在所有多模态设定上 **+10–30%**。
- P99 TTFT **−30–50%**。
- 稳定区间内 P99 TPOT **−20–40%**。

Encode / Text 管道拆开，模态之间不再抢资源：并发更高，吞吐更好，SLO 咬得更紧。

## Hardware Portability: Ascend NPU

同一套实验迁到 Ascend NPU，改动很少：

- **Environment：** 4×Ascend 910B 32G
- **Model：** Qwen2.5-VL-7B-Instruct
- **QPS：** 1–10

![NPU Short-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/05-npu_plot_len400_epd_vs_non_epd.png)

**图注（原文）。** NPU Short-Text Workloads Performance。

![NPU Long-Text Workloads Performance](../../../../assets/vllm/blog/serving/epd/06-npu_plot_len2000_epd_vs_non_epd.png)

**图注（原文）。** NPU Long-Text Workloads Performance。

所有 Ascend 实验上，EPD 表现出**同一套与硬件无关的收益**：

- 稳定区吞吐持续更高（**5–20%**）。
- P99 TTFT 与 P99 TPOT 明显下降。
- 拥堵点推迟，尾延迟更紧。

这确认：收益来自结构上的解耦，不是某家 GPU 的脾气——GPU 和 NPU 都能搬。

## Conclusion

对着 LMM 推理行为和生产负载，他们做成一套**解耦的、pipeline-parallel 的多模态 serving**：

- 降 TTFT 和 TPOT，
- 抬吞吐、更稳，
- 消掉跨模态干扰，
- 让多模态 serving 可以独立扩缩。

这是下一代高性能 LMM serving 的一张能落地的蓝图。当时点名的后续：[编码器实例的参数加载](https://github.com/vllm-project/vllm/pull/30242)、[更多 EC connector](https://github.com/vllm-project/vllm/pull/30468)。

## Related Work

### ViT DP + LM TP

集群 EPD 之前，vLLM 先做了单机混合并行：[ViT Data Parallel + LLM Tensor Parallel](https://github.com/vllm-project/vllm/issues/22743)——视觉编码器跨 GPU 走 DP，语言模型走 TP。降 TTFT、抬吞吐。后来被别的 serving 框架跟进，例如 [SGLang](https://github.com/sgl-project/sglang/pull/13126)。

### Prior Art and Industry Adoption

NVIDIA Dynamo 团队先用 vLLM 做了 [EPD 风格的分离](https://github.com/ai-dynamo/dynamo/blob/44a2cba976d12a79b2164ed11612c1bc7491a3d8/examples/backends/vllm/launch/agg_multimodal_epd.sh#L5)，文档当时有限。vLLM 原生 EPD（[PR #25233](https://github.com/vllm-project/vllm/pull/25233)）2025 年 11 月初合入，**0.11.1** 起，把编码器分离做成开源生态里的一等公民。

## Reference

- Qiu, Haoran, et al. *ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving*. 2025.
- Singh, G., et al. *Efficiently Serving Large Multimodal Models Using Encoder-Decoder Disaggregation*. 2025.

## Acknowledgments

主要贡献者：ZHENG Chenguang、Nguyen Kha Nhat Long、Tai Ho Chiu Hero、Le Manh Khuong、Wu Hang、Wu Haiyan。社区维护者：Roger Wang、Nicolò Lucchesi、Cyrus Leung——合入时的反馈、审阅和把关。

Router 管文本的 P/D；EPD 管「图先去另一栋楼」。[大规模 serving](large-scale.md) 把文本 P/D 再和 Wide-EP 焊在一起。
