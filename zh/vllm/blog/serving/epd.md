---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Encoder 分离（EPD）：别让一张图堵住整列车

英文对照：[en/vllm/blog/serving/epd.md](../../../../en/vllm/blog/serving/epd.md)  
原文：https://vllm.ai/blog/2025-12-15-vllm-epd  
2025-12-15。标题里的 EPD 是 **Encoder / Prefill-Decode 分离**，不是 [Router](router.md) 那篇的文本 P/D——两件事常被缩写成「分离」。原生实现 [PR #25233](https://github.com/vllm-project/vllm/pull/25233)，2025 年 11 月初合入，**v0.11.1** 起。NVIDIA Dynamo 先做过 EPD 风格的拆（文档很薄）。

多模态模型在开口之前，图像要先过 ViT。编码器：一次性、compute-bound、要高并行；Prefill：大 GEMM、吃带宽；Decode：memory-bound、活得久。三件事绑在同一张 GPU 上，屋子会塌。

`optimization.md` 里那条 `mm_encoder_tp_mode="data"` 是同一麻烦的单机解法：编码器很小，按权重做 TP 不划算，改成按 batch 切数据（ViT DP + LM TP）。EPD 把这把刀拿到集群上——编码器住另一栋楼。

本地图（原文版权仍归原站；学习对照用）：

![image](../../../../assets/vllm/blog/serving/epd/01-image.png)

![workflow](../../../../assets/vllm/blog/serving/epd/02-workflow.png)

![plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/03-plot_len400_epd_vs_non_epd.png)

![plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/04-plot_len2000_epd_vs_non_epd.png)

![npu plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/05-npu_plot_len400_epd_vs_non_epd.png)

![npu plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/06-npu_plot_len2000_epd_vs_non_epd.png)

## 绑在一起时屋子怎样塌

**1. Encoder–Prefill–Decode 互相踩**

同卡流水线：

```
[E PD] -> [E PD] -> [E PD]
```

每个请求两段都走完，下一辆才能过。编码器不能和别人的 Prefill / Decode 重叠。一个人在看图，整列车厢为他停车。

后果：分辨率、图数一变，编码器时延乱跳；混进纯文本，一张图能让整批抖动；Prefill 和流式 Decode 的尾巴变得不可预测；compute-bound 的编码器和 memory-bound 的 Decode 共用硬件、共用一套并行策略。

**2. 资源比例被焊死**

三种剖面要三种最优：

- **Encoder：** 一次性、compute-bound、高并行。
- **Prefill：** 高带宽、大 GEMM。
- **Decode：** 极度 memory-bound、活得久、顺序吐字。

同卡意味着你无法只给编码器加卡，而不给文本生成集群买多余的 GPU。偶尔几张图，成本却按高峰来付。

## 拆开以后

```
E → P D   (Request 1)
......E → P D   (Request 2)
..........E → P D   (Request 3)
```

请求 N 的编码器可以在 N–1 已经 Prefill / Decode 时跑。纯文本**绕过**编码器，不必在图后面排队。由编码器引起的排队消失，系统变成 pipeline-parallel。

独立扩缩：编码器 GPU 跟图像量走；Prefill / Decode GPU 跟请求率和输出长度走。不必为了偶发的图去买一整排 Decode 卡。每池用对的硬件和并行。

**Encoder Cache（EC）。** 集中的编码器服务天然能跨请求缓存 embedding（logo、示意图、产品图）。命中时编码器代价为零，直接降 TTFT；命中率涨，编码器负载跟着掉。

## 设计

**Proxy & Router。** 编排。把多模态输入送给编码器实例。等编码器写完，再把原请求（embedding 已在远端存储里）转给 Prefill / Decode。

**数据传输层。** 编码器产出的 embedding 的远程存储，编码器 worker 和 PD worker 之间的共享走廊。

**EC connector。** 把 worker / scheduler 接到那一层。

- **Scheduler 侧：** 这一拍调度该 load 还是 save 哪些 embedding；给下游 worker 做 metadata。
- **Worker 侧：** 真去读写远端；管每张卡上 embedding 的搬运。

## 请求一生

1. **Proxy 接到请求。** 抽出多模态输入。拆出 **N 个编码器任务**（每个 MM 输入一个），派到编码器实例。
2. **编码器调度。** 算 embedding，经 EC connector 写入远端。
3. **编码器完成。** worker 通知 proxy：都存好了。
4. **Proxy → PD。** 原请求只带 **image hash，不带像素**。
5. **PD 执行。** 用 EC connector 从远端把 embedding 灌进 model runner cache，Prefill / Decode 照常。

## 实现上的 API

### `ECConnectorRole`

connector 实例住在哪：

```python
class ECConnectorRole(enum.Enum):
    SCHEDULER = 0   # scheduler 进程
    WORKER = 1      # worker 进程
```

### `ECConnectorMetadata`

scheduler 侧与 worker 侧共享的抽象同步 / 状态对象（`ABC`）。

### `ECConnectorBase`

字段：`role`、`config`、`metadata`。

方法：

- `has_caches(request)` — 远端是否已有 embedding
- `build_connector_meta(sched_output)` — worker 必须 load 哪些 cache
- `update_state_after_alloc(request, item)` — 命中 / 未命中之后更新分配
- `save_caches(encoder_cache)` — 把编码器输出推到远端
- `start_load_caches(metadata)` — PD 侧在 Prefill / Decode 之前加载

和文本 KV 的 **KVConnector** 是表亲：算过的中间状态，不要隔着机器再算一遍。

### Scheduler 侧

若 `vllm_config.ec_transfer_config is not None`：

```python
self.ec_connector = ECConnectorFactory.create_connector(
    config=self.vllm_config,
    role=ECConnectorRole.SCHEDULER,
)
```

Worker 侧 `ensure_ec_transfer_initialized(vllm_config)`：若 `ec_transfer_config.is_ec_transfer_instance` 且还没有全局 `_EC_CONNECTOR_AGENT`，用 `ECConnectorRole.WORKER` 建一个。

调度媒体时：`remote_cache_has_item = self.ec_connector.has_caches(request)`。

调度之后，对每个 `external_load_encoder_input`：`encoder_cache_manager.allocate`，再 `ec_connector.update_state_after_alloc`。

调度迭代末尾：`build_connector_meta(scheduler_output)`，挂到 `scheduler_output.ec_connector_metadata`。

### Worker 侧

`ECConnectorModelRunnerMixin` 把 connector 手术折进 GPU model runner。

**编码器（save）：** 算出 embedding 之后，scatter 进 `self.encoder_cache[mm_hash]`，再 `maybe_save_ec_to_connector(...)`。

**Prefill / Decode（load）：** 用 `maybe_get_ec_connector_output(scheduler_output, encoder_cache=...)` 当 context manager 包住媒体编码器路径，然后 `_execute_mm_encoder` / `_gather_mm_embeddings`。缓存的 embedding 在本地编码器跑之前灌进去。

## 成绩（goodput）

**Goodput：** 同时满足 **P99 TTFT 20,000 ms**、**P99 TPOT 100 ms** 的最大可持续 QPS。

环境：**4×A100 80G**；`vllm bench serve --dataset-name random-mm`；文本 **400 / 2000** token；每请求 **1–4** 张图（640×640 → 大约 **400** 个视觉 token）；输出 **150** token；QPS **4–24**；**Qwen3-VL-4B-Instruct**。对照：**1 编码器 + 3 PD** vs `--data-parallel-size 4`。

### 短文本（约 400 token）

图越多，EPD 越值钱。

- **1 图：** goodput 23→24 QPS（轻微）。
- **4 图：** **6→12 QPS（翻倍）**。
- P99 TTFT / TPOT 常常低 **20–50%**。

无 EPD 时，多图在 **12–14 QPS** 附近失稳：P99 TPOT 暴涨 **30–50%**，SLO 破掉——那就是「一张图堵住整列车」。EPD 把失稳点往后推，延迟曲线长得更慢：编码器与 Decode 不再抢队；纯文本绕过视觉。

### 长文本（约 2000 token）

编码代价变成小头，已经是 Decode 为主。即便如此：

基线在 P99 违约前能撑的 QPS：1 图 **8**，3–4 图 **4**。

EPD 维持 **18 / 11 / 9 / 8**，大约 **2× 到 2.5×** goodput。

另外：Decode 吞吐 **+10–30%**；P99 TTFT **−30–50%**；稳定区间内 P99 TPOT **−20–40%**。

### Ascend 910B（可移植）

**4×Ascend 910B 32G**，**Qwen2.5-VL-7B-Instruct**，QPS **1–10**，改动很少。

同一方向：稳定区吞吐 **+5–20%**；P99 TTFT / TPOT 下降；拥堵点推迟。收益来自结构，不是某家 GPU 的脾气。

## 单机表亲与前史

集群 EPD 之前，vLLM 先做了单机 **ViT Data Parallel + LLM Tensor Parallel**（[issue #22743](https://github.com/vllm-project/vllm/issues/22743)）：视觉编码器跨 GPU 走 DP，语言模型走 TP。降 TTFT、抬吞吐。SGLang 后来也跟了（[sglang#13126](https://github.com/sgl-project/sglang/pull/13126)）。

论文：Qiu 等，*ModServe*（2025）；Singh 等，*Encoder-Decoder Disaggregation*（2025）。

当时点名的后续：[编码器实例的参数加载](https://github.com/vllm-project/vllm/pull/30242)、[更多 EC connector](https://github.com/vllm-project/vllm/pull/30468)。

## 致谢

主要贡献者：ZHENG Chenguang、Nguyen Kha Nhat Long、Tai Ho Chiu Hero、Le Manh Khuong、Wu Hang、Wu Haiyan。维护者：Roger Wang、Nicolò Lucchesi、Cyrus Leung。

Router 管文本的 P/D；EPD 管「图先去另一栋楼」。[大规模 serving](large-scale.md) 把文本 P/D 再和 Wide-EP 焊在一起。多模态缓存（processor cache / IPC cache / `mm_processor_cache_gb`）是同一栋楼里少传同一张图；EPD 是把楼拆开。
