---
source: https://vllm.ai/blog/2025-12-15-vllm-epd
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Encoder 分离（EPD）：别让一张图堵住整列车

英文对照：`en/vllm/blog/serving/epd.md`  
原文：https://vllm.ai/blog/2025-12-15-vllm-epd  
2025-12-15。标题里的 EPD 是 **Encoder / Prefill-Decode 分离**，不是 Router 那篇的文本 P/D——两件事常被缩写成「分离」。原生实现 PR #25233，**v0.11.1** 起。

多模态模型在开口之前，图像要先过 ViT。编码器：一次性、compute-bound、要高并行；prefill：大 GEMM、吃带宽；decode：memory-bound、活得久。三件事绑在同一张 GPU 上，屋子会这样塌：

- 流水线变成 `[E PD] → [E PD]`，编码器不能和别人的 decode 重叠。一个人在看图，整列车厢为他停车。
- 分辨率、图数一变，编码器时延乱跳；混进纯文本请求，一张图能让整批抖动。
- 一套并行策略伺候三种剖面，图一多就得给 decode 集群买多余的卡。

`optimization.md` 里那条 `mm_encoder_tp_mode="data"` 是同一麻烦的单机解法：编码器很小，按权重做 TP 不划算，改成按 batch 切数据。EPD 把这把刀拿到集群上——编码器住另一栋楼。


本地图（原文版权仍归原站；学习对照用）：

![image](../../../../assets/vllm/blog/serving/epd/01-image.png)

![workflow](../../../../assets/vllm/blog/serving/epd/02-workflow.png)

![plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/03-plot_len400_epd_vs_non_epd.png)

![plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/04-plot_len2000_epd_vs_non_epd.png)

![npu plot len400 epd vs non epd](../../../../assets/vllm/blog/serving/epd/05-npu_plot_len400_epd_vs_non_epd.png)

![npu plot len2000 epd vs non epd](../../../../assets/vllm/blog/serving/epd/06-npu_plot_len2000_epd_vs_non_epd.png)

## 拆开以后

`E → PD` 可以流水；纯文本绕过编码器；图的 embedding 进远程 **Encoder Cache**，logo / 产品图算一次。Proxy 拆出 N 个编码器任务 → worker 写入远端 → 通知 proxy → PD 节点只带 image hash、用 connector 把 embedding 灌进 model runner。

接口骨架：`ECConnectorRole`（scheduler / worker）、`has_caches` / `save_caches` / `start_load_caches`。调度器决定这一拍谁该 load/save；worker 真去读写。和文本 KV 的 KVConnector 是表亲：都是「算过的中间状态，不要隔着机器再算一遍」。

## 成绩（4×A100、Qwen3-VL-4B、`vllm bench serve --dataset-name random-mm`）

**Goodput：** 同时满足 P99 TTFT **20 s**、P99 TPOT **100 ms** 的最大 QPS。对照：1 编码器 + 3 PD vs `--data-parallel-size 4`。

短文本（约 400 token）：1 图 goodput 23→24；**4 图 6→12（翻倍）**。P99 常常低 20–50%。无 EPD 时多图在 12–14 QPS 附近 TPOT 暴涨——那就是「一张图堵住整列车」。

长文本（约 2000 token）已是 decode 为主：基线 1 图 8 QPS、3–4 图 4 QPS；EPD 维持 **18 / 11 / 9 / 8**，大约 **2–2.5×**。decode 吞吐 +10–30%；P99 TTFT −30–50%，TPOT −20–40%。

Ascend 910B 上 Qwen2.5-VL-7B 同一方向：吞吐 +5–20%，尾延迟更紧——收益来自结构，不是某家 GPU 的脾气。

相关：单机上更早的 **ViT DP + LM TP**（SGLang 后来也跟了）；NVIDIA Dynamo 先做过 EPD 风格。论文：ModServe、encoder-decoder disaggregation。

Router 管文本的 P/D；EPD 管「图先去另一栋楼」。大规模 MoE 那篇把文本 P/D 再和 Wide-EP 焊在一起。多模态缓存（processor cache / IPC cache / `mm_processor_cache_gb`）是同一栋楼里少传同一张图；EPD 是把楼拆开。
