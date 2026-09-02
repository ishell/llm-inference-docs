---
source: https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Qwen3.5 25K TPS/GPU：GDN + 异构 cache 搬完才到 Pareto 左端

英文对照：`en/vllm/blog/serving/qwen35-25k-tps.md`  
原文：https://vllm.ai/blog/2026-08-06-qwen35-25k-tps  
GB200 NVL72。Qwen3.5-397B-A17B-NVFP4。ISL/OSL=8192/1024。

系统 TPS/GPU ≠ 单用户 TPS。他们扫的是 Pareto **左端**（总吞吐），并发 64–5120，没测 1–32。decode 固定 1×DEP8；prefill 4–8×DEP2。峰值 **25,000 tok/s/GPU**。GSM8K 五套拓扑都是 **88%**，和聚合跑对齐。

三刀：Blackwell GDN prefill（FlashInfer，相对 FLA/Triton 约 1.02–5.78×；8×B200 上端到端 prefill 约 **1.13×**，mean TTFT −12%）。`--gdn-prefill-backend flashinfer`，`auto` 时会自己选。HMA+NIXL 把描述符 4284→1650；Qwen3.5 再接 GDN 状态搬运。两处 async scheduling 竞态修完才敢开 `--async-scheduling`——不开，精度会掉到零。

菜谱要点：`VLLM_SSM_CONV_STATE_LAYOUT=DS`（P/D 强制）；`--mamba-ssm-cache-dtype bfloat16` 抬 decode KV 容量；`--language-model-only` 关掉多模态、打通 fused QK-norm+RoPE+gate；prefill `--max-num-batched-tokens 16384`（2×ISL，少 prefill 时约 +8%）；高并发把 `--max-cudagraph-capture-size` 抬到 cc/8+128；随机集关掉 prefix cache；`--stream-interval 100` 砍前端，但会缓冲，优化 ITL/TPOT 时别用。`--api-server-count 1` 才能看到默认 stats。下一步才是 TEP/TP 扫 **每用户** Gen TPS。接 [hybrid-ssm](hybrid-ssm.md) 与 [dcp](../performance/dcp.md)。

本地图（原文版权仍归原站；学习对照用）：

![pareto curves by prefill endpoints](../../../../assets/vllm/blog/serving/qwen35-25k-tps/01-pareto-curves-by-prefill-endpoints.png)

![pareto frontier qwen35 nvfp4](../../../../assets/vllm/blog/serving/qwen35-25k-tps/02-pareto-frontier-qwen35-nvfp4.png)
