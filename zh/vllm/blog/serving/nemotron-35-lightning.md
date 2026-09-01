---
source: https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Nemotron 3.5 Lightning：常开 agent 的小炉子

英文对照：`en/vllm/blog/serving/nemotron-35-lightning.md`  
原文：https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm  
2026-08-10。图在原网页。数字是演示。镜像 `vllm/vllm-openai:v0.27.1`。

从 Nemotron 3 Ultra **蒸馏**下来的 hybrid MoE：30B 总参、3B 激活、1M 上下文、纯文本。角色是「前沿模型管编排、小模型跑碎步」。架构与 Nemotron 3 几乎同构，差在权重和投机栈。**不是新引擎。**

## 旗与硬件

BF16 / NVFP4。卡面很宽：DGX Spark、Station、RTX PRO、Jetson、H100/H200/A100/L40S、B200/GB200、B300/GB300。1×H100 BF16 示例：`--max-num-seqs 256`、`--max-num-batched-tokens 32768`、`--enable-prefix-caching`、`--async-scheduling`、`--mamba-backend flashinfer`、`--moe-backend humming`、`--linear-backend humming`、`--reasoning-parser nemotron_v3`、`--tool-call-parser qwen3_coder`。

投机三条：**MTP**、**DFlash**、**DSpark**。低延迟用 DSpark（H100/H200/Spark）；冲吞吐当时建议关掉投机。NVFP4 常配 `--kv-cache-dtype fp8`、`--moe-backend marlin`。Humming 换掉默认 Marlin 做 W4A16 ReLU2 MoE，演示约 **+20%** 吞吐；ReplaySSM 接 Mamba2。

## 数字（演示）

相对同尺寸开源模，声称最高 **4×** 吞吐。PinchBench：完成 1 万任务大约快 **30%**（相近准确率）。Pareto 图在原文：Spark / H100，prefix 32K，再 10 轮 2k in / 1k out。
