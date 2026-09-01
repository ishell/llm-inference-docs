---
source: https://vllm.ai/blog/2024-10-23-vllm-serving-amd
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 2024 年在 MI300X 上侍候 Llama：八个旋钮

英文对照：`en/vllm/blog/architecture/mi300x-serving.md`  
原文：https://vllm.ai/blog/2024-10-23-vllm-serving-amd  
2024-10-23。Embedded LLM / Hot Aisle。vLLM **0.6.2** 时代。图在原网页。数字是 8× MI300X、BF16、ShareGPT 相对 TGI 的演示。后来的 ROCm attention 路由见 [rocm-attention](rocm-attention.md)；硬件插件见 [hardware-plugin](hardware-plugin.md)。旗标是当时的，今日请以文档为准。

相对 TGI：Llama 3.1 405B 吞吐约 **1.5×**、TTFT 约 **1.7×**；70B 吞吐约 **1.8×**、TTFT 约 **5.1×**。16 QPS 优化配置下 405B TTFT 平均约 **3.8×** 快。演示。

## 当时的口诀

- **关掉 chunked prefill**（MI300X 上多数情况）。不确定就先关。
- `--num-scheduler-steps` **10–15**（multi-step，把 CPU 调度摊到多步 GPU）。再高收益递减。
- prefix cache 命中低 → 连 chunked prefill 一起关。ShareGPT 二轮大约 50% hit；~0.9% 时开两者不如全关。
- 长上下文 `--max-seq-len-to-capture 16384`。再大 bucket 变粗，图可能更差。
- `echo 0 > /proc/sys/kernel/numa_balancing`；`NCCL_MIN_NCHANNELS=112`。
- KV dtype 默认 auto（跟模型走）。FP8 能省房间，70B 上吞吐略慢于 auto。
- 吞吐：用能装下的**最小 TP**，多开实例。延迟：TP = 节点 GPU 数。MI300X 太大，TP 拉满会饿着每卡。
- `--max-num-seqs` 512+。ShareGPT 短进出时 1024 仍可能是瓶颈。
- `VLLM_USE_TRITON_FLASH_ATTN=0` 走 **CK Flash Attention**。

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve meta-llama/Llama-3.1-70B-Instruct \
  -tp 4 --max-num-seqs 1024 --max-seq-len-to-capture 16384 \
  --enable-chunked-prefill=False --num-scheduler-steps 15
```

当时镜像 `ghcr.io/embeddedllm/vllm-rocm:cb3b2b9`。这篇写的是 chatbot 短进出；摘要 / 长生成要另调。CK vs Triton、hipBLASLt、PP 见 Leonard Lin 那篇。
