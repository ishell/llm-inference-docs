---
source: https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# AFD Plugin：Attention 和 FFN 也可以不住在同一栋楼

英文对照：`en/vllm/blog/serving/afd.md`  
原文：https://vllm.ai/blog/2026-07-23-vllm-afd-plugin  
2026-07-23。实验性外部插件：https://github.com/vllm-project/afd-plugin。走 `vllm.general_plugins` 和 `--additional-config`，不改 vLLM 源码。当时钉在 vLLM **0.19.1**、Python 3.10–3.13、仅 model runner v1。数字是受控实验，不是 SLA。

MoE 每一层里两件脾气相反的活：Attention 有状态，跟调度和 KV 绑在一起；FFN / 专家是 routed compute + all-to-all。绑在同一套 rank 上，伸缩只能选一个数字。[Wide-EP](large-scale.md) 把专家铺开；AFD 再把 **Attention 服务** 和 **FFN 服务** 拆开，两边拓扑可以不同。请求仍然打到 Attention 的 OpenAI 口；FFN 是一只吃 activation 的 daemon。

## 三件零件

- **Attention worker：** 调度、KV、batch、采样都留在这边。plugin 的 model runner 把 DP / ubatch / 层 / graph 状态告诉 FFN。
- **FFN worker：** 没有请求、没有 KV。后台收 metadata + activation，跑 `compute_ffn_output()`，把结果送回。
- **Connector：** 每个切开的层上搬 hidden state 和元数据。合同中立，CUDA / Ascend 各自实现。

| Connector | 后端 | 执行 | 适合 | Graph |
|---|---|---|---|---|
| `P2pNcclAFDConnector` | GPU | 同步 P2P | Decode | `FULL_DECODE_ONLY` CUDA graph |
| `CAMP2pAFDConnector` | NPU | 同步 CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY` ACL graph |
| `CAMAsyncAFDConnector` | NPU | 异步 CAM | Prefill | 当时还没有 graph |

模型包装：DeepSeek V2/V3 家族（含 V3.2）、GLM MoE DSA。同步路径 DBO **恰好两个** ubatch。两边都加载**完整权重**——这是当时的边界，不是终局。

## 成绩（受控，强制均衡专家）

Ascend 910C，DeepSeek-V3.2 W8A8，饱和 decode，按 die 归一化。物理 48A16F / 64A16F 用来模拟更大的逻辑比。`AFDDecodeBenchConnector` 提供 decode-only KV；AFD 开了 DBO。强制均衡会改输出，所以这不是精度实验。

16K 固定输入：EP64 **232.6** tok/s/die；48A16F 220.3（−5.3%）；64A16F **258.9**（+11.3%）。32K：168.2 / 151.4 / **183.3**（+9.0%）。拆开本身不保证更快——Attention 和 FFN 的配比才是那句话。他们猜当时 FFN 还有算力余量，再加 Attention 比可能还有空间。

异步 prefill（两台 910C、砍到 10 层的 DeepSeek V3.2 W8A8）：`DP4 PCP8 TP1` vs Attention `DP3 PCP8 TP1` + FFN `EP8`。12 rps 时中位 TTFT **15.1 s → 8.0 s**（约 −47%）。这是路径验证，不是全模型声明。

食谱在仓库的 recipes 目录：GPU 同步用 DeepSeek V2 Lite + P2P NCCL；NPU 异步用 V3.2 CAM。下一步：跟更新的 vLLM、MRV2、更多 graph / ubatch、跨异构卡、和 Omni 管道里的 DiT 阶段。

EPD 拆的是视觉编码器；Router 拆的是文本 P/D；AFD 拆的是层内的 Attention 与专家。三把刀切的不是同一块肉。
