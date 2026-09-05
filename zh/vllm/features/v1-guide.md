---
source: https://docs.vllm.ai/en/stable/usage/v1_guide/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vLLM V1 指南

英文对照：[en/vllm/features/v1-guide.md](../../../en/vllm/features/v1-guide.md)  
原文：https://docs.vllm.ai/en/stable/usage/v1_guide/  
发布故事：[v1-alpha](../blog/architecture/v1-alpha.md)。解剖：[Anatomy](../blog/architecture/anatomy.md)。调优顺序：[optimization.md](../optimization/optimization.md)。RFC [#18571](https://github.com/vllm-project/vllm/issues/18571)：V0 **已经拆掉**。V0 能跑、V1 不能的用例，去 GitHub 或 Slack 讲。

V1 留住 V0 里已经稳的东西（模型、GPU kernel、工具），把 scheduler、KV cache manager、worker、sampler、API server 重盖一遍。目标：简单、模块化、好改；CPU 开销接近零；优化进同一座房子；**零配置**——能开的默认开。长上下文数字页上仍写着 “performance benchmark (To be added)”。博客：[vLLM V1: A Major Upgrade…](https://vllm.ai/blog/2025-01-27-v1-alpha-release)（2025-01-27）。这是一份会动的行为变化与支持表。

统一 scheduler 用 `{request_id: num_tokens}` 给每条请求分 token 预算。chunked prefill、prefix cache、speculative decoding 共用这本账，不再各开 Prefill/Decode 旁门。策略：FCFS 或 priority（`--scheduling-policy`；同分走 FCFS）。

## 跟 V0 不一样的地方

- **Chunked prefill** 能开就默认开（V0 按模型条件开）。先排 decode，剩下的 `max_num_batched_tokens` 给 prefill，塞不下就切块。
- **CUDA Graphs** 捕获比 V0 **更吃显存**。
- **Logprobs：** 默认交的是模型 raw 输出 **刚算完**、**还没** 过 temperature / penalty 的值。`--logprobs-mode`：`raw_logprobs`（默认）、`processed_logprobs`、`raw_logits`、`processed_logits`。Raw = 任何 logit processor 之前（含 bad words）；Processed = 全部 processor 之后，包括 temperature 和 top_k/top_p。
- **Prompt logprobs + prefix caching：** 能开，但 logprobs **不** 进缓存。这类请求会 **无视** prefix cache，把整段 prompt prefill 重算一遍。

默认抢占是 `RECOMPUTE`，不是 `SWAP`。频繁发生时先给 KV 房间（提高 `gpu_memory_utilization` 或 TP），见 optimization。

## 支持图例

- 🟢 Functional — 跟 V0 相当或更好
- 🟡 In Progress — 计划中，有 PR/RFC
- 🔴 Removed — 除非有强需求，否则不请回来

### 硬件

| Hardware | Status |
|---|---|
| NVIDIA | 🟢 |
| AMD | 🟢 |
| INTEL GPU | 🟢 |
| TPU | 🟢 |
| CPU | 🟢 |

更多平台走插件：[vllm-ascend](https://github.com/vllm-project/vllm-ascend)、[vllm-spyre](https://github.com/vllm-project/vllm-spyre)、[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)、[vllm-openvino](https://github.com/vllm-project/vllm-openvino)。

### 模型

| 类型 | Status |
|---|---|
| Decoder-only | 🟢 |
| Encoder-Decoder | 🟢 Whisper；其余原生 🔴 |
| Pooling | 🟢 |
| Mamba | 🟢 |
| Multimodal | 🟢 |

**Pooling：** 已完整支持；prefix caching 和 chunked prefill 先给 **last-pooling**；更多 pooling 类别还在做。

**Mamba：** Mamba-2 / Mamba-1（`Mamba2ForCausalLM`、`MambaForCausalLM`、`FalconMambaForCausalLM`）以及混合（`Zamba2ForCausalLM`、`NemotronHForCausalLM`、`FalconH1ForCausalLM`、`GraniteMoeHybridForCausalLM`、`JambaForCausalLM`）。别的混合也有（`Lfm2ForCausalLM`）。上面这些 **都还不支持 prefix caching**。

**Encoder-decoder：** Whisper 原生。其余走插件：**BART** / **Florence-2** 经 [bart-plugin](https://github.com/vllm-project/bart-plugin)。再其余（例如 `MllamaForConditionalGeneration`）同一套路，见 [plugin system](https://docs.vllm.ai/en/stable/design/plugin_system/)。库里的亲戚：[hardware-plugin](../blog/architecture/hardware-plugin.md) / [plugin-system](../blog/architecture/plugin-system.md)。

### 功能

| Feature | Status |
|---|---|
| Prefix Caching | 🟢 |
| Chunked Prefill | 🟢 |
| LoRA | 🟢 |
| Logprobs Calculation | 🟢 |
| FP8 KV Cache | 🟢 |
| Spec Decode | 🟢 |
| Prompt Logprobs with Prefix Caching | 🟢（整段 prompt 重算，见上） |
| Structured Output Alternative Backends | 🟢 |
| Concurrent Partial Prefills | 🟡 [#14003](https://github.com/vllm-project/vllm/issues/14003) |
| `best_of` | 🔴 [#13361](https://github.com/vllm-project/vllm/issues/13361) |
| Per-Request Logits Processors | 🔴 [#13360](https://github.com/vllm-project/vllm/pull/13360) |
| GPU <> CPU KV Cache Swapping | 🔴 |
| Request-level Structured Output Backend | 🔴 |

### 拆掉的

- **采样：** `best_of`（用的人少）。每请求 logits processor → 启动时的 **全局** processor（[RFC #17799](https://github.com/vllm-project/vllm/issues/17799)）。
- **KV：** V1 抢占不再需要 GPU↔CPU swap。要把 KV 寄存在 CPU，走后来的 [Offloading Connector](../blog/serving/kv-offload.md)，不是 V0 那套同步 swap。
- **结构化输出：** 请求级 backend 没了；替代 backend（outlines、guidance）带 fallback 现在支持。

多进程是这座房子的结构：API server、engine core、每卡一个 worker。CPU 核不够时，GPU 像在等端菜的人——optimization 把这件事放第一位。
