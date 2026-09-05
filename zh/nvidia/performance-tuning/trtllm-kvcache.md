---
source: https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# KV Cache 系统（邻居页）

生成阶段不必重复计算已经算过的 K/V。TensorRT-LLM 的 KV 还支持**跨请求复用**，以及卸载、带优先级的驱逐。它认得不同的 attention 窗口，也认得 MQA / GQA。

运行时那一页（`trtllm-runtime-flags.md`）只拧「给 KV 多少显存」和 sliding window。这一页是同一栋房子的结构图。

## 块池

KV 是一块一块的池子。每块装固定数量的 token。**每块的 token 数必须是大于 1 的 2 的幂**，在建引擎时设定。多层可以挤在同一块里，前提是头数和 attention 窗口相同。窗口或头数不同（GQA、MQA、混合窗口）就再开一个池。

多池时，空闲显存在初始化时按比例切开，之后是静态的。官方承认这不是最优，正在改。

填满的块进一棵 **radix 树**。后来的请求如果前缀相同，就跳过计算、共享这块显存。复用既省算力，也省房子。

## 驱逐与卸载

需要新的空白块时开始驱逐。核心是**带优先级的 LRU**：优先级 0–100（100 最重要）。必须先清空最低优先级，才能动下一档。同一优先级里，最久没用的先走。

从 GPU 驱逐时，KV 可以先拷到 **host（CPU）** 上的块。host 上的块仍留在搜索树里，直到从 secondary 再被赶走。primary 和 secondary 用同一套驱逐规则。

当前实现的一个坑：**只能驱逐叶子**（radix 树上没有子孙的块）。对 full attention 还好；对有限窗口的层不理想。官方说以后会修。

Host 卸载：`host_cache_size`（字节），默认 **0**。低于 `secondary_offload_min_priority`（默认 **35**）的块不卸，直接从 GPU 丢掉，少一点 PCIe 流量。

## 保留策略

请求可以带 retention policy：一段 token 范围一个优先级，还可以设 `duration_ms`。过期后回到默认优先级 35。`TokenRangeRetentionConfig` 只管 **prompt** token；生成 token 用 `decode_retention_policy` / `decode_duration_ms`。duration 设 `None` 表示不过期。`transfer_mode` 是调试开关，别用。

## 跨请求复用开不开

`enable_block_reuse` 默认开。关掉就没有前缀共享。

调度器还有 `scheduler_config.enable_prefix_aware_scheduling`（默认 True）：只用**估计**的可复用 token 来推迟重复的第一块 context、少记一些 token 预算。设成 False 只关掉调度器的这笔估计，**真正的块复用仍由 `enable_block_reuse` 决定**。

```yaml
kv_cache_config:
  enable_block_reuse: true
scheduler_config:
  enable_prefix_aware_scheduling: false
```

部分复用（一块里只匹配到若干 token）：`enable_partial_reuse` 默认开。`copy_on_partial_reuse` 决定是拷一份再给别人用，还是只有没人占用时才能借。

## 显存与 dtype

`free_gpu_memory_fraction`：模型加载完后，空闲 GPU 显存的比例，默认 **0.9**，必须在 (0, 1) 之间。若同时设了 `max_tokens`，按两者里较小的那个分配。`dtype` 默认 `auto`，从模型配置推断。

`max_attention_window` 可以是按层的整数列表。比层数短就循环：`[4096, 256]` 表示奇数层全窗口、偶数层 256。

## 盐值：谁能复用谁的记忆

`cache_salt` 把盐混进块的 hash。只有盐相同的请求才能共享缓存——用来隔离租户，挡住「从别人的 KV 里偷 prompt」这类攻击。隔离**完全靠 hash**：盐进摘要，前缀是否命中只看 digest 相不相等，不再逐 token 比对。因此块 key 必须是有抗碰撞能力的密码学 hash（256-bit digest；SHA-256 大约 128-bit 碰撞阻力）。**不要换成非密码学 hash**，否则可以构造碰撞绕过隔离。

## 多模态 UUID

视觉模型默认用内容 hash 当多模态输入的身份。跨会话不好管理。可以在 `TextPrompt` 上提供 `multi_modal_uuids`。缓存 key 是 `BLAKE3(UUID || Content)`：同一 UUID 不同内容仍是不同条目；不同 UUID 同一内容也隔离。原始 UUID 会出现在 KV cache events 里，方便外部系统记账。部分条目可以是 `None`，回退到纯内容 hash。

## 选哪个 KV manager

`use_kv_cache_manager_v2` 默认 `auto`：听模型的，没有声明就回退 V1 C++ manager。可以显式 `true` / `false`。

默认走 V2 的模型：

| 模型 | 原因 |
|---|---|
| Hybrid Mamba（NemotronH、Qwen3-Next） | attention KV 和 Mamba state 必须一起定大小 |
| DeepSeek-V4 | 稀疏 attention 带每层辅助 buffer |
| GPT-OSS | 隔层 sliding window（VSWA），两个池独立定大小 |
| Gemma3 / Gemma4（文本和多模态） | 交替 sliding window 与全窗口（VSWA） |

Gemma4 的 hybrid / sparse 会**无条件**走 V2：V1 的统一池装不下它们的 per-layer layout。

**双模型 speculative decoding**（例如 Eagle3 且 `eagle3_one_model=False`）不能用 V2：草稿模型是另一份引擎、另一份 KV，V2 会按整份 `max_gpu_total_bytes` 给两边各自定预算，而不是切开。`auto` 时会为这种组合回退 V1；显式 `use_kv_cache_manager_v2: true` 会报错。

Hybrid Mamba 还有 snapshot 边界（`kv_cache_config.mamba_state_config`）。需要时看原文；`avg_seq_len` 用来按负载比例切开 attention KV 和 Mamba 池，不设的话会警告并回退到 `max_seq_len / 2`。

已废弃：`use_uvm`；`sink_token_length` 在 PyTorch backend 上被静默忽略（kernel 不支持 StreamingLLM）。

Speculative decoding 的各类模型都支持跨请求复用，细节见官方 speculative decoding 页。
