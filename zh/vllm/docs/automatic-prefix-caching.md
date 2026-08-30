---
source: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
lang: zh
fetched: 2026-08-30
---

# 自动前缀缓存（Automatic Prefix Caching）

APC 会缓存已处理请求的 KV cache。新请求如果和旧请求共享同一段前缀，可以直接复用这段 KV，跳过共享部分的计算。

实现细节：https://docs.vllm.ai/en/stable/design/prefix_caching/

## 如何开启

引擎里设 `enable_prefix_caching=True`。

## 典型受益场景

- **长文档反复问：** 同一份长文档（手册、年报）配不同问题。文档只 prefill 一次，后续请求复用 KV。
- **多轮对话：** 复用聊天历史的 KV，不必每轮重算整段历史。

## 局限

APC 一般不会把性能弄差。它只加速 **prefill**，不加速 **decode**。下面两种情况收益很小：

- 时间主要花在生成很长的答案上；
- 新请求和缓存里的请求没有共享前缀。
