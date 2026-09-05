---
source: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 自动前缀缓存（Automatic Prefix Caching）

英文对照：[en/vllm/features/prefix-caching.md](../../../en/vllm/features/prefix-caching.md)  
原文：https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/  
实现：[prefix-caching-design.md](prefix-caching-design.md)。图在文档站。

已经算过的 KV 不必再算。新请求若和旧请求共享同一段前缀，引擎直接把那几块房间的钥匙递过去，prefill 从分叉处接着干。V1 里这件事接近零额外开销——V0 时代 prefix cache 是要付记账税的。

打开：`enable_prefix_caching=True`，或 `vllm serve` 的 `--enable-prefix-caching`。哈希默认 `sha256`（`--prefix-caching-hash-algo`）。跨环境要可复现用 `sha256_cbor`；`xxhash` / `xxhash_cbor` 更快，碰撞隔离更弱——多租户先读 [设计页](prefix-caching-design.md) 的安全注。

## 谁会发光

- **同一份长文档、不同的问题。** 手册、年报、系统提示只 prefill 一次。
- **多轮对话。** 历史不必每回合从第一个字重读。Agent 把这个形状放大到 131:1 的 ISL/OSL——见 [Mooncake](../blog/serving/mooncake.md)。

## 谁几乎看不见

APC 一般不会把性能弄差。它只加速 **prefill**，不加速 **decode**。答案很长、或新请求和缓存里的人完全没有公共前缀时，收益接近零。GPU 上的块仍会按近似 LRU 被挤走；挤走之后，本机 DRAM 的 [Offloading Connector](../blog/serving/kv-offload.md) 和集群池的 Mooncake 是下一层。路由器若把下一回合送到从没见过这段前缀的实例，本机 APC 等于没开——[Router](../blog/serving/router.md) 的 consistent hashing 就是为了少发生这种事。
