---
source: https://docs.vllm.ai/en/stable/design/prefix_caching/
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Prefix cache 怎么记账

英文对照：`en/vllm/features/prefix-caching-design.md`  
原文：https://docs.vllm.ai/en/stable/design/prefix_caching/  
对外说明：[prefix-caching.md](prefix-caching.md)。Anatomy 调度那一节是同一张图。

对 prompt token 做 hash → `get_computed_blocks()` 看命中了几块 → `allocate_slots()` 从空闲队列**头**取块（可能把还缓存着、但引用计数已是 0 的块赶走）→ 填满的块立刻入缓存，同一 batch 里后来的人也能用。请求结束且引用计数归零，块按**反序**接到空闲队列**尾**——先来的前缀更可能还留着，近似 LRU。

哈希算法是安全与速度的交易：默认 `sha256`；`xxhash` 更快。碰撞会把别人的记忆当成你的，所以隔离要求高时不要为了几个微秒去换。
