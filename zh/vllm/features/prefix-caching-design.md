---
source: https://docs.vllm.ai/en/stable/design/prefix_caching/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Prefix cache 怎么记账

英文对照：[en/vllm/features/prefix-caching-design.md](../../../en/vllm/features/prefix-caching-design.md)  
原文：https://docs.vllm.ai/en/stable/design/prefix_caching/  
对外说明：[prefix-caching.md](prefix-caching.md)。Anatomy 调度那一节是同一张图。文档站上的 overview / free-queue / 时间线例图不收进仓库。

给 KV block 做前缀缓存几乎是免费午餐，**不改**模型输出。公开端点和多数开源引擎都这么干。vLLM 选 **hash**：每一块 KV 由块 **里** 的 token 加上它 **前面** 的 token 一起哈希。

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

块哈希是 `hash(tuple[components])`：

- 上一块的 **parent hash**
- 本块的 **token ID**（精确 ID，为了抗碰撞）
- **extra hashes**：LoRA ID、多模态输入哈希、多租户隔离用的 `cache_salt`

只缓存 **满** 块。

**v0.11** 起默认 `sha256`，旧版哈希不保证无碰撞。`--prefix-caching-hash-algo`：

| Algo | 序列化 | 口径 |
|---|---|---|
| `sha256`（默认） | Python `pickle` | 跨 Python / vLLM 版本 **不一定** 可复现 |
| `sha256_cbor` | `cbor2` | 可复现、跨语言；跨环境要确定性缓存时推荐 |
| `xxhash` | pickle + xxHash 128-bit | 更快，非密码学。要可选包 `xxhash`。理论上碰撞可在多租户里漏隐私 |
| `xxhash_cbor` | canonical CBOR + xxHash | 可复现的 xxHash。同样要 `xxhash` 包 |

## 多模态哈希

`[IMG]` 先变成占位 token，prefill 时再换成图像 embedding。只看占位符，不同图会撞。所以前端图像处理器算出的 **image hash** 作为 extra hash，挂在含这些占位符的每一块上。页上例子：block size 16、41 个占位符 → 四块，每块都带 `<image hash>`。

## 隔离：`cache_salt`

可选的每请求 salt 打进 **第一块** 的哈希。只有盐相同的人能复用 KV。用来挡住靠延迟差猜缓存内容的计时攻击。页上 JSON 示例：聊天请求里 `"cache_salt": "your-cache-salt"`。

## 数据结构

做在 KV cache manager 里。简化的 `KVCacheBlock`：`block_id`（不变）、`block_hash`（装满才赋、驱逐时清）、`ref_cnt`，再加上 `prev_free_block` / `next_free_block` 做成侵入式空闲队列。

两点：

1. 管理器初始化时把所有块一次分配成池，免得 Python 对象来回造，块也随时能点名。
2. 双向指针直接长在块上 → 中间元素挪到队尾 O(1)，不必再套一层 `deque`。

初始化后四件套：**block pool**、空闲队列的头尾指针、**cache blocks**（`hash → block ID`）、**request blocks**（`request ID → 已分配 ID`）。文档站 Figure: Component Overview。

## 操作

### 新请求

1. 调度器：`kv_cache_manager.get_computed_blocks()` —— 给 prompt 做哈希，查缓存。
2. `allocate_slots()`：
   1. 算还要几块新的；不够就返回。
   2. **Touch** 已命中的块：`ref_cnt += 1`；若没别人在用，从空闲队列摘下来（免得被赶走）。
   3. 从空闲队列 **头** 弹块。头若还在缓存里，等于 **驱逐** —— 从此别人不能再复用。
   4. 刚装满的块 **立刻** 进缓存，同一 batch 里后来的人也能打中。

### 正在跑的请求

还是 `allocate_slots()`：计数 → 弹头（缓存头就驱逐）→ 往槽里追加 token ID → 装满就进缓存。

### 重复块（V1 的 block table 只追加）

块大小 4。请求 1 prompt `ABCDEF`、解码 3 个 token：时间线上先缓存块 0（`ABCD`），再缓存块 1（`EFGH`）。请求 2 同样 greedy：Time 0 复用 0，却给 `EFG` 新开 **块 3**；Time 1 把 3 装满成 `EFGH` 再缓存 —— 跟块 1 **重复**。V0 会释放 3，把表改成 `[0, 1]`。V1 的表 **只追加**，于是 `[0, 3]` 留下；请求释放时才清掉这份重复。

### Free

请求结束且 `ref_cnt = 0`：块按 **反序** 接到空闲队列 **尾**。最后一块哈希了更多 token，更不像会被别人复用，该先被赶走。文档站 Figure: Free queue after a request is freed。

### 驱逐（LRU）

空闲队列 **头** 还在缓存里时：

1. 弹头（LRU）。
2. 从 cache map 删掉它的 ID。
3. 清掉它的 hash。

## 时间线（块大小 4，一共 10 块）

- **Time 1：** 缓存空；新请求分 4 块；3 块装满进缓存；第 4 块只装了 3/4。
- **Time 2：** 请求 0 把块 3 装满，再要一块；缓存 3，分配 4。
- **Time 3：** 请求 1，14 个 prompt token，前 10 个跟请求 0 一样 → 只打中前 **2** 块（8 token）；第 3 块只对上 4 里的 2 个。
- **Time 4：** 请求 0 结束。块 2、3、4 反序进空闲队列（2、3 仍在缓存）。0、1 不进队列（请求 1 还占着）。
- **Time 5：** 请求 1 结束并释放。
- **Time 6：** 请求 2，29 个 prompt token，前 12 个跟请求 0 一样。空闲队列本是 `7-8-9-4-3-2-6-5-1-0`；命中的 0、1、2 先被 **touch 并摘走**，队列变成 `7-8-9-4-3-6-5`。分到的是 0、1、2（缓存），然后 7、8、9、4、3（**被驱逐**）。
