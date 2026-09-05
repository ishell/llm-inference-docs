---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 第 6 章：运行时旗标

前面几章改的是引擎怎么被造出来。这一章改的是引擎已经造好之后，推理时怎么排队、怎么分 KV。**不用重建。** LLM-API 端到端示例里，这些旋钮出现在真正 `generate` 的那一次，而不是 `llm.save(...)` 的那一次。

## 容量调度（Capacity Scheduler Policy）

三种：

- **`GUARANTEED_NO_EVICT`（默认）**：已经开始的请求不会被暂停。对 KV 更保守——宁可少塞，也不把客人请出座位。
- **`MAX_UTILIZATION`**：每一拍尽量塞满。吞吐通常更好。KV 触顶时，可能把已经在飞的请求暂停。尾延迟会变坏。
- **`STATIC_BATCH`**：遗留。生产别用。

追求吞吐就试 `MAX_UTILIZATION`，但要记住：暂停不是免费的，它会把某个人的 ITL 撕开一道口。

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bindings.executor import SchedulerConfig, CapacitySchedulerPolicy

def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
    )
    llm = LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4,
        scheduler_config=scheduler_config,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt, output.outputs[0].text)

if __name__ == "__main__":
    main()
```

## Context chunking 策略

第 3 章说过：chunking 让 prefill 和 generation 更容易打在同一拍，吞吐通常更好，平均 TTFT 也更稳。

两种排队方式：

- **`FIRST_COME_FIRST_SERVED`（默认）**：先来的请求，尽量把它的 context chunk 排完。整体成绩通常更好。
- **`EQUAL_PROGRESS`**：先给所有请求各一块，再给任何人第二块。理论上 TTFT 更齐——没有人被永远留在门厅。

多数服务用默认；若你在意「大家同时看见第一个字」，再试 EQUAL_PROGRESS。

```python
from tensorrt_llm.bindings.executor import SchedulerConfig, ContextChunkingPolicy

scheduler_config = SchedulerConfig(
    context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS
)
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    scheduler_config=scheduler_config,
)
```

## KV 能装多少 token

两只旋钮控制 KV 管理器的上限。KV 越大，通常吞吐越高——更多请求能同时把记忆留在 GPU 上。

- **`max_tokens_in_paged_kv_cache`**：直接钉死「最多多少 token」。
- **`kv_cache_free_gpu_mem_fraction`**：模型加载完之后，空闲显存里拿出多少给 KV。浮点，**0.0 到 1.0 之间，但不能是 1.0**——输入输出还要住。默认 **0.90**。

只设 fraction 时，引擎按剩余显存算出 token 上限。两只都设，取**较小**的那个。

不清楚上限就别设 `max_tokens`。GPU 独占、没有别的程序抢显存，可以把 fraction 试到 **0.95** 去追吞吐。

```python
from tensorrt_llm.bindings.executor import KvCacheConfig

kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.95)
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    kv_cache_config=kv_cache_config,
)
```

若你明确知道 token 上限：

```python
kv_cache_config = KvCacheConfig(max_tokens=<number of tokens>)
```

更细的块复用、卸载、盐值隔离，在邻居页 `trtllm-kvcache.md`。

## Maximum attention window

`max_attention_window_size` 给 sliding window attention 设「生成一个 token 时最多看多远」。默认等于引擎的 `max_seq_len`，等于功能关着。

设得比 `max_seq_len` 小：只保留最后这一扇窗口的 KV。输入比窗口长时，精度可能开始掉，但算力和显存都会轻松一些。用延迟换准确，或用准确换延迟——你选。

同样走 `KvCacheConfig`：

```python
kv_cache_config = KvCacheConfig(max_attention_window=<number of tokens>)
```

运行时旗标是便宜的实验：不用重建，改完再打一轮 `trtllm-bench`。把吞吐从「引擎允许的上限」里再抠一点出来，同时盯着尾延迟——`MAX_UTILIZATION` 喜欢把平均数化妆，把某几个请求藏进暂停里。
