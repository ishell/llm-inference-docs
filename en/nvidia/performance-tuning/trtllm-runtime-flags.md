---
source: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-runtime-flags.html
lang: en
fetched: 2026-08-31
---

# Useful Runtime Options

No rebuild. Apply when actually running inference (LLM-API `generate` / serve), not only when `llm.save(...)`.

## Capacity scheduler

- **`GUARANTEED_NO_EVICT` (default):** a started request is never paused. Conservative vs KV.
- **`MAX_UTILIZATION`:** pack as many requests as possible each iteration. Better throughput; may pause in-flight work if KV fills — tail latency risk.
- **`STATIC_BATCH`:** legacy. Skip in production.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.bindings.executor import SchedulerConfig, CapacitySchedulerPolicy

def main():
    prompts = ["Hello, I am", "The president of the United States is",
               "The capital of France is", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
    )
    llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct",
              tensor_parallel_size=4, scheduler_config=scheduler_config)
    for output in llm.generate(prompts, sampling_params):
        print(output.prompt, output.outputs[0].text)

if __name__ == "__main__":
    main()
```

## Context chunking policy

Chunking mixes prefill chunks with generation (see chapter 3).

- **`FIRST_COME_FIRST_SERVED` (default):** finish chunks of the earlier request first. Usually better overall.
- **`EQUAL_PROGRESS`:** one chunk from everyone before anyone’s second chunk. More even TTFT.

```python
from tensorrt_llm.bindings.executor import SchedulerConfig, ContextChunkingPolicy

scheduler_config = SchedulerConfig(
    context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS
)
```

## How much KV memory

Both knobs cap the KV manager. More KV memory → usually more throughput.

- **`max_tokens_in_paged_kv_cache`:** hard token cap.
- **`kv_cache_free_gpu_mem_fraction`:** fraction of **free** GPU memory after the model loads. Default **0.90**. Range (0, 1) — **cannot be 1.0** (need room for I/O). Dedicated GPU: try **0.95**.

If both are set, the engine takes the **min**. Leave `max_tokens` unset unless you know the cap.

```python
from tensorrt_llm.bindings.executor import KvCacheConfig

kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.95)
# or: KvCacheConfig(max_tokens=<n>)
llm = LLM(model="...", tensor_parallel_size=8, kv_cache_config=kv_cache_config)
```

Block reuse / host offload / salting: `trtllm-kvcache.md`.

## `max_attention_window_size`

Sliding-window attention: how many tokens are attended when generating one token. Default = engine `max_seq_len` (feature off). Smaller than `max_seq_len` stores only the last window of KV — less compute and memory, possible accuracy drop.

```python
kv_cache_config = KvCacheConfig(max_attention_window=<number of tokens>)
```
