---
source: https://vllm.ai/blog/2026-03-30-extract-hidden-states
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 把 hidden states 从 vLLM 里取出来

英文对照：`en/vllm/blog/architecture/extract-hidden-states.md`  
原文：https://vllm.ai/blog/2026-03-30-extract-hidden-states  
PR#33736，`vllm>=0.18.0`。图在原网页。

EAGLE-3 / P-EAGLE / DFlash 这类 draft 模型吃的是 verifier 的中间层，不是表面文字。训练它们需要大量 hidden states。以前两条路都不体面：用 `transformers` 生成，丢掉 vLLM 的分布式和优化，还可能和线上 hidden 对不齐；或给 vLLM 打补丁，prefix cache / async / 自动 batch 常常得关。

## 设计：假装去投机

vLLM 已经会把 verifier hidden 灌进 Eagle-3 draft；KV Connector 已经会把 KV 写出磁盘 / NIXL / 共享内存。Hidden 和 KV 一样，按 token 对齐、只在自己的前缀下有意义。于是：

1. 一只 **dummy draft**：走现成 Eagle-3 管道接 hidden。
2. dummy 的「attention」并不算 attention，只把 hidden **塞进自己的 KV cache**。
3. 自定义 KV Connector 把这块 dummy KV（其实是 hidden）导出。

显存由 paged KV 分配器管：chunked prefill、抢占、prefix cache 都还在。热路径上没有新开销——你不开这套，引擎当它不存在。8k token、4 层、hidden 4096、FP16 大约 268 MB，不能塞进 HTTP body。

## 怎么开

两只配置必须一起：

```bash
vllm serve Qwen/Qwen3-8B --speculative_config '{
  "method": "extract_hidden_states",
  "num_speculative_tokens": 1,
  "draft_model_config": {
    "hf_config": { "eagle_aux_hidden_state_layer_ids": [3, 18, 33, 36] }
  }
}' --kv_transfer_config '{
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": { "shared_storage_path": "/tmp/hidden_states" }
}'
```

请求回来的 `kv_transfer_params.hidden_states_path` 指向 safetensors：`token_ids [prompt_seq_len]`，`hidden_states [prompt_seq_len, num_layers, hidden_size]`。当时只存 **prompt** 段，建议 `v1/completions` 且 `max_tokens=1`。单机 TP/DP 可用。`ExampleHiddenStatesConnector` 写盘且阻塞；异步写和设备直传还在做。Speculators ≥0.5.0 用这套做 online 训练。
