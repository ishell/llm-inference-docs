---
source: https://vllm.ai/blog/2026-03-30-extract-hidden-states
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 取出 hidden states：不必再补丁 vLLM

英文对照：[en/vllm/blog/architecture/extract-hidden-states.md](../../../../en/vllm/blog/architecture/extract-hidden-states.md)  
原文：https://vllm.ai/blog/2026-03-30-extract-hidden-states  
2026-03-30。署名 **Fynn Schmitt-Ulms**。PR [#33736](https://github.com/vllm-project/vllm/pull/33736)，进 `vllm>=0.18.0`。学习笔记，不是官方译文。路子是：一只 dummy draft 接 verifier 的 hidden，再走现成 [KV Connector API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/) 导出去，[Speculators](https://github.com/vllm-project/speculators/) 就不必再补丁引擎内部。

后来吃这些中间层的草稿：[P-EAGLE](../performance/p-eagle.md)、[并行草稿](../performance/parallel-drafting.md)（P-EAGLE / DFlash / DSpark）。库侧接到这条路上是 [Speculators v0.5.0](../performance/speculators-v050.md)（库 PR [#353](https://github.com/vllm-project/speculators/pull/353)）。导出那扇门和 [KV offload](../serving/kv-offload.md) / [Mooncake](../serving/mooncake.md) 同一族 `KVConnector`。引擎侧投机解码主线：[spec-decode](../performance/spec-decode.md)。

适用：离线或在线训 speculator，要 verifier 的 hidden，又不想只靠 `transformers`、也不想 fork 引擎。不适合：把 hidden 塞进 HTTP JSON body；也不适合当时就指望 Decode 段的 hidden——那时的 connector **只存 prompt**。

## Motivation

Hidden states 是模型对 token 序列的内部中间表示。投机解码很吃这一层。

### Speculative decoding recap

常见拼法：一只 **verifier**（你真正在 serve 的大模型）加一只小 **draft**。Draft 出候选，verifier 并行验收。原文给的量级：方法合适时解码可到 **2–5×**，尤其 **低 batch**、模型还卡在 memory-bound 的时候。

把 verifier 的内部 hidden 喂给 draft，对齐会好一截。要 **多层 verifier hidden** 当输入的方法：[EAGLE-3](https://arxiv.org/abs/2503.01840)、[P-EAGLE](https://arxiv.org/abs/2602.01469)、[DFlash](https://arxiv.org/abs/2602.06036)。

训这些 draft，需要一大份 hidden 加上 verifier 输出。多数库（包括 Speculators）以前只有两条路：

1. **用 `transformers` 生成。** 能跑，两个坑：（A）vLLM 那套性能路径没了（大模型、分布式等等）；（B）Transformers 和 vLLM 的 hidden 有一点点对不齐，就会长出一整类 bug。
2. **给 vLLM 打重补丁。** 手工把核心组件接起来、直接调内部 API。内部一改就要跟着修。许多功能还得 **关掉**：prefix caching、自动 batch、async server。Speculators **`<0.5.0`** 就是这么抽 hidden 的。

投机解码越热，这两条路越难看。帖子要的是引擎里一条能跑得动的抽取路径。

## Design considerations

进引擎时原文列过的约束：

**不要把 hidden 放进响应 body。** 它很大。`Qwen3-8B`，`hidden_size` **4096**，抽出来的形状是 `[seq_len, num_layers_to_extract, 4096]`。一条 **8k** token、**4** 层、**FP16**，就是 **268 MB**。序列化进请求响应不现实。

**并发请求的显存也不便宜。** 哪怕临时缓冲，也要预先分配，还得管 chunked prefill、请求抢占，不然就是 OOM。

**热路径上不能加税。** 多数部署根本不要 hidden。改动范围要窄，能复用的就复用。功能关掉时，不该多出运行时或心智负担。

**下游要能换。** 离线训 speculator：先给整份数据集生成 hidden、落到盘、再训。在线训练：边训边生成，传到各个训练进程，最好 **先不落盘**。抽取系统得能接不同的 sink。

## Design insights

四块现成零件，再拼成 Figure 1：

1. vLLM 已经能 serve EAGLE-3（以及同类）投机模型，draft 吃的就是 verifier hidden。Verifier → draft 的管子已经在。
2. 可扩展的 [KV Connector API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/) 已经能从 KV cache 里抽数据（Prefill/Decode 分离一类）。现成实现：走 **Nixl**、写 **磁盘**、进 **共享内存**。API 支持 **异步** 传输，块在传完之前不会被放掉。
3. Hidden 和 KV 一样，按 token 对齐：每个 token 一个值，只在它前面的前缀下有意义。
4. vLLM 已经允许投机 draft 模型用 **单独的 KV cache 配置 / 尺寸**。

拼法：

1. 做一只 **dummy draft**：走现成 EAGLE-3 管道接 verifier hidden。
2. Dummy 有一层假 attention，带着 **自己的 KV cache**。不算 attention，只把 hidden **塞进这块 KV cache**。
3. 自定义 **KV Connector** 把 dummy draft 的 KV（其实是 hidden）存下来，或用别的方式转走。

管子复用 EAGLE-3，出口复用 KV Connector。Hidden 住在 dummy attention 层里，vLLM 知道要给它分显存；这块内存又走同一套 **paged** KV，prefix cache、chunked prefill、高效 batch 都还在。

![design diagram](../../../../assets/vllm/blog/architecture/extract-hidden-states/01-design_diagram.png)

**Figure 1。** Hidden 抽取：EAGLE-3 路上的 dummy draft，dummy KV cache 当缓冲，KV Connector 当出口。

## Usage and Limitations

Python API 示例在树里：[`examples/offline_inference/extract_hidden_states.py`](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/extract_hidden_states.py)。同一套也走 vLLM **server**。原文命令：

```bash
vllm serve Qwen/Qwen3-8B --speculative_config '{
	"method": "extract_hidden_states",
	"num_speculative_tokens": 1,
	"draft_model_config": {
		"hf_config": {
			"eagle_aux_hidden_state_layer_ids": [3, 18, 33, 36]
		}
	}
}' --kv_transfer_config '{
	"kv_connector": "ExampleHiddenStatesConnector",
	"kv_role": "kv_producer",
	"kv_connector_extra_config": {
		"shared_storage_path": "/tmp/hidden_states"
	}
}'
```

两块配置 **必须一起**：

- `--speculative_config` 选假投机方法 `extract_hidden_states`，把 dummy draft 立起来，并用 `eagle_aux_hidden_state_layer_ids` 点名抽哪几层。
- `--kv_transfer_config` 立起自定义 KV Connector，从那些 dummy draft 层里把 hidden 抽走。

写帖子时只有 **`ExampleHiddenStatesConnector`**：简单写盘。更像样的 connector 说是随后会加。只开其中一块，系统不会按预期工作。

Server 起来之后，每个请求会带回 `kv_transfer_params`，里面有 `hidden_states_path`。路径指向一份 **safetensors**：hidden 和 token id。保存目录就是上面的 `shared_storage_path`。

```
# `/tmp/hidden_states/{req_id}.safetensors`
{
	"token_ids": [prompt_seq_len],
	"hidden_states": [prompt_seq_len, num_hidden_layers, hidden_size]
}
```

原文注明：

- 单机多卡可以用 `--tensor-parallel-size` 和 `--data-parallel-size`。
- **只存 prompt token 和它们的 hidden。** 调 `v1/completions`，采样参数用 `max_tokens=1`。

## Ongoing work

- **接到 Speculators。** 这个库本来就是训投机算法的。已合入的 [speculators PR #353](https://github.com/vllm-project/speculators/pull/353) 改走这条 vLLM 原生抽取，并打开 **online** draft 训练。进 `speculators v0.5.0`。
- **示例 connector 的性能。** `ExampleHiddenStatesConnector` 还没优化：写 hidden **会阻塞**。当时在做异步写。
- **设备直传。** 示例先落盘，训练进程再读。当测试实现够用；更大的训练负载撑不住。后续要做 **device-to-device** 的 hidden connector，包括 **多机**。
