---
source: https://docs.vllm.ai/en/stable/features/speculative_decoding/
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Speculative decoding（功能页）

英文对照：[en/vllm/features/speculative-decoding.md](../../../en/vllm/features/speculative-decoding.md)  
原文：https://docs.vllm.ai/en/stable/features/speculative_decoding/  
原理与 2024 年那组数字：[博客 spec-decode](../blog/performance/spec-decode.md)。后来 CATALOG 里还有 P-EAGLE、DSpark、EAGLE 3.1、AMD 投机解码。训练 draft：[vllm-project/speculators](https://github.com/vllm-project/speculators)。JSON / 工具参数是另一间房：[structured decoding](../blog/performance/struct-decode.md)。

这一页要砍的是中低 **QPS**、memory-bound 负载下的 **ITL**。带模型的方法（EAGLE、MTP、draft model、PARD、MLP）延迟降得最狠；n-gram / suffix 加速温和，高峰时也不额外加重。

## 怎么选（定性）

真实增益看模型家族、流量、硬件、采样。页上的起点：

| Method | 低 QPS（延迟） | 高 QPS（吞吐） | 备注 |
|---|---|---|---|
| EAGLE | 高 | 中–高 | 通用、带模型 |
| MTP | 高 | 中–高 | 目标模型自带 MTP 时最好 |
| Draft model | 高 | 中 | 要另备一只 draft |
| Parallel Draft Model | 高 | 中–高 | draft 延迟低 |
| MLP speculator | 中–高 | 中 | 有兼容的 MLP speculator 时 |
| N-gram | 低–中 | 中 | 轻 |
| Suffix decoding | 低–中 | 中 | 不必另备 draft；深度动态 |
| Custom Proposer | 看实现 | 看实现 | 自带 class（实验） |
| Dynamic Speculative Decoding | 高 | 高于底座 SD | RL / QPS 乱晃 |
| Adaptive Verification | 高 | 高于底座 SD | 按 drafter 信心给每条请求定验证量；目前 **只有 DSpark** |

自己测：`examples/features/speculative_decoding/spec_decode_offline.py` 或 [benchmark CLI](../benchmarking/cli.md)。

## Custom proposer（实验）

`method = "custom_class"`，`model = "your_module.YourCustomProposerClass"`。构造时吃 `VllmConfig`，实现 `propose`。

## `--speculative-config` 合同

CLI 上是一份 JSON；Python 是 `LLM(..., speculative_config={...})`。不是穷尽 schema——生成页的 engine args 和 `vllm.config.SpeculativeConfig` 才是。YAML 配置用嵌套映射，不要逃逸后的 JSON 字符串。这里 **不能** 写 `tensor_parallel_size`，用 `draft_tensor_parallel_size`。`temperature` / `top_p` 是采样参数，不是这个对象。`target_model_config` / `draft_*_config` 由 vLLM 自己填。

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

### 常用键

| Key | Type | Default | 含义 |
|---|---|---|---|
| `method` | string | `None` | `draft_model`、`ngram`、`suffix`、`mtp`、`eagle3`、`dflash`… 常常能推断 |
| `model` | string | `None` | draft / EAGLE head / 辅助模型。`ngram`、`ngram_gpu`、`suffix`、`mtp` 常常可省 |
| `num_speculative_tokens` | int > 0 | `None` | 每步提议几个；推断不出来就要写 |
| `draft_tensor_parallel_size` | int ≥ 1 | `None` | draft 的 TP |
| `max_model_len` | int ≥ 1 | `None` | draft 上下文 |
| `parallel_drafting` | bool | `false` | 只跟 EAGLE 和 draft-model 兼容 |
| `rejection_sample_method` | string | `standard` | `standard` / `synthetic` / `block` |
| `synthetic_acceptance_rates` | list[float] | `None` | 每位接受率 `[0,1]`；长度 = `num_speculative_tokens`；必须非增 |
| `synthetic_acceptance_length` | float | `None` | 目标平均接受长度，落在 `[1, num_speculative_tokens+1]`；跟 rates 互斥 |
| `use_heterogeneous_vocab` | bool | `false` | token 级交集；**只跟 `draft_model`**。打开时概率 draft 采样还不行 |

Gemma 4 assistant checkpoint 当 **MTP speculator** 用，不是普通 draft：`"method": "mtp"`，`model` 填 assistant checkpoint。日志若打出 `SpeculativeConfig(method='draft_model', ...)`，该升级，而不是硬塞进 draft-model。

### N-gram

`prompt_lookup_max` / `prompt_lookup_min`：两个都省则默认 **5**；只省一个则镜像另一个。

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

### Suffix decoding

| Key | Default | 含义 |
|---|---|---|
| `suffix_decoding_max_tree_depth` | 24 | 前缀匹配 + 投机树的合计深度 |
| `suffix_decoding_max_cached_requests` | 10000 | 全局 suffix tree 缓存；`0` 关掉 |
| `suffix_decoding_max_spec_factor` | 1.0 | 投机长度不超过前缀匹配长度的这个倍数 |
| `suffix_decoding_min_token_prob` | 0.1 | 估计概率低于此不投机 |

### 跨词表 draft（TLI）

默认要求词表相同。`use_heterogeneous_vocab: true` 在初始化时按规范化 token 字符串做交集，draft logits 只留共享 token，拒绝采样前再翻译 ID。页上例子：目标 Qwen3-8B，draft SmolLM2-135M-Instruct，3 个投机 token，`gpu_memory_utilization=0.5`。目前 **只有 greedy draft 采样**。

## lossless 口径

- **理论：** 采样在硬件数值精度内 lossless（投机采样那篇论文一族）。浮点误差仍可能轻轻拧分布。
- **算法：** rejection sampler 收敛测试；带 SD 的 greedy 等于不带 SD 的 greedy（`tests/spec_decode/e2e`）。
- **vLLM 的 logprobs 跨 run 不稳定** —— FAQ「Can the output of a prompt vary across runs in vLLM?」。
- batch size / 数值稳定性也会动 logprobs。缓解办法在那条 FAQ，不在这一页再加旗标。

## 已知不兼容

1. Pipeline parallelism 到 `vllm<=0.15.0` **不能** 跟投机解码组合。
2. `vllm<=0.10.0` **不支持** draft-model 投机解码。

页上给贡献者的入口：Office Hours #40、Hacker’s Guide、lookahead scheduling、batch expansion、dynamic speculative decoding。
