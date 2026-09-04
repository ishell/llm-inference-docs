---
source: https://vllm.ai/blog/2025-12-13-speculators-v030
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Speculators v0.3.0：把 EAGLE-3 草稿训出来

英文对照：[en/vllm/blog/performance/speculators-v030.md](../../../../en/vllm/blog/performance/speculators-v030.md)  
原文：https://vllm.ai/blog/2025-12-13-speculators-v030  
2025-12-13。署名 **Fynn Schmitt-Ulms、Helen Zhao、Rahul Tuli and Dipika Sikka（Red Hat AI Model Optimization Team）**。仓库：[vllm-project/speculators](https://github.com/vllm-project/speculators)，发版 [v0.3.0](https://github.com/vllm-project/speculators/releases/tag/v0.3.0)。学习笔记。后来的 DFlash / 在线训练：[v0.5.0](speculators-v050.md)。hidden 导出后来收成引擎能力：[extract-hidden-states](../architecture/extract-hidden-states.md)。验收数学仍是 [spec-decode](spec-decode.md)。并行草稿总览：[parallel-drafting](parallel-drafting.md)。

投机解码要 **每只 verifier 一只草稿**。训起来难，给 vLLM 用的生产级训练工具当时又少。v0.3.0 把离线数据 → 训练 → `vllm serve` 串成一条。数字是页上的量级，不是你的 SLA。

## 要点

- 投机解码能压推理延迟，可每只 LLM 都要单独训草稿；面向 vLLM 的现成训练工具当时不够用。
- [Speculators v0.3.0](https://github.com/vllm-project/speculators/releases/tag/v0.3.0) 提供 EAGLE-3 草稿的端到端训练，产物能直接在 vLLM 里跑。
- 训练侧：用 vLLM 做离线数据生成；单层 / 多层草稿；MoE 与非 MoE verifier 都覆盖。

## 规模化推理

过去十年模型又大又强，推理账单跟着涨。LLM 按 token 顺序吐字，每一步都要穿过几十亿参数；模型再大，这段顺序计算就越像瓶颈——能力在，速度不在。

投机解码是当时看好的一条：小草稿先猜，大 verifier 再并行验收，把「一步一个 token」拆开。这篇讲这套优化、介绍 Speculators，并钻进 v0.3.0：研究者、工程师、ML 实践者可以端到端做出投机解码模型，再无缝接到 vLLM。

## 什么是投机解码

投机解码让 LLM **一次前向吐出多个 token**。一只便宜、跑得快的 **draft**（常常就一块 transformer block）自回归猜一串；完整尺寸的 **verifier**（你真正在 serve 的那只）并行处理这些候选。每个位置，verifier 决定同不同意草稿的预测：拒绝则丢掉后面的序列；接受则进最终回复。

原文列的好处：

1. 最终回复与单独跑 verifier **同分布**，投机解码本身不伤模型质量。
2. verifier 能 **并行** 生成多个 token。
3. 草稿很小，通常只加一点点开销。

合在一起，延迟可以掉到大约 **1.5–3×**——页上的量级，草稿必须对齐 verifier。

## 在 vLLM 里用投机解码模型

vLLM 和 Speculators 把跑投机解码做成跟 `vllm serve` 任何一只模型差不多。投机解码最吃香的是 **低吞吐**：GPU 还没灌满，verifier 的并行验收才有空档。草稿还得跟 verifier **贴得很近**，所以几乎总是 **一只 verifier 一只草稿**。训 LLM 专用草稿又难又慢；Speculators 把这条训练链收短，产物直接进 vLLM。

## 训新草稿

当时投机解码算法的 SOTA 是 Eagle3（[Zhang et al., 2025](https://arxiv.org/abs/2503.01840)）。

Eagle3 草稿吃 verifier **三层** hidden，抓住潜特征；再拼上 token id，送进更小的草稿，自回归吐 draft token。

于是训练数据要四样：

1. verifier 三层中间 hidden
2. token id
3. loss mask（只在模型回复上训练，用户 prompt 不算）
4. verifier 输出概率（草稿的训练目标）

### 数据生成

从 vLLM 里直接抠这些值并不轻松。v0.3.0 用 hidden states generator 做 **离线** 数据：从普通 LLM 文本数据集抽出 hidden 张量，落到盘上，留给后面的训练。

三截：预处理、hidden 生成、保存。

本地图（原文版权仍归原站；学习对照用）：

![data generation](../../../../assets/vllm/blog/performance/speculators-v030/01-data_generation.png)

**Figure 1。** 离线数据生成总览：原始对话进预处理，再抽 hidden、落盘。

预处理吃生数据集，然后：

1. 重排、归一化对话轮次
2. 套模型的 chat template
3. tokenize
4. 按 assistant 回复 span 算 loss mask
5. 和 token id 一起落盘
6. 统计 token 频率，留给后面用

loss mask 保证训练只盯 **机器生成** 的 token。推理模型常常只在 **最后一轮** 插入 thinking token；Speculators 另给一面旗，随机丢掉若干轮，让模型见到长短不一的对话。

hidden 生成器走 vLLM 插件：自定义 worker 扩展，在 **Prefill** 阶段打补丁截中间 hidden。用 vLLM 的多进程 executor 做批量推理；大模型可以开 tensor parallelism。

![hidden state generator](../../../../assets/vllm/blog/performance/speculators-v030/02-hidden_state_generator.png)

**Figure 2。** Prefill 里截 hidden：插件 worker 拦 forward，多进程 executor 往外送。

保存阶段，每条样本一个 `.pt`：

- `input_ids`：tokenize 后的输入
- `hidden_states`：按捕获层分开的张量列表
- `loss_mask`：哪些 token 可训练

生成器一边继续抽 hidden，一边用 `ThreadPoolExecutor` 做异步 I/O 写盘，吞吐尽量拉满。

另外两份文件：

- `data_config.json`：这次数据生成的元数据
- `token_freq.pt`：token 频率

`token_freq.pt` 用来建 target-to-draft（**t2d**）和 draft-to-target（**d2t**）映射：verifier 的全词表收到草稿更小的词表。缩小的 draft vocab 只留最常出现的 token，草稿更省。

离线脚本：

- [`data_generation_offline.py`](https://github.com/vllm-project/speculators/blob/main/scripts/data_generation_offline.py)：预处理、存频率、抽 hidden
- [`build_vocab_mapping.py`](https://github.com/vllm-project/speculators/blob/main/scripts/build_vocab_mapping.py)：建 t2d / d2t 张量

### 训练

v0.3.0 训 Eagle3 草稿。输入是上一步的样本和词表映射，加上模型配置，初始化一只 `Eagle3DraftModel`。训练手法来自 Eagle3 作者，叫 **train-time-testing**：训练时模拟多步草稿采样，让模型不只会猜第一个 token，后面几步也要会。

![flex attention](../../../../assets/vllm/blog/performance/speculators-v030/03-flex_attention.png)

**Figure 3。** Eagle3 论文里的 train-time-testing 与逐步 attention mask（[Zhang et al., 2025](https://arxiv.org/abs/2503.01840)）。每个前缀先猜下一步（蓝），再在「前缀 + 第一步」上猜第二步（黄），以此类推。

难处：attention mask **稀疏**，普通 attention 又算又占显存。Speculators 用 FlexAttention（[He et al., 2024](https://arxiv.org/abs/2412.05496)）：把 mask 切成块，只算非空区域。再叠 `torch.compile`，前向更快，反向要的 activation VRAM 也小很多。

另一件：batch。序列长短不一。一条路是截断 + padding，长度齐整的数据还行，padding 多了就浪费算力。v0.3.0 走第二条：沿 **sequence** 维拼接，再用 attention mask 把它们当成互不相干的序列。这和 FlexAttention 合得上；再配会把样本打包到接近 max sequence length 的采样，效率更好。

这些拼在一起，Eagle3 训练又快又省显存，入口是一只 [`train.py`](https://github.com/vllm-project/speculators/blob/main/scripts/train.py)。

## 在 vLLM 里跑 Speculators 模型

训完，库吐出完整产物：`config.json` 里多一段 `speculators_config`。然后短命令就能 serve：

```bash
vllm serve RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3
```

vLLM 读 `speculators_config` 里的投机设置（例如 verifier 名字），把草稿和 verifier 装进同一只 server。标准化配置让模型 **自己知道该怎么跑**；部署投机解码跟部署普通 LLM 一样短。细节见文末 [附录](#speculators_config)。

短命令适合上手。要更细的控制，用长语法：换 config 里那只 verifier、拧投机参数（例如猜多少 token）。长命令 serve 的是底座 verifier，草稿走 `--speculative-config`。例如换成量化 verifier：

```bash
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"model": "RedHatAI/Qwen3-8B-speculator.eagle3", "num_speculative_tokens": 5, "method": "eagle3"}'
```

这里 verifier 是 FP8 的 Qwen3-8B（而不是 `speculators_config` 里默认的 BF16），投机 token 从默认 **3** 加到 **5**，吞吐或许更高——仍是页上的旋钮，不是承诺。

## 生产接入：投机解码进 serving

Speculators 和 vLLM 绑紧之后，投机解码从研究技巧变成能上生产的功能。vLLM 的 Eagle3 支持跨一批架构：

**vLLM serving + Speculators 训练：**

- Llama（3.1、3.2、3.3）：8B 到 70B
- Qwen3：8B、14B、32B
- Qwen3 MoE：235B-A22B
- GPT-OSS：20B、120B

**当时只 serving：**

- 多模态：Llama 4 视觉语言模型

## 下一步

当时规划：

- **在线** 数据生成（训练时抽 hidden，不在中间落盘缓存）
- 视觉语言模型的数据生成
- 用 verifier **重写** 数据集里的 assistant 回复，让训练数据和 verifier 更齐

在线这条后来在 [v0.5.0](speculators-v050.md) 落地，hidden 改走 [extract-hidden-states](../architecture/extract-hidden-states.md)。

## 参与

仓库：[Speculators](https://github.com/vllm-project/speculators)。[Good First Issues](https://github.com/vllm-project/speculators/issues) 欢迎新补丁。

- **文档**：https://docs.vllm.ai/projects/speculators/en/latest/
- **vLLM Slack**：`#speculators`、`#feat-spec-decode`
- **数据生成与训练脚本**：https://github.com/vllm-project/speculators/blob/main/scripts/README.md
- **端到端例子**：https://github.com/vllm-project/Speculators/tree/main/examples/data_generation_and_training
- 已训好的模型：[Red Hat AI Hub](https://huggingface.co/collections/RedHatAI/speculator-models)

## 附录

### Eagle3 算法

![Eagle3 Algorithm](../../../../assets/vllm/blog/performance/speculators-v030/04-EAGLE3.png)

**Figure 4。** Eagle3：三层 verifier hidden 进草稿，再自回归猜 token。

### `speculators_config`

页上的例子（键名大小写按原文）：

```yaml
{
  "architectures": ["Eagle3Speculator"],
  "auto_map": {"": "eagle3.Eagle3SpeculatorConfig"},
  "Speculators_model_type": "eagle3",
  "Speculators_version": "0.3.0",

  "draft_vocab_size": 10000,
  "transformer_layer_config": {
    "num_hidden_layers": 1,
    "hidden_size": 4096,
    ...
  },

  "Speculators_config": {
    "algorithm": "eagle3",
    "proposal_methods": [{
      "proposal_type": "greedy",
      "speculative_tokens": 3,
      ...
    }],
    "verifier": {
      "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
      "architectures": ["LlamaForCausalLM"]
    }
  }
}
```

这份配置把 speculator 定义成完整模型：

- **身份：** `architectures`（例如 `Eagle3Speculator`）；`auto_map`（Hugging Face 自定义加载）；`Speculators_model_type`
- **草稿结构：** `transformer_layer_config`；`draft_vocab_size`（缩小词表，原文典型 **10k–32k**）；以及模型相关选项
- **投机解码：** `algorithm`（EAGLE3）；`proposal_methods`（`speculative_tokens`、`verifier_accept_k`、`accept_tolerance`）；`verifier` 的 `name_or_path` 与 `architectures`（兼容性检查）
