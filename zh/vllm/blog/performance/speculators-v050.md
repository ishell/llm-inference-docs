---
source: https://vllm.ai/blog/2026-05-28-speculators-v050
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Speculators v0.5.0：DFlash 与在线 hidden

英文对照：[en/vllm/blog/performance/speculators-v050.md](../../../../en/vllm/blog/performance/speculators-v050.md)  
原文：https://vllm.ai/blog/2026-05-28-speculators-v050  
2026-05-28。署名 **Fynn Schmitt-Ulms、Helen Zhao、Rahul Tuli and Dipika Sikka（Red Hat AI Model Optimization Team）**。发版：[v0.5.0](https://github.com/vllm-project/speculators/releases/tag/v0.5.0)。学习笔记。上一拍离线 EAGLE-3：[v0.3.0](speculators-v030.md)。hidden 不再钩引擎内部，改走 [extract-hidden-states](../architecture/extract-hidden-states.md)（`vllm>=0.18.0`）。DFlash 和并行草稿族一起读：[parallel-drafting](parallel-drafting.md)。页上的 Gemma 4 数字是他们的评测，不是你的 SLA。

v0.5.0 把投机解码训练的架构往前推了一截：DFlash、统一的在线训练、以及彻底迁到 vLLM 原生 hidden 抽取。训练更活，也更像能上生产的工作流。

要点：

- **DFlash**：一次前向、块扩散出全部 draft token
- Gemma 4 的 DFlash 成绩
- 在线 / 离线训练共用 vLLM 原生 hidden 抽取
- 文档和例子按关键工作流重写过

## DFlash

相对自回归的 Eagle 3，DFlash 是另一条草稿路。Eagle 3 多步前向、一个一个猜；DFlash 用 **block diffusion**，**一次前向**吐出整块草稿。

一次前向能把投机解码的草稿税压下去，尤其草稿序列更长的时候。每个前缀，草稿吐出长度 **B** 的一块。块结构完全靠 attention mask。和 Eagle3 另一处不同：块内是 **非因果**——同一块里的 query 可以看见块内所有 token。

训练时多块并行。最笨的做法：序列每个位置都开一块预测。序列一长，attention mask 炸开，显存和算力都撑不住。所以他们 **并不到处开块**：只在真正贡献 loss 的位置里，随机抽一小撮 **anchor**，预测块只挂在这些锚上。块数与序列长度脱钩，上下文可以更长，mask 仍管得住。

## 训一只 DFlash speculator

工作流和 Eagle 3 的在线训练相近。教程：[train DFlash online](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_dflash_online/)。

和 Eagle 3 的差别主要在训练命令里的 speculator 参数：

```bash
torchrun --standalone --nproc_per_node 2 scripts/train.py \
    --verifier-name-or-path "Qwen/Qwen3-8B" \
    --vllm-endpoint "http://localhost:8000/v1" \
    --speculator-type dflash \
    --draft-vocab-size 8192 \
    --block-size 8 \
    --max-anchors 3072 \
    --num-layers 5 \
    --target-layer-ids "2 18 33" \
    --epochs 5 --lr 1e-4
```

DFlash 专用：

```bash
--block-size # 每个扩散块吐多少 token
--max-anchors # 训练时最多多少个投机锚点
--speculator-type # 必须是 dflash
```

## Gemma 4 的 DFlash

按这套算法训出 [Gemma 4 31B DFlash speculator](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dflash)，在多种任务上量接受率。原文说推理和代码生成上尤其强——正文 **没有** 把柱状图读成表，不要从图里编数字。

本地图（原文版权仍归原站；学习对照用）：

![gemma4 dflash acceptance rates](../../../../assets/vllm/blog/performance/speculators-v050/01-gemma4-dflash-acceptance-rates.png)

**Figure 1。** Gemma 4 DFlash 在不同任务类型上的接受率。

Gemma 4 DFlash 的 inter-token latency 好过 Eagle 3，也好过单独一只 FP8 量化 verifier。再把 DFlash 叠到 FP8 verifier 上，ITL 更短：

![gemma4 dflash latency](../../../../assets/vllm/blog/performance/speculators-v050/02-gemma4-dflash-latency.png)

**Figure 2。** Gemma 4 DFlash 的 inter-token latency 对照。页上没有把毫秒写成表。

## 在 vLLM 里 serve DFlash

DFlash 接到 vLLM 的投机解码基础设施，自 PR [#38300](https://github.com/vllm-project/vllm/pull/38300)，含在 `vllm>=0.20.0`。

和 Eagle 3 一样，`config.json` 里有 `speculators_config`：target 模型、投机 token 数、算法名等。有了这份配置，短命令即可：

```shell
vllm serve -tp 2 RedHatAI/gemma-4-31B-it-speculator.dflash
```

## 统一的在线 / 离线训练

v0.5.0 给两种模式都接上 [vLLM 的 hidden 抽取](https://vllm.ai/blog/extract-hidden-states)（`vllm>=0.18.0`）。以前 Speculators 用更底层的 vLLM 工具抽 hidden，训练管道把 vLLM 当成 **直接的 Python 依赖**。内部 API 一改就要手工对齐。这次拆掉自定义数据生成管线，**不再**把 vLLM 当直接 Python 依赖。

两种模式走同一条 vLLM 抽取路径：

- **在线：** 训练时当场抽 hidden
- **离线：** 先生成、缓存到盘，再训

借引擎的抽取，也就借到 vLLM 的推理优化：显存、batch、硬件加速。训练进程跟一只正在跑的 vLLM server 走标准 REST API 说话，不再扣内部实现。vLLM 和 Speculators 可以各自升级。

在线训练时发生的事：

1. vLLM server 用底座模型起来（带一些特殊配置）
2. 训练 prompt 送到 vLLM 做推理
3. hidden 抽出来，暂时写到盘（或 ram disk）
4. 训练进程读走，删文件
5. speculator 在抽到的状态上训练

在线教程：[train Eagle3 online](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/)。

离线数据生成也改成同一套抽取和同一数据格式。新脚本把请求灌满正在跑的 vLLM server，再写盘。两条路绑得紧，可以混用：先离线生成一部分，训练时已有的直接加载、缺的再在线补；也可以让在线作业 **不删** 文件，第一 epoch 生成，后面几个 epoch 复用。

离线教程：[train Eagle3 offline](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_offline/)。

## 文档

[文档站](https://docs.vllm.ai/projects/speculators/en/latest/) 重写过：各算法的短介绍、训 speculator 的教程。给开发者的：怎样往库里加新投机算法，以及一份 API reference。
