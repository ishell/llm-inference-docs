---
source: https://vllm.ai/blog/2026-03-24-mrv2
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# Model Runner V2：给执行核换一块更干净的底板

英文对照：`en/vllm/blog/architecture/mrv2.md`  
原文：https://vllm.ai/blog/2026-03-24-mrv2  
2026-03-24。V1 是引擎骨架；MRV2 是 **model runner** 的重写。用户 API 不变。当时还不是默认：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

他们计划不久后默认打开。文中「还不支持」以 **v0.18.0** 为准，会过时。图在原网页。

## 为什么又拆一次

V1 发布后，runner 上继续叠 async scheduling、投机解码。单独看每块都有用，合在一起开始缠：

- persistent batch 的状态和每步输入绑死，增删改序像在一张挤满人的长桌中间换座位。
- async 是后装的，许多特性要绕路才能跟它共存。
- 记账仍在 CPU 上：GPU 越快，这些小手术越显眼。
- 新模型、新特性越来越难干净地伸进去。

三条原则：把模型相关的逻辑从公共路径里隔离（**modular**）；把记账搬到 GPU 上（**GPU-native**）；把 CPU/GPU 重叠当成设计约束，而不是补丁（**async-first**）。

## 新在哪里

**Persistent batch 与 GPU 上的输入准备。** V1 把 persistent 状态直接当模型/sampler 输入，布局约束别扭。MRV2：每个活着的请求在固定大小的状态表里占**稳定的一行**；每步按当前顺序 gather 出这一拍需要的张量。不再需要 `CachedRequestState` 那种备份。输入准备用 Triton kernel 在 GPU 上做：`input_ids`、`positions`、`query_start_loc`、`seq_lens` 直接在设备上长出来。少 Python、少同步，投机解码的 rejection 结果可以留在 GPU 上被下一拍直接吃掉。

**Async 当作默认宇宙。** 调度器和 worker 在 GPU 跑第 N 步时准备 N+1。MRV2 的目标是：所有支持的模型/特性组合，CPU 与 GPU 之间 **零同步**。输出走另一条 CUDA stream 异步回 CPU。结构化输出 + 投机解码也走同一条路。

**Triton sampler。** Gumbel-Max 不必物化整份 softmax，kernel 内无状态 RNG；先找 top-k logits 再算 logprobs；prompt logprobs 更细的切块；投机解码用 `idx_mapping` 间接寻址，不必为每个 logits 向量膨胀请求状态。峰值显存下降，采样参数更好组合。

**ModelState。** 抽象出模型相关的 `add_request` / `remove_request` / `get_mm_embeddings` / `prepare_inputs` / `prepare_attn` / `prepare_dummy_inputs`……主 runner 只管公共路径。旧 `gpu_model_runner.py` 超过 **6700** 行；MRV2 最大文件压到 **1300** 行以下。只关心 DeepSeek 或某一家内模的人，不必再读完整座迷宫。

## 成绩（他们故意挑的极端）

很小的模型 × 很快的卡，好让 host 开销显得胖：Qwen3-0.6B on **1×GB200**，吞吐 **16K → 25K** output tok/s，大约 **+56%**。

投机解码：GLM-4.7-FP8、MTP=1、**4×GB200**，mean TPOT 大约 **−6.3%**。来自「打开 spec decode 也不再插入 CPU–GPU 同步点」。

## 当时还不支持（v0.18.0）

线性注意力（Qwen3.5、Nemotron 3 Super）；Eagle/Eagle3/MTP 以外的投机方法；EPLB、DBO；logits processors；LoRA。完整清单在设计文档第二页。他们说搬 V1 特性进 MRV2 时要从第一性原理重想，所以会慢——抄复杂度进来，就白拆了。

读完 V1 再读这一篇：骨架立住以后，下一步瓶颈又回到「每一步怎么把 batch 摆上 GPU」。Anatomy 里的 model runner，指向的就是这块正在被换掉的底板。
