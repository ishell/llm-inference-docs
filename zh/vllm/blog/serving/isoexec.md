---
source: https://vllm.ai/blog/2026-08-21-isoexec
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# IsoExec：同一份执行合同盖住训练和推理

英文对照：[en/vllm/blog/serving/isoexec.md](../../../../en/vllm/blog/serving/isoexec.md)  
原文：https://vllm.ai/blog/2026-08-21-isoexec  
2026-08-21。Alexander Jiang 与 SkyRL 团队。仓库：[zanderjiang/SkyRL-IsoExec](https://github.com/zanderjiang/SkyRL-IsoExec)。更早的「两份模型、对齐 kernel」：[bitwise-rl](bitwise-rl.md)。这条训练环仍坐在上面的 pause / 权重 API：[native-rl](native-rl.md)。

SkyRL × vLLM × Megatron。单机 **8×H100**，同步 **Qwen3.5-35B-A3B** DAPO：合同覆盖区域的平均 rollout–train logprob 差压到 **1e-6 以下**，相对当时 SkyRL 基线约 **25%** 墙钟（50 step）。

## TL;DR

理论上，on-policy RL 假定 rollout 和训练评的是 **同一份** 政策。实际上两套引擎：模型定义、kernel、batch 形状、并行布局都不一样。浮点不可结合，于是「同一政策」也会走出不同的 token 概率。新算法、harness / 环境改动、kernel / 硬件改进，调试都会变难。

**IsoExec** 是跨框架的统一执行抽象。两件套：

1. **Execution contract**：把会影响舍入的细节写死，并在两边引擎上强制执行。
2. **统一模型**：对齐的、batch-invariant 的 kernel，训练和 rollout 逐 bit 一致。

落在 SkyRL 里，推理 vLLM、训练 Megatron。一台 8×H100，同步 Qwen3.5-35B-A3B DAPO，端到端平均 rollout–train logprob 差 **低于 \(10^{-6}\)**，相对当时 SkyRL 基线 **25%** 开销，窗口 **50** step。

原文列的贡献：

- **统一执行合同：** 训练和推理共用一份数值合同；合同覆盖区域内 mismatch 为零；算法、环境、kernel 一改，调试成本仍低。
- **并行不变 kernel：** 数值在 tensor / expert / sequence parallelism 下仍保住。
- **CPR Gated DeltaNet：** 训练、Prefill、recurrent Decode 对齐，不必把长序列前向串成一条。

## Introduction

RL 要把同一政策执行两遍：rollout 引擎按 \(\mu\) 采一个 token；trainer 稍后用同一份参数按 \(\pi\) 重算 logprob。同步 on-policy 假定 \(\mu = \pi\)。系统上这很难，因为

\[
(a+b)+c \neq a+(b+c).
\]

常见拼法：推理用 vLLM、SGLang，训练用 Megatron、FSDP。Kernel、batch 形状、执行模式（训练、Prefill、Decode）、分布式布局一变，归约顺序就变，token 分布就变。

字节跳动 [VeXact](https://arxiv.org/abs/2605.14220)：单凭 mismatch 就能让 REINFORCE / GRPO 不稳，在 KL 估计器反应之前就扭曲按 advantage 加权的损失，让 IS / rejection 一类修补对校准过敏。[Fireworks](https://fireworks.ai/blog/frontier-lab-training-infrastructure-as-a-service)：一次 GLM-5.2 run，train–inference KL 大约 **0.013**，clip 掉约 **45%** 的 token，奖励在大约 **第 20 step** 塌掉；逐 bit 对齐的那次 **零** clip，稳住了。

前人切过这个问题的几片：

- [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) —— batch invariance：batch 里别的元素、batch 大小，都不该改某一个元素的计算。
- [vLLM × TorchTitan](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)（[bitwise-rl](bitwise-rl.md)）—— 两边引进同一套 kernel；仍是 **两份对齐过的模型代码**。
- [线性 attention 与异步 RL 的零 mismatch](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) —— TorchTitan 和 vLLM 共用一份模型定义；[Gated DeltaNet](https://arxiv.org/pdf/2412.06464) 的前向全部走 recurrent，chunked kernel 只留给 backward。
- [Tree-Based Invariant Kernels (TBIK)](https://arxiv.org/abs/2511.17826) —— 不同并行配置下仍逐 bit 一致。

IsoExec：合同 + 统一模型。合同把舍入敏感的选择（kernel、累加 dtype、归约顺序）写成与框架无关的条款，再在各 runtime 上强制。统一模型用的 kernel 已在训练、Prefill、Decode 上验证过逐 bit；同时仍接到 vLLM 的 scheduler / KV manager / CUDA graph capture，以及 Megatron 的训练栈。

## Unified execution contract

合同声明每一个会影响 bit 的执行选择，两边必须写成一样。

```jsonc
"ExecutionContract": {
  "cases": [ ... ],        // logprob computations: trainer_fwd, engine_decode, etc.
  "composition": [ ... ],  // (region, case) -> implementation + pinned constants
  "claims": { ... },       // topology invariance, state invalidation, tolerances
  "identities": { ... }    // semantic / numerical_policy / deployment digests
}
```

每次算 token logprob 是一个 **case**（rollout 的 `engine_prefill`、trainer 的 `trainer_fwd`，……）。前向算子划成 **region**：一段算术，由一只可能融合了多步的 kernel 实现。每个 `(region, case)`，**composition** 选定实现，以及钉死的常量——累加 / 边界 dtype、split-K 和 split-KV 的分区数，凡是能改 bit 的都算。一个 region 必须先在各 case 上测过逐 bit，实现才准登记。

例如：

```jsonc
"composition": [
  {
    "region": ["gdn.core", "gdn.gating", "norms.l2"],
    "cases": ["trainer_fwd", "trainer_fwd_no_autograd", "engine_prefill", "engine_decode"],
    "impl": {"id": "native_fused_sigmoid", "version": 1, "arch": "sm90"}
  },
  {
    "region": ["moe.combine"],
    "cases": ["engine_prefill", "engine_decode"],       // the trainer side is its own entry
    "impl": {"id": "pik_leaf_tree", "version": 2, "arch": "sm90"},
    "constants": {"leaves": 8, "leaf_dtype": "fp32"},
    "discharge": {"kind": "equivalence_proof", "ref": "gates/ep_invariant_combine"} // proved equivalence
  }
]
```

**Claims** 写清这些保证在什么条件下成立，运行时强制。比如一条 topology claim 列出归约树被证明逐 bit 不变的并行规模。装 kernel 时，adapter 拿 runtime 真实并行规模去对那张名单，**没被证明过的规模直接拒绝**。

**Identities** 是序列化合同的 SHA-256，用来核对 trainer 和 rollout 是否谈妥：

- `semantic` —— 同一份逻辑模型。
- `numerical_policy` —— 每一个能影响数值的执行选择（实现、版本）。
- `deployment` —— 已证明 **不影响** bit 的设置（内存大小、传输）。合同 **不要求** 两边一致。

`semantic` 和 `numerical_policy` 对上，再加上 adapter 强制，就是原文说的：两边在覆盖区域上跑的是同一份已验证的数值政策。

本地图（原文版权仍归原站；学习对照用）：

![unified execution abstraction](../../../../assets/vllm/blog/serving/isoexec/01-unified_execution_abstraction.png)

IsoExec 盖在训练和推理 runtime 上的统一执行合同。

每个 runtime 一只 **contract adapter**：把合同和实现装进去并强制——把每条 composition 绑到框架的扩展点（比如选哪只 attention kernel），再盯已装 kernel、已声明 claims、跨进程 identity digest。

## Unified model

SkyRL 这边的统一模型：batch-invariant 的 GEMM、attention、normalization，加上确定的 MoE routing 和 combine。在 tensor / expert / sequence-parallel 布局下仍逐 bit；GDN hybrid 再用 chunkwise-parallel recurrent。

实验里，这套抽象在 dense（**MiMo-7B**）、MLA MoE（**GLM-4.7-Flash**）、hybrid（**Qwen3.5-9B**）、hybrid MoE（**Qwen3.5-35B-A3B**）上做到 **合同覆盖区域 mismatch 为零**。实现：[SkyRL-IsoExec](https://github.com/zanderjiang/SkyRL-IsoExec)。

### Parallelism-invariant kernels

训练和推理想要的布局并不一样。Trainer 要装下 optimizer state、activation、gradient，MoE 还要分散的 expert 权重。Rollout 引擎要的是 KV 容量，还不能伤 Decode 延迟。

输入和权重固定时，前向数值会被这六根轴碰到：

- **DP** —— 切 batch；batch-invariant kernel 保住每条 sample 的数值。
- **PP** —— 整层在设备间搬家；边界 dtype 钉死后，层内归约并不被切开。
- **TP** —— 把 contraction 的归约拆到各 rank。树不固定就会 **改 bit**。
- **EP** —— 把 expert 计算铺开，也改 expert 输出怎么 combine。**改 bit。**
- **SP** —— 行并行归约从 all-reduce 变成 reduce-scatter。**改 bit。**
- **CP** —— 沿序列维切开 attention 归约。**改 bit。**（这里的不变性是下一步，这篇还没认领。）

[TBIK](https://arxiv.org/abs/2511.17826) 给 TP-invariant 推理钉死一棵全局归约树，罩住行并行 GEMM 和跨 GPU 归约。IsoExec 还是固定树，但沿 **K** 维来。`pik` 把 K 切成 \(G\) 段连续的 **leaf**。每片 leaf 用确定的 Tensor Core MMA，**FP32** 累加。合同钉死 rank→leaf 映射和二叉树算术日程；**部分和走 NCCL** —— 不必自研通信 kernel。

![pik figure](../../../../assets/vllm/blog/serving/isoexec/02-pik_figure.png)

`pik` 用来在不同并行布局下保住数值的那棵固定二叉树。

EP 和 SP 同一原则：

- **EP：** 按 **固定 routing 序** combine expert 输出，不按 rank 序。
- **SP：** 复用非 SP 的那棵归约树；每个 rank 只留自己那片输出，不必 gather 全量。Trainer logits 在 SP 开或关时逐 bit 相同。

### Chunkwise-parallel recurrent (CPR) GDN

线性 attention 更烦：训练和推理用的算法就不一样。现成的 GDN 栈，训练和 Prefill 走 **chunkwise-parallel**，Decode 走 **recurrent**。数学上等价，舍入不是。拿 [FLA](https://github.com/fla-org/flash-linear-attention) 的 chunkwise-parallel kernel 对 vLLM 融合 recurrent kernel：逐元素平均绝对差大约 **\(1.7 \times 10^{-2}\)**，最大 **0.25**。

[TorchTitan 那篇](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) 让 rollout Prefill **和** trainer 前向都走 recurrent，chunkwise 只留给 backward。GDN mismatch 没了，Prefill / trainer 前向却按序列长度串起来：数学负载大约 **2–3×**，terminal-agent 负载大约 **5×** —— 撑不起整场训练。他们表上全程 recurrent 会把 Prefill 拉到 **4×+**。

**CPR** 仍以 recurrence 当主函数，但按 chunk 并行求：

- 训练 / Prefill：和 chunkwise-parallel 一样，先算每个 chunk 边界上的 recurrent 状态；再在块内做并行 recurrent scan，填输出。
- Decode：仍是 recurrent，但每 \(C\) 个 decode 出来的 token **再对齐一次 hidden state**（\(C\) = chunk size），于是 Prefill、训练、Decode 共用一份舍入日程。

单层代价，**H100**，\(C=64\)。Trainer 和 rollout 引擎用各自生产上的 TP 布局和 kernel。倍数相对该阶段的 native mixed（越小越好）：

| 阶段 | 形状 | Native mixed | 全程 chunkwise | 全程 recurrent | CPR |
| --- | --- | --- | --- | --- | --- |
| 逐 bit | — | 否 | 是 | 是 | **是** |
| Trainer forward + backward | 1 × 10,240 tokens | 5.177 ms | 5.177 ms (1.00×) | 22.863 ms (4.42×) | **7.386 ms (1.43×)** |
| Rollout-engine Prefill | 5 × 2,048 tokens | 0.844 ms | 0.844 ms (1.00×) | 3.639 ms (4.31×) | **1.412 ms (1.67×)** |
| Rollout-engine Decode | 256 sequences × 1 token | 0.0612 ms | 2.2374 ms (36.6×) | 0.0612 ms (1.00×) | **0.0846 ms (1.38×)** |

## Results

IsoExec 对 SkyRL 原生栈，单机 **8×H100**，**Qwen3.5-35B-A3B**，DAPO-Math-17k，**同步** RL。其余对齐。IsoExec 相对当时评估过的最高吞吐同步 RL 配置（vLLM + Megatron）端到端大约 **25%** 开销。

![result logprob diff](../../../../assets/vllm/blog/serving/isoexec/03-result_logprob_diff.png)

Rollout 对训练的绝对 logprob 差：原生 SkyRL 对 IsoExec。

**50** step 里，更新前 rollout–train 绝对 logprob 差：

| | Native | IsoExec |
| --- | ---: | ---: |
| 均值 | \(1.648 \times 10^{-2}\) | \(6.744 \times 10^{-7}\) |
| 标准差 | \(4.035 \times 10^{-2}\) | \(6.821 \times 10^{-7}\) |
| 逐步最大的平均 | 5.073 | \(7.358 \times 10^{-6}\) |

覆盖区域均值 **低于 1e-6**。

![result time](../../../../assets/vllm/blog/serving/isoexec/04-result_time.png)

同一 50 step 窗口的平均 RL step 耗时。

| 指标 | Native | IsoExec | 开销 |
| --- | ---: | ---: | ---: |
| Generation | 591.3 s | 776.6 s | **31.3%** |
| Policy training | 498.6 s | 591.3 s | **18.6%** |
| Full RL step | 1224.6 s | 1534.0 s | **25.3%** |

标题里的 **25%**，就是这整步的 **25.3%**。

![result reward](../../../../assets/vllm/blog/serving/isoexec/05-result_reward.png)

50 step 的 Pass@16 和 raw reward。短跑里，杀掉合同覆盖的 train–inference mismatch **没有** 明显的奖励提升——太短，看不出稳定红利。

## Next steps

- **Blackwell support**
- **Context parallelism invariance**
- **Sparse attention**
- **Block-FP8 MoE**

## Acknowledgements

[Alexander Jiang](https://www.linkedin.com/in/akj2) 与 SkyRL 团队。感谢 [Charlie Ruan](https://www.charlieruan.com/)、[Sumanth Hegde](https://sumanthrh.com/about/)、[Eric Tang](https://erictang000.github.io/)、[Philipp Moritz](https://www.linkedin.com/in/philipp-moritz-61419682)、[Yichuan Wang](https://yichuan-w.github.io/)、[Mayank Mishra](https://www.mayank.site/)、[Lingxiao Ma](https://xysmlx.github.io/) 的讨论。
