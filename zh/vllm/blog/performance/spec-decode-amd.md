---
source: https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# AMD GPU 上的投机解码：五条草稿路

英文对照：[en/vllm/blog/performance/spec-decode-amd.md](../../../../en/vllm/blog/performance/spec-decode-amd.md)  
原文：https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus  
2026-08-23。署名页上致谢的 **AMD and Embedded LLM**。学习笔记。bench 在 Instinct **MI300X / MI355X**、ROCm 上；disclaimer 里的快照：vLLM `0.23.1rc1.dev1120+g0f0f28b53`，ROCm/HIP `7.2.53211`。数字是他们那套环境，不是你的 SLA。

验收数学仍是 [投机解码主线](spec-decode.md)。草稿怎么长：[并行草稿](parallel-drafting.md)（P-EAGLE / DFlash / DSpark）、[P-EAGLE](p-eagle.md)。DSpark 按信心改验收预算（**这批 AMD 实验没开**）：[DSpark 自适应](dspark-adaptive.md)。后来 EAGLE 注意力漂移的修法：[eagle-3-1](eagle-3-1.md)。训练用的 hidden 导出：[extract-hidden-states](../architecture/extract-hidden-states.md)。同一代 GPU 上的 ROCm attention：[rocm-attention](../architecture/rocm-attention.md)。

适用：在 ROCm 上开五条投机路、对照 `N`、读他们扫出来的输出 TPS。不适合：把页上的 **2.87×** 当承诺，或指望这里重写验收公式。

**原文 TL;DR。** 投机解码让 vLLM 用 **一次** target 前向核好几枚草稿 token。输出 token 吞吐跟着草稿方法、提议长度 `N`、模型家族、draft checkpoint、负载和接受行为走。测到的上沿：**2.87×** DFlash on `gemma-4-26B-A4B-it`，**2.83×** Gemma 4 MTP 同一 target，**2.68×** DFlash on Kimi-K2.5。有的 sweep 几乎没赚，甚至掉到基线以下。这篇是 **ROCm 上怎么开、怎么量**。

## Introduction

Serving 的底座仍是标准自回归：吐一个 token，接上去，再吐下一个。简单、可靠；输出必须严格从左到右，所以循环一次只提交一个字。

投机解码 [[1]](#ref-1) 保住这份输出行为，把循环拆成 **draft** 和 **verify**。轻量草稿组件先猜未来候选；原模型当 **target**，提交前再核对。几枚草稿活下来，一次 target 验收就能提交多个输出 token。

下文先复习自回归基线和 draft-and-verify，再看五条草稿路——从 target 拿什么、候选是顺序、自回归、并行还是杂交：**Native MTP**、**Gemma 4 MTP**、**EAGLE-3**、**DFlash**、**DSpark**。然后 CLI、现成 draft、MI300X / MI355X 上的测量、拧 `N`、以及一小节训练。

## 自回归 Decode 基线

每一步 Decode 生产并提交 **一个** 新 token。四个输出 token → 四步顺序 Decode：

1. `context` → model → T1
2. `context + T1` → model → T2
3. `context + T1 T2` → model → T3
4. `context + T1 T2 T3` → model → T4

生成的 token 接到序列里，成为下一步的输入。长生成时，这个一字一步的环会吃掉延迟、卡住 serving TPS。

投机解码要问的就是：能不能保住原模型的输出行为，同时少做「一次只往前一个 token」？

把提议和验收分开。草稿先猜一串候选；target 提交前再验。

## 投机解码的核心

原模型不换掉。它仍是 target，仍对最终输出负责。前面只多一级更快的提议。

- **Draft：** 提出若干未来候选。先不提交。
- **Verify：** target **一次** 前向核完整串候选。

验收从左到右。每个草稿 token 对那个位置核对。接受的提交。**第一处拒绝** 截断这一轮：后面的候选丢掉，拒绝位由 target 自己给 token。生成从更新后的序列继续。

一轮例子：草稿猜 T1 T2 T3 T4；target 接受 T1、T2，拒绝 T3，丢掉 T4；提交是 T1、T2，外加 target 的 **替补 token**。对上则一次验收提交多个输出；没对上仍由 target 决定下一枚。相对 target **无损**。

![figure 01](../../../../assets/vllm/blog/performance/spec-decode-amd/01-figure-01.svg)

**Figure 1。** 草稿提出未来候选；target 验收之后才提交输出 token。

### 接受 / 拒绝的小例子

Figure 2 里绿框是活过验收的草稿，红框是第一处拒绝，灰框是拒绝之后丢掉的候选。输出里的蓝 token 来自 target，不是草稿。

![figure 02](../../../../assets/vllm/blog/performance/spec-decode-amd/02-figure-02.svg)

**Figure 2。** 从左到右验收。前两枚接受；拒绝位用 target token；其余丢掉。

Prompt：`The weather today is`。草稿猜：`sunny`、`and`、`warm`、`outside`。

| | pos 1 | pos 2 | pos 3 | pos 4 |
| --- | --- | --- | --- | --- |
| draft proposes | sunny | and | warm | outside |
| model verifies | ✓ | ✓ | ✗ | stop |
| commit | sunny | and | **clear**（target） | — |

`sunny` 和 `and` 接受。第三位草稿说 `warm`，target 选了 `clear`。`outside` 跟在第一处拒绝后面，丢掉。下一轮从 `The weather today is sunny and clear` 接着走。

## 五条草稿怎么工作

外壳都是 draft-and-verify；草稿组件不一样。差别主要是：从 target 拿什么信息、怎么折进草稿、候选是顺序还是并行。

三个桶（说的是 **草稿** 架构，不是 target 家族）。一只 target 可以自带 native MTP，同时又有另训的 EAGLE-3 / DFlash / DSpark。

- **Native MTP 模块。** 长在 target 里；模型自带的辅助预测路；候选顺序生成。
- **独立 MTP 草稿。** 单独 checkpoint，绑一只特定 target；推理时吃 target 激活、共享 KV；候选仍顺序。Gemma 4 MTP 落在这里。
- **专门的 target 条件草稿网。** 为某只 target 训的 speculator：EAGLE-3（对着 target hidden 自回归）、DFlash（对着 target hidden 并行一块）、DSpark（DFlash 骨架 + 轻量因果修正 + 信心前缀）。

方法不同，草稿可能看到：target 的一份 hidden；若干选定层的 hidden；target 的 KV cache；或几路表示融在一起。

### Native MTP

Multi-Token Prediction：模型原生、预测「下一个」以外的 token。vLLM 里，target 带兼容的辅助预测组件才能开 native MTP [[2]](#ref-2)。各家族结构不同，都给一条猜未来的辅路。

第一步：MTP 把 target hidden 和当前 token 的信息合起来 → 第一枚草稿。后续步：新草稿 token + 上一 MTP 步的 hidden → 下一枚候选。凑够配置的 `N` 枚，target 一次核完。

常见融合：target（或上一 MTP）hidden **加** 移位输入 / 最新草稿 token 的 embedding → fusion / projection → 辅助预测层 → 草稿 logits。Hidden 带着前面的序列；embedding 标明从哪个 token 接着猜。沿 hidden 维拼起来。

`num_speculative_tokens` 和 **物理** MTP 层数不是一回事。`N` 大于头深度时，vLLM 多跑几轮 MTP 前向、复用那条路。`N` 更大 = 候选更多，也 = 顺序草稿税更重。

Native MTP 跟 target 架构绑死。共享组件往往让额外显存不大。多枚投机 token 仍要在验收前顺序草稿。

### Gemma 4 MTP

Gemma 4 把 MTP 草稿 **单独打包**，绑一只特定 target [[3]](#ref-3)。自己的 checkpoint，推理时仍贴着 target。

草稿用 target 已经算过的激活，并 **共享 target KV cache**，不必独自再走一遍已接受前缀。

草稿层数和配置的 `N` 仍分开。多枚候选 → 配对 MTP 组件顺序生成，再一次 target 验收。

### EAGLE-3

为特定 target 训的专用草稿网 [[4]](#ref-4)。自己的执行路径，条件在 target 内部。

Target 前向时，EAGLE-3 从 Transformer **三段** 记 hidden：靠前、中间附近、靠后。拼接 + 投影 → 一份融合的 target 特征。再和已采样 token 的 embedding 合起来，拼/投影进 EAGLE-3 draft decoder。

- 融合特征：已接受序列在 target 前向好几段上的样子。
- 采样 token 的 embedding：标明从哪个 token 接着草稿。

EAGLE-3 **自回归** 出草稿。第一枚：融合特征 + 采样 embedding。出了一枚之后，它的 embedding 喂下一草稿步。更后面的投机位还没有 target hidden（target 还没算过那些位置），所以 EAGLE-3 用上一轮 **草稿组件自己的输出** 往下走。

后面的草稿直接依赖前面已猜的 token。`N` 越大，验收前的顺序草稿越长。更长 `N` 上的注意力漂移是 [eagle-3-1](eagle-3-1.md) 的故事；这篇 AMD 帖按当时的 `method: eagle3` 来量。

### DFlash

也是专用草稿网，但和 MTP、EAGLE-3 不同：一整块未来位置 **一次并行** 预测 [[5]](#ref-5)。家族写法：[并行草稿](parallel-drafting.md)。

每块草稿以 **anchor** 开头：target 已经产出或确认过的已知 token。DFlash 不猜 anchor；它是后面 mask 位的起点。后续轮次里，这常常是上一轮验收多出来的那枚 target token。

长度 7 的例子：位置 0 是 `anchor`，位置 1–6 是 `mask`。一次 DFlash 前向把所有 mask 位一起预测：输出 `anchor, draft1 … draft6`。

和 EAGLE-3 一样，先把若干 target 层 hidden 融成一份（拼接 + 投影）。用法不同。EAGLE-3 把融合结果和采样 embedding 接到自回归草稿网的 **输入**。DFlash 把融合的 target 上下文变成额外的 **K/V**，供草稿网 **每一层** 使用。Mask 位上的 query 既可以看 target 来的 K/V，也可以看草稿块自己的 K/V。Target 上下文贯穿整张草稿网，不是只在入口喂一次。

块生成完，target 一次核完全部提议。从左到右接到第一处拒绝；其余丢掉；拒绝位换成 target token。所有 mask 位在 **一次** 草稿前向里一起预测（`draft1 … draft4` 一块出，不是 `draft1 → draft2 → …`）。同一 pass 里后面的位置 **并不** 条件在前面已采样的输出上——更后面几位好不好用，看 checkpoint 和负载，块一长更明显。

### DSpark

DSpark 在并行草稿上再加两件 [[6]](#ref-6)：

1. 轻量顺序头，让同一块里的 token 互相看见。
2. 按信心选出提交给 target 验收的前缀。

骨架：改过的 DFlash。一次并行前向给每个草稿位一份 hidden 和一份 base logits，target 条件方式和 DFlash 相同。

完全并行的草稿看不见同一块里更早选中的 token。几种续写都说得通时，组合会拧：`of course` 和 `no problem` 各自合理，按位置独立预测却可能拼出 `of problem`。

骨架之后，轻量 **Markov head** 从左到右选。位置 `k` 用紧邻的上一枚已选 token 造一个小 bias，加到骨架的 base logits 上 → 该位的调整分布。重的草稿网仍只跑一次；只有 Markov 头顺着块走。

设计里还有 **confidence head**，可以缩短送给 target 的前缀。**他们这批 vLLM 路径上没开这一项**，所以下面的数字只反映并行骨架 + Markov 修正。后来 CUDA 路上按信心改验收预算：[DSpark 自适应](dspark-adaptive.md)。

Target 仍一次核完整串；从左到右提交到第一处拒绝。

### 五条对照

Figure 3 并排：草稿长什么样、吃 target 哪路信息、顺序还是并行。五条都是 target 验一次；接受到第一枚拒绝为止。

![figure method summary](../../../../assets/vllm/blog/performance/spec-decode-amd/03-figure-method-summary.svg)

**Figure 3。** 五种投机解码的草稿结构和出 token 方式。

| 方法 | 草稿组件 | 用到的 target 信息 | 草稿怎么长 |
| --- | --- | --- | --- |
| Native MTP | 模型自带辅助 MTP 路 | Target 或上一 MTP 的 hidden + 当前草稿 token | 顺序，复用 MTP 路 |
| Gemma 4 MTP | 单独 MTP checkpoint，配对 target | Target 激活 + 共享 target KV | 配对 MTP 组件顺序生成 |
| EAGLE-3 | 专用自回归草稿网 | 前 / 中 / 后 target hidden，融成一份 | 顺序；已猜的 token 影响下一枚 |
| DFlash | 专用并行草稿网 | 融合 hidden 当额外 K/V，进每一草稿层 | 所有候选位一次并行前向 |
| DSpark | DFlash 式并行网 + 轻量 Markov 头 | 与并行网相同的 target 条件 | 一次并行前向，再轻量顺序调整 |

## 在 vLLM 里怎么开

走 `--speculative-config`。差在方法名、要不要单独 draft checkpoint、以及 `num_speculative_tokens`。页上点名当时支持的 method 值：`mtp`、`eagle3`、`dflash`、`dspark`。

| 方法 | 单独 draft checkpoint | 典型配置 |
| --- | --- | --- |
| Native MTP | 否 | `"method": "mtp"`，`"num_speculative_tokens": <N>` |
| Gemma 4 MTP | 是 | `"method": "mtp"`，`"model": "<matching-assistant>"`，`"num_speculative_tokens": <N>` |
| EAGLE-3 | 是 | `"method": "eagle3"`，`"model": "<matching-speculator>"`，`"num_speculative_tokens": <N>` |
| DFlash | 是 | `"method": "dflash"`，`"model": "<matching-speculator>"`，`"num_speculative_tokens": <N>` |
| DSpark | 是 | `"method": "dspark"`，`"model": "<matching-speculator>"`，`"num_speculative_tokens": <N>` |

Native MTP 不写 `model`（草稿在 target 里）：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": <N>
  }'
```

Gemma 4 MTP、EAGLE-3、DFlash、DSpark 的 `model` 指到为该 target 训的 checkpoint：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "<method>",
    "model": "<matching-draft-checkpoint>",
    "num_speculative_tokens": <N>
  }'
```

Gemma 4 assistant checkpoint 即使从 `model` 进来，走的仍是 MTP 路。vLLM 把 assistant 接到 target，并让它共享 target KV。

打开之前先核对：装上的 vLLM 支持该方法与架构；draft checkpoint 对得上 target 和方法；`num_speculative_tokens` 和 checkpoint 兼容；model card 覆盖打算用的硬件和推理后端。

### 显存

Native MTP 不另载一份 draft，还可能和 target 共享 embedding 表或输出头。Gemma 4 MTP、EAGLE-3、DFlash、DSpark 要再载草稿权重——GPU 留余量。开销跟草稿大小、精度、张量并行布局、运行时 buffer 走。

## 现成的预训练草稿在哪

页上点名的 Hugging Face 发布方：

| 发布方 | 方法 | 代表模型和 target |
| --- | --- | --- |
| Google | Gemma 4 MTP | Gemma 4 E2B、E4B、12B、26B-A4B、31B 的 assistant [[7]](#ref-7) |
| LightSeek Foundation | EAGLE-3 和 EAGLE-3.1 | Kimi-K2.5 / K2.6 / K2.7-Coder 的 EAGLE 草稿，含标准与 MLA 变体 [[8]](#ref-8) |
| Red Hat AI | EAGLE-3、DFlash、DSpark | Llama、Qwen、Gemma、GPT-OSS、GLM、Nemotron、Mistral；后缀 `-speculator.eagle3`、`-speculator.dflash`、`-speculator.dspark` [[9]](#ref-9) |
| Z-Lab | DFlash | Qwen3 / Qwen3.5 / Qwen3.6、Gemma 4、Kimi、MiniMax、GPT-OSS、Llama；名字大致 `<target>-DFlash` [[10]](#ref-10) |
| DeepSeek AI | EAGLE-3、DFlash、DSpark | DeepSpec：Qwen3-4B / 8B / 14B 和 Gemma 4 12B 三条方法都有。例如 `eagle3_qwen3_8b_ttt7`、`dflash_qwen3_8b_block7`、`dspark_qwen3_8b_block7` [[11]](#ref-11) |
| Inferact | EAGLE-3 和 DSpark | `Inferact/MiniMax-M3-EAGLE3`（及 GQA 变体）、`Inferact/Kimi-K3-DSpark` [[12]](#ref-12) |

## 实验设置和测量

打开之后，真正的问题是：多出来的草稿活，端到端 serving 有没有变好。候选不必每位都对；target 提交前会验。性能取决于接受了多少，以及省下的 target Decode 能不能盖过草稿 + 验收的税。

他们用 **任务向** benchmark，不用随机 token 串。接受率跟真实输出的结构和可预测性绑在一起。

主要指标：

- 输出 token 吞吐，以及相对非投机基线的 speedup。
- 平均接受长度、草稿 token 接受率（有则记）。
- 相对非投机基线的模型质量。

### 模型和覆盖

勾 = 该 target–方法组合有结果；横杠 = 这轮没做。格子里是草稿发布方。

| Target | Native MTP | Gemma 4 MTP | EAGLE-3 | DFlash | DSpark |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-4-26B-A4B-it` | — | ✓ Google | ✓ Red Hat AI | ✓ Z-Lab | — |
| `google/gemma-4-31B-it` | — | ✓ Google | ✓ Red Hat AI | ✓ Z-Lab | ✓ Red Hat AI |
| `Qwen/Qwen3-8B` | — | — | ✓ Red Hat AI | ✓ Z-Lab | ✓ DeepSeek |
| `Qwen/Qwen3.5-27B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.5-122B-A10B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.6-27B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `Qwen/Qwen3.6-35B-A3B` | ✓ Built-in | — | — | ✓ Z-Lab | — |
| `moonshotai/Kimi-K2.5` | — | — | ✓ LightSeek | ✓ Z-Lab | — |
| `MiniMaxAI/MiniMax-M3-MXFP8` | — | — | ✓ Inferact | — | — |

每个数字都要放回它的测试配置里读：架构、激活参数量、草稿大小、负载、serving 条件都会推数字。

### 吞吐测量

生成 token / 秒，对照标准自回归基线；扫投机长度 `N`，看深度怎么推端到端 serving TPS。

原文 **Figure 4** 是可切换 target 的 Plotly 柱图（悬停看 speedup 和选中的 `N`）。这里不复刻交互件。下面只收正文里写成数字的结论。

### 主要观察

数字随 target、方法、负载、`N` 变。

**`gemma-4-26B-A4B-it`。** 这轮 sweep 里最大吞吐比：Gemma 4 MTP 在 GSM8K / MBPP 上 **2.74×**、**2.62×**；DFlash 在 MATH500 / HumanEval 上 **2.87×**、**2.79×**。EAGLE-3 四个数据集 **2.11×–2.27×**。

**`gemma-4-31B-it`。** Gemma 4 MTP：GSM8K **2.00×**，MBPP **1.99×**。DFlash：MATH500 **2.34×**，HumanEval **2.05×**。EAGLE-3 和 DSpark 在四个评测集上也高于基线（正文没写精确倍数）。最大吞吐对应的 `N` 随负载变。

**`Qwen3-8B`。** DSpark：**1.15×**（MATH500）到 **1.63×**（GSM8K）。DFlash：**1.08×–1.27×**。EAGLE-3 在 GSM8K、HumanEval、MBPP 高于基线；MATH500 上测到的最大值仍 **低于** 基线。

**`Qwen3.5-27B`、`Qwen3.5-122B-A10B`、`Qwen3.6-27B`。** 各自 sweep 里 native MTP 的最大值高于对应的 DFlash 最大值。这组最大比：**2.20×**，Qwen3.5-122B-A10B on MATH500。native MTP 取到最大吞吐时的 `N` 在 **4 到 7**，随模型和数据集。

**`Qwen3.6-35B-A3B`。** DFlash **1.77×–2.06×**，四个数据集的最大都在 **N=7**。Native MTP **1.28×–1.49×**，最大在 **N=6**。和 Qwen3.6-27B 同族、排名不同——一家模型之间也会换脸。

**`MiniMax-M3-MXFP8`。** EAGLE-3 在 HumanEval、**N=4** 上到 **2.09×**。（这只 target 跑在 MI355X；见 disclaimer。）

**`Kimi-K2.5`。** EAGLE-3 最高 **2.33×**；DFlash 最高 **2.68×**。EAGLE-3 的最大多半在 **N=4**；DFlash 的最大在 **N=7**。

跨实验，最大吞吐绑住的 `N` 不是常数。顺序方法：前几个 `N` 吞吐常往上，然后平台。DFlash / DSpark：**N=7** 经常落在较高吞吐的设置里；再大并不稳定地更快。

以上只对这套硬件、软件、target、draft checkpoint、负载和 sweep 负责。

## 拧参数

投机解码是运行时优化，不是一个 `N` 通吃。最好的 `num_speculative_tokens` 取决于接受了多少，以及省下的 target Decode 能不能盖过草稿 + 验收。

Model card 上的建议是起点；最终设置要用代表负载和端到端测量来选。有用的信号：吞吐、平均接受长度、总体接受率、**按位置** 接受率。

提议窗口更大，一次验收提交多枚的机会更多。后面几位的接受率却常常掉。多出来的候选白付草稿税 → TPS 走平甚至退。

### 从支持的配置起

Native MTP：**N=1** 最保守（额外顺序草稿最少）：

```json
{"method": "mtp", "num_speculative_tokens": 1}
```

正确性和稳定性确认之后，再扫 2、3、4、5、6、7。

这批测量里，native MTP 取到最大吞吐的 `N`：

- **Qwen3.5-27B：** GSM8K / MATH500 为 N=5；HumanEval / MBPP 为 N=4；MT-Bench 为 N=3。
- **Qwen3.5-122B-A10B：** 列出的四个推理 / 代码数据集都是 N=7。
- **Qwen3.6-27B：** N=4 或 N=5。
- **Qwen3.6-35B-A3B：** 吞吐一路涨到 N=6。

Gemma 4 MTP 和 EAGLE-3 的 `N` 变大同样加顺序草稿。就算 checkpoint 给了推荐，短 sweep 仍值得做。这批 Gemma 4 / EAGLE-3 里，测到的 TPS 多半在前几个 `N` 上升，然后平台。

DFlash：从 checkpoint 推荐或支持的提议长度起。许多 DFlash checkpoint 按固定 `block_size` 训。例如：

```text
block_size = 16
num_speculative_tokens = 15
```

第一位是确认过的 anchor，其余 15 位才是草稿候选。这是 **支持的最大** 提议长度，不一定是最高 TPS。原文建议试更小的：`N = 3, 7, 11, 15`。他们的 DFlash 实验里 **N=7** 经常落在较高吞吐；有的负载最大测到的 TPS 在 **N=11**。

DSpark：`num_speculative_tokens` 是每一投机轮生成多少候选。这批 vLLM 实验把 **整段** 配置好的提议都送去验收，所以 N=3 对 N=7（等等）要用端到端 TPS 比。

### 盯接受行为

| 信号 | 它在说什么 |
| --- | --- |
| Throughput | 端到端 serving 相对非投机基线 |
| Mean accepted length | 平均每一投机轮提交几枚草稿 |
| Overall acceptance rate | 提议草稿里接受了多大比例 |
| Per-position acceptance rate | 提议里更后面的位置还值不值得付 |

按位置接受是拧 `N` 的把手。前几位经常活、后面几乎不贡献，缩小 `num_speculative_tokens` 可能靠少付无用草稿而抬 TPS。

接受率要和吞吐一起读。草稿便宜时，接受率低也能打过基线。接受率高、草稿贵，TPS 不必涨。

### 让 sweep 跟负载匹配

GSM8K / MATH500：这轮里中等或更深的 `N` 常常对应更高测得 TPS。Qwen3.5-122B-A10B 的 native MTP 一路涨到 N=7。DFlash 的高点经常在 N=7 或 N=11。

HumanEval / MBPP：中等 `N` 常常落在较高吞吐。代码有局部结构，但格式、标识符、实现选择仍能让一段「看起来像」的续写岔开。

### 拧参流程（原文）

1. 从 checkpoint 支持 / 推荐的配置起。
2. 用代表 prompt 和生成设置做 benchmark。
3. 记下吞吐、平均接受长度、接受率。
4. 扫若干更小和更大的提议长度。
5. 按打算服务的负载选指标。这批实验的主指标是端到端 serving TPS。

赢家不必是最长提议、最高接受率、或最大平均接受长度。权衡草稿成本、验收成本、接受的 token，以及你真正在乎的指标。

## 给新 target 训一只 speculator

这篇不当训练教程。下面是 vLLM Speculators 和 DeepSpec 的工作流摘要 [[13]](#ref-13) [[14]](#ref-14) [[15]](#ref-15)。经 vLLM 导出 hidden：[extract-hidden-states](../architecture/extract-hidden-states.md)。

1. 准备代表 prompt。
2. 用 **恰好那只** target 生成回复。
3. 选 hidden 生成模式。
4. 收集该方法要的 target hidden。
5. 训 speculator。
6. 测接受和 serving 吞吐。

Prompt 应对上预期负载（chat、数学、代码、工具、多语）。留一份评测集。Tokenizer、chat template、thinking mode、生成配置应对上部署。把 target 的 tokenizer / chat template 套到 **已有** 回复上，并不会让数据变成「这只 target 的」；回复本身必须来自 target。

### Hidden 从哪来

| 训练模式 | 怎么做 | 主要代价 |
| --- | --- | --- |
| Online | 需要时由正在跑的 vLLM 服务现场生成 hidden，用完丢掉 | 不必堆磁盘缓存；推理和训练要同时占资源 |
| Offline | 训练前先生成并存下 hidden | 之后 GPU 可以全给训练；存储很重 |
| Hybrid | 第一 epoch 生成并缓存，之后复用 | 生成成本付一次，不必单独预处理阶段 |

模式只改 hidden 从哪来；后面的训练大体一样。

vLLM 服务可以跑 target，把该方法要的层 hidden 露出来。自定义层选择必须和 speculator 训练配置一致。

- EAGLE-3：选定层 hidden，自回归草稿 [[4]](#ref-4)。
- DFlash：用 target 特征训并行块预测 [[16]](#ref-16)。
- DSpark：在 DFlash 式网上加轻量顺序头和信心头 [[6]](#ref-6)。
- MTP：微调 target **自己的** MTP 组件——target 必须已经有兼容的 MTP 层 [[13]](#ref-13)。

训完检查 checkpoint，再和 target 一起在 vLLM 里 serve。训练 loss 不够：要看接受长度、接受率、草稿延迟、GPU 显存、端到端 serving TPS。某类负载接受弱 → 改 prompt 配比或训练配置再来。原则：同一只 target、同一种生成模式、同一类代表负载。

## Summary

草稿提议，target 验收；target 没点头就不提交。

五条路差在怎么用 target 信息，以及候选是顺序、并行、还是并行再加一层轻量顺序修正。

实验：Gemma、Qwen、MiniMax、Kimi 里选出的几只，Instinct **MI300X / MI355X**，ROCm。测到的 TPS 随 target、draft checkpoint、负载、`N`、serving 配置走。

有的设置变化很小，或 **低于** 非投机基线。若干模型–负载组合 **超过 2×**。页上写的上沿：**2.87×** DFlash on `gemma-4-26B-A4B-it`，**2.83×** Gemma 4 MTP 同一 target，**2.68×** DFlash on Kimi-K2.5。

`N` 也是变量。加大 `num_speculative_tokens` 有时在前几档有用，然后平台或回落。Checkpoint 推荐是起点；部署配置要用代表负载的测量和接受指标来选。

## Future work

- 非学习方法，例如 n-gram 投机和 suffix decoding，尤其适合重复 token 多的负载（改代码、agent 环）。
- 更宽的评测：并发、prompt / 输出长度、batch、采样设置。
- Speculator 训练数据怎么推代码、数学、chat、多语、工具、结构化输出上的接受。
- 更深地剖草稿生成、target 验收、KV-cache、图执行、调度。

## References

1. <a id="ref-1"></a> vLLM, Speculative Decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/
2. <a id="ref-2"></a> vLLM, MTP Speculative Decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
3. <a id="ref-3"></a> Google Developers Blog, Gemma 4 的 multi-token prediction — https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
4. <a id="ref-4"></a> EAGLE-3 论文 — https://arxiv.org/pdf/2503.01840
5. <a id="ref-5"></a> Z-Lab DFlash GitHub — https://github.com/z-lab/dflash
6. <a id="ref-6"></a> DSpark 论文 — https://arxiv.org/pdf/2607.05147
7. <a id="ref-7"></a> Google Gemma 4 collection — https://huggingface.co/collections/google/gemma-4
8. <a id="ref-8"></a> LightSeek Foundation models — https://huggingface.co/lightseekorg/models
9. <a id="ref-9"></a> Red Hat AI Speculator Models — https://huggingface.co/collections/RedHatAI/speculator-models
10. <a id="ref-10"></a> Z-Lab DFlash collection — https://huggingface.co/collections/z-lab/dflash
11. <a id="ref-11"></a> DeepSeek-AI DeepSpec — https://huggingface.co/collections/deepseek-ai/deepspec
12. <a id="ref-12"></a> Inferact models — https://huggingface.co/Inferact/models
13. <a id="ref-13"></a> vLLM Speculators，Training a Speculator — https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/
14. <a id="ref-14"></a> vLLM Speculators GitHub — https://github.com/vllm-project/speculators
15. <a id="ref-15"></a> DeepSeek-AI DeepSpec GitHub — https://github.com/deepseek-ai/DeepSpec
16. <a id="ref-16"></a> DFlash 论文 — https://arxiv.org/pdf/2602.06036

## Appendix：交互热力图（不抄 HTML）

原文附录是 **交互 HTML 热力图**：9 只 target × 方法 × 实验（按提议长度 `N` 的逐位接受率；每一行还有测到的 speedup 和输出 tok/s）。那是页上的 CSS/JS 控件，这里不倒。清洗稿里 **没有** 把逐位接受百分比印成静表——不要编。要悬停看格子，回原网页。

九只 target（覆盖表同上）：

1. `google/gemma-4-26B-A4B-it`
2. `google/gemma-4-31B-it`
3. `Qwen/Qwen3-8B`
4. `Qwen/Qwen3.5-27B`
5. `Qwen/Qwen3.5-122B-A10B`
6. `Qwen/Qwen3.6-27B`
7. `Qwen/Qwen3.6-35B-A3B`
8. `moonshotai/Kimi-K2.5`
9. `MiniMaxAI/MiniMax-M3-MXFP8`

页上对每一热力行的描述：按 `N` 的逐位接受，外加该次的 **speedup** 和 **输出 tok/s**。

**清洗稿 caption 里印出的基线 tok/s**（都是 `google/gemma-4-26B-A4B-it`；Gemma 4 MTP / EAGLE-3 / DFlash 的四条 caption 复用同一组基线）：

| Dataset | Baseline 输出 tok/s |
| --- | ---: |
| GSM8K | 2,344 |
| MATH500 | 2,181 |
| HumanEval | 1,854 |
| MBPP | 2,163 |

**正文写成数字的 speedup**（只收已写的；Summary 另点了同一 target 上 Gemma 4 MTP **2.83×** 作为上沿例子，「主要观察」里 GSM8K / MBPP 那对是 2.74× / 2.62×）。

| Target | 方法 | 写成的 speedup | 备注 |
| --- | --- | --- | --- |
| gemma-4-26B-A4B-it | Gemma 4 MTP | GSM8K 2.74×，MBPP 2.62×；Summary 2.83× | |
| gemma-4-26B-A4B-it | DFlash | MATH500 2.87×，HumanEval 2.79× | |
| gemma-4-26B-A4B-it | EAGLE-3 | 四个数据集 2.11×–2.27× | |
| gemma-4-31B-it | Gemma 4 MTP | GSM8K 2.00×，MBPP 1.99× | |
| gemma-4-31B-it | DFlash | MATH500 2.34×，HumanEval 2.05× | |
| gemma-4-31B-it | EAGLE-3、DSpark | 四个数据集高于基线 | 精确 × 未写 |
| Qwen3-8B | DSpark | MATH500 1.15× … GSM8K 1.63× | |
| Qwen3-8B | DFlash | 1.08×–1.27× | |
| Qwen3-8B | EAGLE-3 | GSM8K / HumanEval / MBPP 高于基线；MATH500 **低于** 基线 | 精确 × 未写 |
| Qwen3.5-27B / 122B-A10B / Qwen3.6-27B | Native MTP vs DFlash | native MTP 最大 > DFlash 最大；Qwen3.5-122B-A10B MATH500 **2.20×** | native MTP N=4–7 |
| Qwen3.6-35B-A3B | DFlash | 1.77×–2.06× at N=7 | |
| Qwen3.6-35B-A3B | Native MTP | 1.28×–1.49× at N=6 | |
| MiniMax-M3-MXFP8 | EAGLE-3 | HumanEval 2.09× at N=4 | MI355X |
| Kimi-K2.5 | EAGLE-3 | 最高 2.33×，多半 N=4 | |
| Kimi-K2.5 | DFlash | 最高 2.68×，N=7 | |

## Acknowledgements

Hongxia Yang、Peng Sun（AMD）；Pin Siang Tan、Jun Kang Chow、Ye Hur Cheong（Embedded LLM）。

## Disclaimer

测量在 AMD Instinct™ MI300X 和 MI355X 上，配置如下。

**Hardware**

- Hardware 1：**8×** AMD Instinct™ **MI300X**（gfx942），**2×** AMD EPYC™ **9654** 96-Core。
- Hardware 2：**8×** AMD Instinct™ **MI355X**（gfx950），**2×** AMD EPYC™ **9575F** 64-Core。用于 **MiniMax-M3-MXFP8** 实验。

**Software**

Ubuntu **22.04.5** LTS，ROCm/HIP runtime **7.2.53211**，vLLM **0.23.1rc1.dev1120+g0f0f28b53**，PyTorch **2.11.0+gitd0c8b1f**，Transformers **5.13.1**，Python **3.12.13**。

服务器厂商配置可能不同。性能随配置、软件、vLLM 版本、驱动和优化变化。
