---
source: https://vllm.ai/blog/2026-06-10-diffusion-gemma
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DiffusionGemma：把画布当成一次全拒的草稿

英文对照：[en/vllm/blog/architecture/diffusion-gemma.md](../../../../en/vllm/blog/architecture/diffusion-gemma.md)  
原文：https://vllm.ai/blog/2026-06-10-diffusion-gemma  
2026-06-10。署名 **The vLLM Team and Google DeepMind Team**。第一只进 vLLM 的 dLLM。数字是单卡 **batch=1**、`vllm bench serve` 在 H100 / H200 上的演示，不是 SLA。执行核见 [mrv2](mrv2.md)；草稿账本见 [spec-decode](../performance/spec-decode.md)；多模态流水线见 [vllm-omni](../serving/vllm-omni.md)。他们印的菜谱：[recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it)。

适用：把 DiffusionGemma（26B，Gemma4 骨干）当块扩散 dLLM serve——双向去噪、256 token 画布、ModelState 钩子、同一 batch 里混 prefill / denoise / commit。不适合：还当从左往右吐字；也不要把 **1288 tok/s** 当承诺。

Google 的 DiffusionGemma 是 **26B** 离散扩散语言模型，Gemma4 骨干——vLLM 原生伺候的第一只 dLLM。塞不进平常的自回归路径：双向注意力、迭代修、按块生成、每步去噪都要自己采样。

接到 [model runner v2](mrv2.md) 的 **ModelState**：模型自己管输入准备和每条请求的状态。宣称跟 Hugging Face 参考实现精度对齐，还能 batch。

自回归是一个 token 从左往右。dLLM 是对 **定长画布** 反复去噪，一步里修许多 token——用算力换带宽。**低 batch** 时多余 FLOPs 便宜、带宽才是瓶颈，这一换划算。一步吐许多 token，延迟可以压得很低。DiffusionGemma 一次去噪 **256** token 的画布。

![ar vs diffusion](../../../../assets/vllm/blog/architecture/diffusion-gemma/01-ar-vs-diffusion.svg)

**Figure。** 自回归 vs 块扩散解码（学习对照；版权仍归原站）。

## 架构与采样环

一份 Gemma4 骨干，两种模式，**同一份权重**：

- **Encoder：** *因果* 注意力，**写** KV。每块两次：prefill prompt；把写完的块「commit」。
- **Decoder：** *双向* 注意力，**只读** KV。去噪——画布上每个位置看每个位置，整块一起修。

Encoder 的因果注意力和 commit 后的 KV 跟自回归写法一样，**automatic prefix cache 不用改**。共享 prompt 前缀直接复用，扩散这边没有另做 cache。

一块 256 token：prompt 先 prefill（encoder）；画布随机初始化；状态切到 denoising。每步 decoder 跑完整画布，每个位置采一个候选，留下一部分。块不再变，状态切回 encoding；最后一次 encoder commit——写 KV、吐出 256 token——下一块再从随机画布起。

![sampling loop horizontal](../../../../assets/vllm/blog/architecture/diffusion-gemma/02-sampling-loop-horizontal.svg)

**Figure。** DiffusionGemma 按块的采样环（学习对照）。

块内 256 个位置并行去噪；**块与块之间仍从左到右**，因为新块要吃已经 commit 的 token。

### 熵预算去噪

每步把画布 **所有** 位置重新采样。只留下模型有把握的；其余丢掉，换成新的随机 token。把握 = 该位置预测分布的熵——熵低就是主意定了。

**熵预算：** 从最有把握排到最没把握，一路收，累加熵超过固定预算就停。早期几乎什么都不信，只钉几个锚；锚把邻居锐起来，更多位置掉进预算，几步就把块钉住。

![denoising grid](../../../../assets/vllm/blog/architecture/diffusion-gemma/03-denoising-grid.svg)

**Figure。** 几步里的熵预算去噪（学习对照）。

画布算 **收敛**：最好猜（**argmax**）连续几步不变，**并且** 每 token 平均熵低于阈值——或者撞上硬步数上限。Commit 的是这份 **干净 argmax**，不是步间带着走的噪声画布。

### Self-conditioning

下一步吃自己上一步的预测。不是硬 token：上一步完整 softmax，做成 token embedding 的概率加权平均，经一只小 **gated MLP** 加到画布 embedding 上。

![self conditioning](../../../../assets/vllm/blog/architecture/diffusion-gemma/04-self-conditioning.svg)

**Figure。** Self-conditioning 反馈路径（学习对照）。

被重新噪声化成随机 token 的位置，仍带着上一步的信息。Self-conditioning **只在 decoder / denoise**——encoder 的 prefill 和 commit 把反馈置零，那两趟只看见普通 token embedding。

## 在 vLLM 里怎么接

### 复用投机解码那条数据路

引擎里投机解码已经很熟。受 [RFC #36155](https://github.com/vllm-project/vllm/issues/36155) 启发，这条路拿来用。契合点：每步的画布 = 一大串 **draft，整块接受或整块拒绝**。调度器和 model runner 几乎不动。例外：投机解码总要多采一个（bonus token）；他们加了采 **0** 个 token 的支持，由 ModelState 管。

调度器、model runner、Gemma4 骨干原样复用；扩散专用的只有 ModelState 和 sampler：

![stack](../../../../assets/vllm/blog/architecture/diffusion-gemma/05-stack.svg)

**Figure。** DiffusionGemma 在 vLLM 软件抽象里的位置（学习对照）。

### ModelState 接口

没有 ModelState，非自回归模型上 V1 就得 fork runner，把扩散状态穿过输入准备、attention metadata、采样。ModelState 是 runner 每步会叫的钩子：

| Hook | DiffusionGemma 拿它做什么 |
| --- | --- |
| `prepare_inputs()` | 把画布 token embed 进去，加上 self-conditioning |
| `prepare_attn()` | 按请求设因果（encoder）还是双向（denoise）注意力 |
| `custom_sampler()` | 默认 sampler 换成 `DiffusionSampler` |
| `add_request()` / `remove_request()` | 建 / 拆每条请求的扩散状态（画布、self-conditioning 概率） |

模型用 `get_model_state_cls()` 自己注册。Runner 保持通用：`prepare_attn(...)` 建 metadata，`prepare_inputs(...)` 并进 forward kwargs，采样交给 `custom_sampler()` 装上的那只（这里是 `DiffusionSampler`）。

新的块扩散模型：实现一只 ModelState，模型类上一行注册。不必改 runner / scheduler / 公共基建。他们当后来 dLLM 的蓝图。

### `DiffusionGemmaModelState` 和 `DiffusionSampler`

`DiffusionGemmaModelState` 捏着每条请求扩散环上的状态：阶段旗标（commit vs denoise）、当前 `canvas`、给收敛检查用的历史、self-conditioning 概率，等等。活在 **预分配的 GPU 张量** 里，原地改。

`prepare_inputs()`：embed 画布 token，做 self-conditioning（上一步 denoise 的 softmax → 概率加权 embedding → gated MLP）。`prepare_attn()` 按阶段旗标建 attention metadata：因果（commit / encoder）或双向（denoise / decoder）。同一 batch 可以混 prefill、denoise、commit；每条的因果旗标在 GPU 上异步写——所以下面要改 attention kernel。

`DiffusionSampler` 顶替平常的 `(Sampler, RejectionSampler)`。阶段切换时初始化 / 重置画布和每条扩散状态。每步工作是一个 `@torch.compile` 的函数 `_compiled_sample_step`，对在飞的 decode 请求向量化，三种情形：

- **Prefill：** 画布随机初始化；返回 `num_sampled = 0`。
- **Denoise：** logits 除以温度；每个画布位置用 Gumbel-max 抽候选（`argmax(logits/T + gumbel_noise)`）；按熵预算收下最有把握的，其余重新噪声化。记下 argmax 画布，查收敛：argmax 稳定了指定步数且平均熵低于阈值，或撞步数帽。
- **Commit：** 吐干净的 `argmax_canvas`（`num_sampled = 256`）；为下一块重新初始化画布；重置该请求状态。

Denoise 时 sampler 报 `num_sampled = 0`、`num_rejected = query_len`，**KV cache 指针不动**；只有 commit 才往前走。把画布每个位置标成 rejected，调度器就把这条序列留在原地、下一步还排同一块——去噪环整段待在现成的投机解码账本里，**调度器一行都不用改**。

### 逐请求动态因果注意力

Encoder 因果，decoder 双向。此前因果性是 **整 batch 一个** 旗标。普通 decoder 只因果；Whisper 这类 encoder-decoder 只在 encoder 层双向。DiffusionGemma 按请求在 prefill、去噪、commit 之间切。为了延迟，vLLM **同一 batch 混不同阶段**。于是做了 **逐请求动态因果注意力**。

三请求的例子：

- **Request 0：** 长度 6 的 prefill——因果（「encoder」）。对角线以上 mask；每个 query 只看到自己及以前。Attention 按 tile 算（图上画 2×2；真 tile 更大，跟硬件调）。整块被 mask 的 tile 直接跳过（算力和从 HBM 拉 K/V）。
- **Request 1：** prefill 完，decoder / denoise。大小 4 的画布里，所有 query 看所有画布 key（双向），**也** 看全部 context key。没有 mask，不跳 tile。
- **Request 2：** 去噪完，画布可以收。最后一次 encoder：因果，把新接受的 token 填进 KV。Query 也看已经 cache 的 key。

![per seq causal attention](../../../../assets/vllm/blog/architecture/diffusion-gemma/06-per_seq_causal_attention.svg)

**Figure。** 逐请求动态因果注意力（学习对照）。

两套 backend：Triton Attention（`TRITON_ATTN`）和 FlashAttention 4（`FLASH_ATTN`）。原来的布尔 `causal` 换成 **逐请求因果性张量**。Mask 跟着改；tile 行为保留。

### Sliding window

DiffusionGemma 有些层走 sliding-window。画布上窗口必须 **对称**：窗口 `W`，画布 token 看自己、前面 `W` 个、**还有** 后面 `W` 个——总窗口 `2*W + 1`。

![per seq sliding window](../../../../assets/vllm/blog/architecture/diffusion-gemma/07-per_seq_sliding_window.svg)

**Figure。** 动态因果 sliding-window（学习对照）。

还是那三条请求，sliding-window 层 `W=2`。Request 0 和 2（prefill 与 accept）保持单侧因果窗口——query + 前面 `W` 个 key，对角一条带。Request 1 的去噪画布走对称窗口，所以只能看见落在窗口里的 context token。

两套 backend：双向请求只改窗口的 **右边界**。因果仍只向左；双向两边各 `W`。

## 量化 checkpoint

用 [LLM Compressor](https://github.com/vllm-project/llm-compressor) 量化，存成 [compressed-tensors](https://github.com/vllm-project/compressed-tensors)：

1. FP8——权重量化、激活全动态：[RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic)
2. NVFP4——权重和激活都 NVFP4：[RedHatAI/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4)

初步评测开 / 关 thinking，任务是 AIME 2025、GPQA Diamond、GSM8k，跑在 vLLM 上。Recovery 分数在 model card 里。

## 成绩

瞄准交互、低延迟。Bench：`vllm bench serve`，**batch size 1**，一张 H100、一张 H200。FP8 扩散：

- **H200：1,288 generation tok/s**——相对普通 AR 约 **6×**，相对 multi-token prediction（MTP）约 **3×**
- **H100：1,008 tok/s**——相对 AR 约 **5×**，相对 MTP 约 **2.6×**

![perf](../../../../assets/vllm/blog/architecture/diffusion-gemma/08-perf.svg)

**Figure。** H100 / H200 上的生成吞吐——FP8 扩散 vs 自回归基线（学习对照）。他们给的复现：[gist](https://gist.github.com/LucasWilkinson/89185e4dc05d300df33a4ce030973911)。

## 致谢

Google DeepMind × vLLM 近身合作。

- **Google DeepMind：** Martin Kukla、João Gante、Luciano Martins
- **vLLM：** Lucas Wilkinson、Matthew Bonanni、Nicolò Lucchesi、Dipika Sikka、Doug Smith、Edward Arthur Quarm Jnr、Alon Kellner（Red Hat）、Nick Hill（Inferact）
- **NVIDIA：** Dimitrios Bariamis、Alec Kohlhoff、Porras Huang、Eugene Rakhmatulin
