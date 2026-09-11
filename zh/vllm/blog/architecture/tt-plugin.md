---
source: https://vllm.ai/blog/2026-09-07-vllm-tt-plugin
lang: zh
voice: book-zh
fetched: 2026-09-11
---

# Tenstorrent 上的 vLLM：仓外 Platform 插件怎么接一张 mesh

英文对照：[en/vllm/blog/architecture/tt-plugin.md](../../../../en/vllm/blog/architecture/tt-plugin.md)  
原文：https://vllm.ai/blog/2026-09-07-vllm-tt-plugin  
2026-09-07。署名 **Tenstorrent Team**。学习译文，不是官方译本。仓外平台插件：[tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin)。硬件：[Tenstorrent](https://tenstorrent.com/)。运行时：[TT-Metal](https://github.com/tenstorrent/tt-metal)。同一扇门：[hardware-plugin.md](hardware-plugin.md)、[plugin-system.md](plugin-system.md)。页上不报 tokens/$；当时的数字在 tenstorrent.com 和 tt-metal。站点 logo 不收。

和 vLLM 一起装。只要 TT-Metal 的 `ttnn` 能 import，Tenstorrent 就会被发现并注册成一块 vLLM platform。对外 serving 表面不变：OpenAI 兼容 API、同一套请求格式、同一套客户端。

有意思的不是「又多了一家后端」。Tenstorrent 设备看起来不像 GPU，而 vLLM 的插件接口宽到能把这三件事完全写在仓外：受相位约束的调度器、另一套 data-parallel 拓扑、一部分采样发生在设备上。

## 支持哪些模型

插件按 `TT` 前缀注册架构。权重按它声明的 architecture 被捡起来，不按仓库名：

| 模型族 | Architectures |
|---|---|
| Llama 3.1 / 3.2 / 3.3 | `TTLlamaForCausalLM` |
| Llama 3.2 Vision | `TTMllamaForConditionalGeneration` |
| Qwen 2.5 / Qwen 3 | `TTQwen2ForCausalLM`, `TTQwen3ForCausalLM` |
| Qwen 3.5 / Qwen 3.6 | `TTQwen3_5ForConditionalGeneration` |
| Qwen 2.5-VL / Qwen 3-VL | `TTQwen2_5_VLForConditionalGeneration`, `TTQwen3VLForConditionalGeneration` |
| Mistral / Mistral 3 | `TTMistralForCausalLM`, `TTMistral3ForConditionalGeneration` |
| Gemma 3 | `TTGemma3ForConditionalGeneration` |
| Gemma 4 | `TTGemma4ForCausalLM`, `TTGemma4ForConditionalGeneration`, `TTGemma4UnifiedForConditionalGeneration` |
| DeepSeek V3 | `TTDeepseekV3ForCausalLM` |
| GPT-OSS 20B / 120B | `TTGptOssForCausalLM` |

这些类住在 TT-Metal 里，和运行时放在一起：每一个都是面向 vLLM 的 generator，底下包着手写的 TTNN 实现。插件**不带**模型代码——它只注册名字，解析到什么由 tt-metal 提供。

匹配看的是 architecture，一条入口可以盖住几个发行版。例如 `TTQwen3_5ForConditionalGeneration` 用来 serve `Qwen/Qwen3.6-27B`。

多模态覆盖（新后端常常先只做文本）：Llama 3.2 Vision、Qwen-VL、Qwen 3.6、Mistral 3、Gemma 3 写这篇时都能走插件。

模型不必编进插件。把 `EXTRA_MODELS_DIR` 指到一串 bundle 目录，每个目录里有 `vllm_metadata.json` 和一份 adapter 类，启动时就会按 `TT` 约定注册。发行工具可以不改源码就交出一份能 serve 的模型。`TT_VLLM_BUILTIN_MODELS=0` 把注册表收成「只认你提供的那些」。

## 为什么 Tenstorrent 后端长得不一样

Tenstorrent 系统是一张 **core 与芯片用片上网络连起来的 mesh**。一张卡（n150 或 n300）已经是一小张 mesh；[QuietBox](https://tenstorrent.com/hardware/tt-quietbox) 更大；[Galaxy](https://tenstorrent.com/hardware/galaxy) 是 32 颗 Wormhole，运行时直接配拓扑（`FABRIC_1D`、`FABRIC_2D`、`FABRIC_1D_RING`）。程序按 mesh 形状编译、trace。芯片之间搬数据是编译进程序的一部分，不是 host 再发一次 collective。

这套插件里的模型是给 TT mesh **手写的 [TTNN](https://github.com/tenstorrent/tt-metal) 实现**，从两芯 n300 到 32 芯 Galaxy。系统内部仍用跨芯张量并行、跨子 mesh 的数据并行，但写在 TTNN 里、编进 mesh 程序，而不是配成运行时 rank。手调换来更好的 tokens/$；这篇不报数字。

![mesh vs collectives](../../../../assets/vllm/blog/architecture/tt-plugin/01-mesh-vs-collectives.svg)

**图注（原文 Figure 1）。** 跨芯并行发生在哪。GPU 形状的栈上，host 每一层发 collective，并行是运行时选择（TP / PP rank）。Tenstorrent 上整张 mesh 编成一份程序再 trace，fabric 在程序里面搬数据，host 每步只提交、读回一次。

「整张 mesh 一份 trace 程序」几乎决定了后面所有设计：

- **没有可配的 TP / PP rank。** Galaxy 上的 70B 不是「TP=32 个进程」，而是一份为 32 芯 mesh 编译的程序。`MESH_DEVICE=TG` 取代 `--tensor-parallel-size`。插件直接拒绝 `-tp`/`-pp`，不去假装遵守。适合（模型, mesh）的并行写在模型代码里。
- **工作单元是一整步被 trace 的执行。** 设备上主要是按固定 batch 形状回放捕获好的 trace。形状稳定、同质的 batch 比混在一起的 batch 便宜得多。
- **采样可以发生在设备上。** mesh 程序可以把采样做到选 token 为止，token 常常已经选好再回来，host 看不到 logits。

这三件事都和 GPU 形状的推理栈拧着。后面几节就是他们怎么解开的。

## 插进去，不 fork

vLLM 的硬件插件机制 [2025 年 5 月那篇](https://vllm.ai/blog/2025-05-12-hardware-plugin) 引进来，早期用户包括 `vllm-ascend` 和 `vllm-spyre`。Spyre 那套可插拔调度器，才让这条路走得通。他们很依赖它。

两个 entry point：

| Entry point 组 | 名字 | 目标 |
|---|---|---|
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |

`platform_plugin()` **只有 `ttnn` 能 import 时**才返回 TTPlatform，所以把包装进普通 CUDA 环境，不会误选 Tenstorrent。

之后只有一次交接。TTPlatform.check_and_update_config() 校验配置、注册模型架构，再通过现成的扩展点换上 Tenstorrent 自己的运行时类：

| vLLM 配置字段 | TT 实现 |
|---|---|
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `scheduler_config.scheduler_cls` | `vllm_tt_plugin.scheduler.TTScheduler` 或 `vllm_tt_plugin.lane_scheduler.TTLaneCoordinator` |

设备相关选项走 vLLM 通用的 additional-config 命名空间，不加新 CLI 旗标：

```bash
--additional-config.tt.sample_on_device_mode all
--additional-config.tt.fabric_config FABRIC_1D_RING
```

**vLLM 主干里没有任何 Tenstorrent 专用代码。** 支持跟着 vLLM 的发布节奏走，而不是卡在三个月前的 fork 上。写这篇时他们对着一份钉死的 vLLM 发行版验收，插件 API 稳定之后再把窗口放宽。

## 按相位调度：一步只做 Prefill，或一步只做 Decode

上游 V1 调度按 token 预算走。一条请求有 already-computed tokens 和 target tokens；每一步在预算里再发一点 token 工作。Prefill 和 Decode 不是两种模式，chunked prefill 和进度不一的混合 batch 才自然掉出来。

Tenstorrent 这条路径更紧。每一步调度只落到三种结果之一：

- **只做 Prefill**
- **只做 Decode**
- **空**

没有 Prefill+Decode 混在同一步的 batch。这个约束里仍然支持 chunked prefill：超过单步 token 预算的 prompt 拆成多步 Prefill，Decode-only 步插在这些 chunk 之间，长 Prefill 还在飞的时候，已经在飞的请求继续往前走。默认先收 Prefill，后面的 Decode 步就能拿到更大、更有效的 batch。若收不进 Prefill、但还有 Decode 请求在跑，这一步就是 Decode-only，进度继续，KV 压力也可以松一点。

![scheduling phases](../../../../assets/vllm/blog/architecture/tt-plugin/02-scheduling-phases.svg)

**图注（原文 Figure 2）。** 同一条长 prompt 在两套调度下。上游把它拆成四步 chunk，并把别的请求的 Decode 混进这些步。Tenstorrent 上一步仍是全 Prefill 或全 Decode：prompt 按 Prefill-only chunk 跑，中间插入 Decode-only 步，每一步形状稳定、能回放 trace，飞行中的请求也继续前进。

**换来什么。** Trace 执行奖励 batch 形状稳定：一步全是 Prefill 或全是 Decode，就能回放为那种形状捕获的 trace。混在一起的一步需要一份从未捕获过的形状。相位拆分不是 Tenstorrent 的怪癖：最大的 GPU 部署在实例粒度上做同一件事——[分离 serving](https://docs.vllm.ai/en/stable/features/disagg_prefill/)。Tenstorrent 调度器把同样的拆分放到**一步**里，发生在同一台引擎内部。

**它不牺牲什么。** 广义的 continuous batching 还在。请求进 `waiting`，结构化输出的语法还在编译时可以停在 `skipped_waiting`，别的请求还在跑时可以准入，可以被抢占回去，也可以各自完成。限制发生在**设备一步之内**，不是跨过请求生命周期。

**它确实付出什么。** 交错粒度是整整一步。上游把一段 Prefill chunk 和正在进行的 Decode 放进同一步；这里调度器轮流做，Decode 请求要在自己的步之间等完每一段 Prefill chunk，每次切模式还要排空下面那条 async decode overlap 流水线。这两笔都是调度策略成本，不是硬件上限：以后版本仍然可以捕获混合形状的一步。

## Galaxy 上单进程的 lane 数据并行

vLLM 别处没有对应物。页上特意要反馈。

一部分 Tenstorrent 模型——Llama 3.3 70B（`TT_LLAMA_TEXT_VER=llama3_70b_galaxy`）、Qwen3-32B（`TT_QWEN3_TEXT_VER=qwen3_32b_galaxy`）、以及 GPT-OSS——由 *single-execute* generator 来 serve：一份程序罩住整张 Galaxy mesh，每步执行一次。没有子 mesh 可以再给第二个引擎进程。标准的多进程数据并行给每个 rank 分设备，这里没有东西可切。

这些模型是单份权重、单次执行，却保留 **四份独立的 data-parallel KV cache**，各在自己的 DP 子 mesh 上。进程级没有东西可切，调度级有四件要独立排。

第一次尝试仍按 vLLM 惯例给每个 DP rank 一个进程。Rank 必须协商这一步是 Prefill 还是 Decode，而 mesh 的提交/读回其实只有一次，他们就会改到远远超出硬件插件表面的主干。各 rank 的调度器确实并行跑了，但每一步多出来的进程间 scatter/gather，比那点并行赢回来的还贵。

交出去的答案是把并行放进**一个**引擎进程：

TTLaneCoordinator 为每条 **lane** 拥有一份独立的 TTScheduler。每条 lane 有自己的 `waiting` / `running` 队列、自己的准入、自己的 KV cache 管理器、自己的 lane 本地 block ID 空间。新请求分到负载最轻的 lane，并绑在那里。

设备一次执行所有 lane，协调器每步必须选一个共享模式：

- 任何一条 lane 能收 Prefill，**所有** lane 都跑 Prefill 步，受和单调度器相同的 decode-interleave 节奏约束
- 否则所有 lane 跑 Decode 步
- 对选中的模式没有工作的 lane，在合并后的 batch 里贡献一块空切片

然后协调器合并各 lane 的 SchedulerOutput，worker 组一份合并后的设备输入，runner 再按 lane 拆回结果——全程一个进程，**没有任何进程级 collective**。多进程方案输掉的，就是那笔 scatter/gather。

![lane DP](../../../../assets/vllm/blog/architecture/tt-plugin/03-lane-dp.svg)

**图注（原文 Figure 3）。** 同一套四份 data-parallel KV cache，两种排法。上（放弃）：四个引擎进程每步用进程间 scatter/gather 协商共享的 Prefill 或 Decode 模式，尽管 mesh 提交和读回只有一次。下（交出）：一个引擎进程，协调器选定共享模式，四个独立调度器带着 lane 本地 block ID，一份合并后的设备输入，结果再按 lane 拆开。

有一处重试花了他们一段时间才做对。若被迫的 Prefill 步一个 token 都收不进（通常是 KV 压力），而某条 lane 还有正在跑的 Decode，这一步就改成 Decode 再试。没有这次重试，KV 压力会把协调器送进无进展环：因为有 lane *想* 准入所以选了 Prefill，块不够所以一个都收不进，本该腾出块的 Decode 却永远轮不到。

面向用户的表面故意很平常：

```bash
MESH_DEVICE=TG \
TT_LLAMA_TEXT_VER=llama3_70b_galaxy \
VLLM_RPC_TIMEOUT=900000 \
python examples/server_example_tt.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --data_parallel_size 4 \
  --max_num_seqs 8 \
  --async-scheduling \
  --additional-config.tt.dispatch_core_axis col \
  --additional-config.tt.sample_on_device_mode all \
  --additional-config.tt.fabric_config FABRIC_1D_RING \
  --additional-config.tt.worker_l1_size 1344544 \
  --additional-config.tt.trace_region_size 220000000
```

`--data_parallel_size 4 --max_num_seqs 8` 变成四条进程内 lane，每条八个请求：并发 32。`--max_num_seqs` 是**每条 lane** 的容量。用户写已经认识的旗标，后端再映射到 single-execute Galaxy 模型的进程内 lane，或其它模型的普通多进程 DP（启动时发现子 mesh，用 `TT_VISIBLE_DEVICES` 分配）。启动日志会写明选了哪一种。

## 设备上采样，回退不用人配

打开 `sample_on_device_mode` 时，mesh 程序把采样做到选出 token，回来的是 token 而不是 logits。

很多请求走不了这条路：logprobs、penalty、allowed-token mask、bad-word 过滤、自定义 logits processor。插件不拒这些请求，也不让用户自己选模式。**它按 batch 决定**，只要这批需要设备路径表达不了的东西，就退回 vLLM 自己的 LogitProcessor 和 sampler，能走设备时再回去。需要 host 侧采样的请求付一次读回，结果仍然正确；其余请求留在快路径。`always_compat_sampling` 强制走 host，用来调试或 A/B。

## Decode overlap 是异步读回，不是异步执行模型

插件支持 decode/host overlap，闸门是模型自己声明的 `supports_async_decode`。模型没声明，platform 就关掉 async scheduling，不让用户打开一份没验收过的东西。

这里的「async」比通常说的窄，老实讲是 **host 侧异步读回**，不是设备上另有一条执行线程：

1. 用 `read_from_device=False` 提交 Decode（不阻塞）。
2. 用 `read_decode_output(..., async_read=True)` 开始 host 读回，把返回的 event 和这次提交记在一起（也不阻塞）。
3. 到最终化时，用 `ttnn.event_synchronize(...)` 等这些 event。
4. 这之后才把设备输出转成 host tensor 和采样结果。

![async decode](../../../../assets/vllm/blog/architecture/tt-plugin/04-async-decode.svg)

**图注（原文 Figure 4）。** overlap 从哪来。没有它，设备要等 host 把上一步读回并采样。有了 async decode，读回留在飞行中，host 可以调度下一步、最终化上一步，同时设备还在忙。唯一会堵住的等待是最终化时的 `ttnn.event_synchronize()`。

引擎保持深度为 2 的 in-flight 队列，填满之前不阻塞，所以 host 可以在第 *N* 步读回还在飞时调度第 *N+1* 步。只有 batch *稳定* 时才保持 overlap——形状稳定、设备上采样、没有结构化输出记账、没有恢复中的 Prefill。任何一条破了，先排空再往下走。

Prefill 实际上仍是同步的。Decode overlap 是稳态生成的快路径，不是一条万能的异步流水线。完整说明在插件仓 [`docs/SCHEDULING.md`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/docs/SCHEDULING.md)，包括 executor 输出线程和引擎线程抢同一份结果时，最终化记账怎么保证正确。

## 当前限制

TTPlatform 在配置时拒绝或改掉不支持的组合，错误在到达设备之前就说清楚：

- **张量并行和流水线并行支持，但方式不同。** 并行来自 mesh 形状（`MESH_DEVICE`）和模型实现，不是 vLLM 的 TP/PP rank。
- **投机解码当时还不支持。**
- **LoRA 当时还不支持。**
- **Prompt logprobs 当时还不支持**，在请求校验时拒绝。
- **Prefix caching** 只对声明支持的模型打开。
- **Async decode overlap** 只对声明能力的模型打开。
- **标准多进程 DP 不支持 MoE。** 需要内部数据并行的 single-execute 模型（例如 GPT-OSS）折进 lane-DP。
- **多机 serving 当时还不支持。** Tenstorrent 硬件能扩过一台机器，但当时的 TT 多机模型实现不能直接映射到 vLLM 的多机范式。

这些是当时 Tenstorrent 运行时和模型实现的性质，不是硬件、软件栈或 vLLM 插件 API 的硬上限。

## 怎么试

先装 [TT-Metal](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) 并激活那个环境，再克隆插件，从仓库根目录跑安装脚本：

```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin.git
cd vllm-tt-plugin
source docs/install-vllm-tt.sh
```

脚本用 `VLLM_TARGET_DEVICE=empty` 编 vLLM——`tt` platform 由插件在运行时提供——再装插件。然后 serve 和查询：

```bash
MESH_DEVICE=T3K VLLM_RPC_TIMEOUT=100000 python examples/server_example_tt.py
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32}'
```

现成的 OpenAI 客户端代码不用改。

当时的安装会在 tt-metal 环境里从源码编一份 **0.26.0** 的 vLLM。按模型的命令、mesh 形状、必要的环境变量见 [插件 README](https://github.com/tenstorrent/vllm-tt-plugin) 和对应的 tt-metal 模型 demo。

## 下一步（页上）

- 更宽的 async decode 覆盖——更多模型族声明 `supports_async_decode`，更少条件会迫使排空（尤其是设备上采样的几种模式）。
- 更多模型上的 prefix caching，以及 lane-DP 对请求级 RoPE 的支持，好让视觉模型也能用。
- 投机解码，等 mesh 侧 draft/verify 故事定下来。
- 多机 serving——比单机装得下的模型更大。

## 致谢

这项工作站在昇腾团队贡献的 platform 插件机制、以及 Spyre 团队的可插拔调度器设计上——没有后者，相位调度器就意味着 fork。感谢 vLLM 维护者把 V1 扩展点留得足够通用，一张 mesh 架构能穿过去。

页上点名：Viktor Puš、Tomasz Cheda、Sanjar Adylov、Salar Hosseini。

他们特别想听两件事：把 `--data_parallel_size` 折进进程内 lane，对 single-execute 模型是不是对的用户表面；以及下一步优先哪些模型族。Issue 和 PR 去 [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin)，也可以在 vLLM Slack 找到他们。
