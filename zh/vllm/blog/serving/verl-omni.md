---
source: https://vllm.ai/blog/2026-05-14-verl-omni
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# verl × Omni：扩散 RL 的 rollout 不另起炉灶

英文对照：[en/vllm/blog/serving/verl-omni.md](../../../../en/vllm/blog/serving/verl-omni.md)  
原文：https://vllm.ai/blog/2026-05-14-verl-omni  
2026-05-14。署名 **VeRL-Omni Team**。[VeRL-Omni](https://github.com/verl-project/verl-omni) 预发布，架在 [`verl`](https://github.com/verl-project/verl) + [`vllm-omni`](https://github.com/vllm-project/vllm-omni) 上。训练仍在 verl；扩散 / omni 的 rollout 走 Omni。v0.2 见 [verl-omni-v020.md](verl-omni-v020.md)。同一条 Omni 线：[vllm-omni.md](vllm-omni.md)。H800 / H200 菜谱数字是他们的合同，不是你的 SLA。

LoRA FlowGRPO、NVIDIA H800：4 卡 colocated **0.305** images/GPU/s、**420 s**/step；5 卡 async reward **0.280** images/GPU/s、**360 s**/step（墙钟约 **14%**）。非 CFG 全参 Qwen-Image OCR、4× H200：**0.510** images/GPU/s、约 **250 s**/step。文字渲染到第 **120** 步肉眼能看出来。NVIDIA GPU 和昇腾 NPU。

图仍在原页（本地没有拷贝）。图注：

**Architecture。** VeRL-Omni 架构总图。

**FlowGRPO。** 算法图：rollout → reward → policy update → weight sync。

**质量对照。** Prompt「Hidden Trail」/「Make A Wish」：step **0** vs **120**。

**曲线。** validation reward 约 **0.7 → 0.95**；rollout reward 均值约 **0.15 → 0.9**（非 CFG 起步低是预期）；zero-std ratio 只在 reward 饱和之后爬；actor `pg_clipfrac` 落在他们说的健康区间。

## 为什么要 VeRL-Omni？

LLM 的 RL 栈这一年跑得快；**多模态生成式 RL**（扩散和 omni，图/视频/音频的理解与生成）还卡在三件事：

- **扩散和 omni 的延伸。** 把 verl 的灵活性接到 DiT（Qwen-Image）、混合 AR-DiT（Qwen-Omni）、统一理解+生成（BAGEL、HunyuanImage3.0）。
- **异构 rollout。** rollout 是连续 latent 里的*去噪轨迹*，不是 token 序列。一次可能串 text encoder → DiT → VAE。
- **复杂调度。** 奖励自己就是多模态模型（VLM judge、OCR scorer）。生成 rollout 的内存峰值比文本高。

## 要点

- **多模态 rollout。** vLLM-Omni 异步 serving；他们说精度和 diffusers 相当。逐步 continuous batching、embedding cache 等。
- **灵活奖励。** 规则奖励和模型奖励（OCR 用 VLM-as-judge）。VLM/LLM 奖励推理走 vLLM。奖励和 rollout、训练重叠。
- **模块化 trainer。** DiffusersFSDP / Megatron / VeOmni；FSDP / USP / TP。
- **硬件。** NVIDIA GPU 和昇腾 NPU。
- **端到端菜谱和基准。** 吞吐数字在页上。

## 算法与模型

| Model | Architecture | Modality | Algorithm | Status |
|---|---|---|---|---|
| Qwen-Image | DiT | Text → Image | [FlowGRPO](https://arxiv.org/abs/2505.05470)、[MixGRPO](https://arxiv.org/abs/2507.21802)、[GRPO-Guard](https://arxiv.org/abs/2510.22319) | Released |
| BAGEL | Unified understand + gen | Text + Image | [FlowGRPO](https://arxiv.org/abs/2505.05470) | PR ready |
| Qwen3-Omni-Thinker | AR | Text / Image / Video / Audio | [GSPO](https://arxiv.org/abs/2507.18071) | PR ready |
| Wan2.2 | DiT | Text → Video | DanceGRPO | WIP |
| SD3.5 | DiT | Text → Image | DPO | WIP |
| HunyuanImage-3.0 | Unified understand + gen | Text + Image | MixGRPO、SRPO | Planned |

## 上手

安装：[Installation Doc](https://verl-omni.readthedocs.io/en/latest/start/install.html)。脚本：[examples](https://github.com/verl-project/verl-omni/tree/main/examples)。用 wandb 盯。

### Demo：Qwen-Image FlowGRPO 后训练

[flowgrpo 例子](https://github.com/verl-project/verl-omni/tree/main/examples/flowgrpo_trainer)：Qwen-Image，OCR 奖励。奖励模型 `Qwen3-VL-8B-Instruct` 读图上的字，对照数据集 ground truth。

#### 算法回顾

FlowGRPO：flow-matching 的在线 policy。多步 SDE 采样做探索；模型奖励打分。四段：

1. **Rollout generation** — logprob 轨迹和图像。
2. **Reward model scoring** — 轨迹 advantage。
3. **Policy optimization** — FlowGRPO CLIP 式 loss。
4. **Weight synchronization** — trainer 权重同步到 rollout worker。

#### LoRA（NVIDIA H800）

| Mode | # GPUs | Actor | Rollout | Async Reward | Throughput (images/GPU/s) | Time per Step (s) |
|---|---:|---:|---:|---|---:|---:|
| FlowGRPO colocated training | 4 | 4 | 4 | 0 (sync) | 0.305 | 420 |
| FlowGRPO w/ async reward | 5 | 4 | 4 | 1 (async) | 0.280 | 360 |

奖励模型独占一卡：墙钟每步约 **14%**，奖励和 policy 训练重叠。每 GPU 吞吐略低（0.280 vs 0.305），因为第五张卡是打分的，不是又一只 actor/rollout。

#### 全参微调

**非 CFG** 全参 Qwen-Image OCR，**4 × NVIDIA H200**：**0.510** images/GPU/s，约 **250 s**/step。文字渲染 **120** 步里「大幅增强」（页上 Hidden Trail / Make A Wish 对照）。

参考曲线：critic 和 validation reward 稳住收敛；rollout 均值起步低（非 CFG 的预期）。指标文档：[Training Metrics](https://verl-omni.readthedocs.io/en/latest/start/metrics.html)。

## 路线

预发布；扩散 RL 核心栈他们称稳定。点名下一步：

- 更多开源扩散 / omni（图/视频/音频；统一理解+生成）。
- 算法随论文进来（例如 DiffusionNFT）。
- actor、rollout、reward **全链路**异步——不止现在的 async-reward。
- 和 vLLM-Omni 合拧：并行、量化、batch、请求调度（rollout 占步时的大头）。
- DiffusersFSDPTrainer 之外，Megatron-core 和 VeOmni 上再出 trainer。
- 昇腾 NPU 再硬一点；硬件插件接更多后端。

## 社区

- **代码：** [github.com/verl-project/verl-omni](https://github.com/verl-project/verl-omni)
- **文档：** [verl-omni.readthedocs.io](https://verl-omni.readthedocs.io/en/latest/index.html)
- **贡献：** [`CONTRIBUTING.md`](https://github.com/verl-project/verl-omni/blob/main/CONTRIBUTING.md)
- **周会：** 每周二 **11:00AM（GMT+8）** — [meet.google.com/rho-aode-kmg](https://meet.google.com/rho-aode-kmg)
