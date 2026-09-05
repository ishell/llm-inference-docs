---
source: https://vllm.ai/blog/2026-08-20-verl-omni-v0-2-0
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# verl-Omni v0.2：请求级 batch 把 gen 从 226s 压到 108s

英文对照：[en/vllm/blog/serving/verl-omni-v020.md](../../../../en/vllm/blog/serving/verl-omni-v020.md)  
原文：https://vllm.ai/blog/2026-08-20-verl-omni-v0-2-0  
2026-08-20。署名 **VeRL-Omni Team**。接五月那篇 [verl-omni.md](verl-omni.md)。仓库：[verl-project/verl-omni](https://github.com/verl-project/verl-omni)。两句标题：扩散 RL 更快（Qwen-Image FlowGRPO 走 vLLM-Omni + verl V1 trainer）；omni 训练更稳（omni V1 trainer、可复用 adapter、FSDP2、vLLM-Omni rollout）。wandb 菜谱数字是他们的合同，不是你的 SLA。

v0.1 的 rollout 几乎是串行 `B≈1` 的 DiT forward（10 步去噪，True-CFG 每步再翻一倍）；GPU 占用约 **80%**。v0.2 请求级 packing：占用约 **100%**，孤立生成 **226 s → 108 s**（**52%**）。MMK12（Qwen3-Omni Thinker × GSPO，4× H800 80GB）：val reward **0.833**，actor-rollout Pearson **0.998**，约 **59 GB**。步时仍引用 v0.1 LoRA 表：4× H800 约 **420 s**；5 卡 async reward 约 **360 s**。

封面总览图仍在原页（本地没有拷贝）：**VeRL-Omni v0.2.0 release overview**。

本地图（原文版权仍归原站；学习对照用）。曲线：**蓝 = v0.1**，**绿 = v0.2**。

![qwen image gpu utilization](../../../../assets/vllm/blog/serving/verl-omni-v020/02-qwen-image-gpu-utilization.svg)

![qwen image timing gen](../../../../assets/vllm/blog/serving/verl-omni-v020/03-qwen-image-timing-gen.svg)

![qwen image timing step](../../../../assets/vllm/blog/serving/verl-omni-v020/04-qwen-image-timing-step.svg)

![omni ppo adapter flow](../../../../assets/vllm/blog/serving/verl-omni-v020/05-omni-ppo-adapter-flow.svg)

![mmk12 training rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/06-mmk12_training_rewards.svg)

![mmk12 val rewards](../../../../assets/vllm/blog/serving/verl-omni-v020/07-mmk12_val_rewards.svg)

## 1. 更快的扩散 RL

贵法跟自回归 LLM RL 不一样。一次 rollout：许多去噪步、大 latent、prompt embedding、可选 CFG、奖励打分、old-log-prob 重算、policy 权重同步。Qwen-Image FlowGRPO **没有单一元凶**——步时是这些加在一起。

### 要点

- **请求级 batching** 对支持的扩散 adapter 成了默认 vLLM-Omni rollout。兼容请求打进更大的 transformer forward；并发旋钮写明。指南：[rollout batching](https://verl-omni.readthedocs.io/en/latest/start/rollout_batching.html)。runtime：[diffusion continuous batching](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching)。
- 扩散也有 **V1 trainer**——靠近别处那套现代 trainer；给 rollout 和训练解开做铺垫。

点名的正确性修复：请求级 batch 的扩散 logprob、async rollout 语义、rank-local LoRA 权重更新、可选 rollout-correction 的 hook。rollout 再快，轨迹和 logprob 也得还在描述**同一份** policy。

### 新支持

| Model × Algorithm | Acceleration / support | Script | W&B |
|---|---|---|---|
| Qwen-Image × FlowGRPO LoRA | **request-level batching** | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh) | [run](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd) |
| Qwen-Image × FlowGRPO full model | step-wise continuous batching | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh) | [run](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb) |
| SD3.5 Medium × FlowGRPO LoRA，**V1 trainer** | **request-level batching**，sync | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh) | [run](https://wandb.ai/mikecheung/flow_grpo/runs/h04p15jr) |
| SD3.5 Medium × FlowGRPO LoRA，**V1 trainer** | **request-level batching**，`separate_async` | [script](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh) | [run](https://api.wandb.ai/links/didan/kk5uxbmh) |

全表：[README.md](https://github.com/verl-project/verl-omni#model-and-algorithm-support-)。

### 菜谱和基准

展品是 Qwen-Image LoRA OCR。v0.1：串行 `B≈1`，10 步去噪，True-CFG 每步两次 forward；GPU 占用约 **80%**。v0.2 把完整请求打进一次 transformer forward；占用约 **100%**；孤立生成 **226 s → 108 s**（**52%**）。每张图的生成延迟跟着掉。对照：[v0.1](https://wandb.ai/mikecheung/flow_grpo/runs/o7x44yrr)、[v0.2](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd)。

**Figure（GPU 占用）。** 蓝 v0.1 → 绿 v0.2。

**Figure（生成时间）。** v0.2 路径上孤立 gen 下降。

**Figure（步时）。** 同一趋势。

生产风 Qwen-Image FlowGRPO LoRA 默认打开请求级 batch。入口：`run_qwen_image_ocr_lora.sh`。关掉逐步执行，让 Omni 调度到 `max_num_seqs`：

```bash
actor_rollout_ref.rollout.step_execution=false
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=32
```

Qwen-Image LoRA + True-CFG、512 px：实用区间 `max_num_seqs=8` 到 `32`；再大撞 HBM。SD3.5 更轻：`max_num_seqs=256`。

菜谱级步时（和 [verl-omni.md](verl-omni.md) 同一句话）：基线 LoRA 4× H800 约 **420 s**/step；async reward 5 卡约 **360 s**/step。

## 2. 更稳的 omni 训练

Omni 不是更大的语言模型；它是小组件：processor、模态塔、可训阶段、必须和 actor 对齐的 rollout。v0.2 从一次性接线走向可复用栈。

### 要点

- omni 走 **verl V1 trainer**：worker 编排、标准配置覆盖、和 vLLM-Omni rollout 对齐。
- **可复用 omni adapter：** 模型、processor、可训阶段、FSDP 准备、rollout 对齐，共用接口。

**Figure（adapter 调用流）。** `main_omni.py` 只决定在线 omni 任务进 verl PPO V1。PPO trainer 管通用 RL 环（rollout、advantage、policy update）。FSDP omni engine 加载 Hugging Face 模型，向 `OmniModelBase` 要 adapter。Qwen3-Omni thinker 训练时，`Qwen3OmniThinkerAdapter` 拆掉闲置模块（Talker、codec），把 `forward` 指到 thinker，备好 processor 和 rollout hook，再把控制权还给 PPO。

Thinker-only：FSDP / FSDP2 wrapping。

### 新支持

| Model × Algorithm | Modality / dataset | Support | Script | W&B |
|---|---|---|---|---|
| Qwen3-Omni Thinker × GSPO | text → text / GSM8K | **V1 trainer**、可复用 adapter、FSDP2、vLLM-Omni rollout | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh) | [run](https://wandb.ai/mikecheung/gspo/runs/j5mro1tn) |
| Qwen3-Omni Thinker × GSPO | image → text / MMK12 | **V1 trainer**、多模态数据、actor-rollout 一致性 | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh) | [run](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36) |
| Qwen3-Omni Thinker × GSPO | text + image + audio → text / AVQA-R1-6K | **V1 trainer**、NPU 菜谱、多模态输入 | [script](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh) | — |
| Qwen3-Omni Thinker × DPO | multimodal → preference / Omni-Preference | `OmniDPOLoss`、按模态分组的 batch | [script](https://github.com/verl-project/verl-omni/blob/main/examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh) | [report](https://api.wandb.ai/links/didan/iumxl2zr) |

### 菜谱和基准：MMK12

锚：`run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh`。K12 视觉数学（`image → text`），GSPO，LoRA rank **32**，actor-rollout colocated，**4 × H800 80GB**。rollout 形状：**128** prompt × **16** response = **2048** 样本。训完：val reward **0.833**，actor-rollout Pearson **0.998**，约 **59 GB**。[wandb](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36)。

**Figure（MMK12 训练奖励）。** 训练均值。

**Figure（MMK12 验证奖励）。** 验证均值。

数据管线：原始 MMK12 parquet → verl RL parquet。图像字节内嵌；prompt 要求结构化答案。奖励：`math_verify` 精度 + `<answer>...\boxed{}...</answer>` 上的渐进格式分。

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12

TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

这才是稳定性故事：不是一次性启动路径——V1 trainer、可复用 adapter、多模态数据、一致性指标、写下来的 image-to-text 基准。Thinker 侧 serving 见 [qwen3-omni.md](qwen3-omni.md)。

## 模型与算法扩展

| Model / family | Category | Modality | Algorithm / recipe | Update |
|---|---|---|---|---|
| [LTX2.3](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/ltx2/README.md) | Diffusion generator | Text → Video + Audio | FlowGRPO | T2V+音频；CLAP、ImageBind 奖励 |
| [Qwen-Image-Edit](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) | Diffusion image editor | Text + Image → Image | FlowGRPO | 编辑训练接口 + 数据准备 |
| [BAGEL](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/bagel/README.md) | Unified understand + gen | Text + Image | FlowGRPO | 全参和 LoRA；OCR、PickScore |
| [SD3.5 + DiNa-LRM](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer_sd35_drm.html) | Diffusion generator | Text → Image | FlowGRPO + latent reward | 直接打干净 latent；奖励时跳过 VAE decode |
| [Flow-DPPO](https://verl-omni.readthedocs.io/en/latest/algo/flowdppo.html) | Diffusion algorithm | Text/Image → Image | Flow-DPPO | Qwen-Image 风格 RL 的另一条 policy-opt |
| [Wan2.2](https://github.com/verl-project/verl-omni/blob/main/examples/dancegrpo_trainer/README.md) | Diffusion video | Text → Video | DanceGRPO | 视频生成 RL 菜谱 |

另外：昇腾 NPU Dockerfile 和安装说明。

## 下一步

omni 全异步训练；MiniMax-H3、MiniCPM-o、OPD/M-OPD trainer；视频扩散靠 batch、TQ、V1 trainer 再快一点；async 下把扩散/omni rollout 焊牢；agentic RL（多阶段、多轮）。

## 社区

- **代码：** [github.com/verl-project/verl-omni](https://github.com/verl-project/verl-omni)
- **文档：** [verl-omni.readthedocs.io](https://verl-omni.readthedocs.io/en/latest/index.html)
- **贡献：** [`CONTRIBUTING.md`](https://github.com/verl-project/verl-omni/blob/main/CONTRIBUTING.md)
