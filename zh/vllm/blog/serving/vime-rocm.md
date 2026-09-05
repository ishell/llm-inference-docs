---
source: https://vllm.ai/blog/2026-07-10-vime-rocm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# vime on ROCm：`vllm/vime-rocm`，MI355X 上 Qwen3-8B ~4100 tok/gpu/s

英文对照：[en/vllm/blog/serving/vime-rocm.md](../../../../en/vllm/blog/serving/vime-rocm.md)  
原文：https://vllm.ai/blog/2026-07-10-vime-rocm  
2026-07-10。署名 **AMD contributors & vime community**。主线发布：[vime](vime.md)（2026-06-09）。镜像：`vllm/vime-rocm`。教程：[amd_tutorial.md](https://github.com/vllm-project/vime/blob/main/docs/en/platform_support/amd_tutorial.md)。slime：[THUDM/slime](https://github.com/THUDM/slime)。硬件插件亲戚：[hardware-plugin.md](../../architecture/hardware-plugin.md)。引擎侧 pause / 权重 API 是另一层：[native-rl](native-rl.md)。逐 bit on-policy **不是**这篇：[bitwise-rl](bitwise-rl.md)。MI355X 上 Qwen3-8B 的数字是页上的合同，不是你的 SLA。

vime 发布之后，AMD 和 vime 组把 ROCm 接上：Instinct 上跑通端到端，ROCm 专用修补往上游送，再给一份预编译容器，AMD 用户不必从源码砌栈。ROCm 路径和 CUDA 路径 knobs 同名，不等于 kernel 同形。对表时看他们给的 `train_rollout_logprob_abs_diff`，不要默认 bitwise。

本地图（原文版权仍归原站；学习对照用）：

![data buffer](../../../../assets/vllm/blog/serving/vime-rocm/01-data-buffer.png)

![image](../../../../assets/vllm/blog/serving/vime-rocm/02-image.png)

![image 1](../../../../assets/vllm/blog/serving/vime-rocm/03-image-1.png)

![image 2](../../../../assets/vllm/blog/serving/vime-rocm/04-image-2.png)

## TL;DR

- 预编译 **`vllm/vime-rocm`**。代码在 `/root/vime`。vLLM 和 Megatron-LM 预装。W&B 在线模式：要有效的 `WANDB_API_KEY`。
- Qwen3-8B 在 **MI355X** 上跑 100 step：约 **4100** `tokens_per_gpu_per_second`，略往上。`train_rollout_logprob_abs_diff` 约 **0.012**，略往下。`raw_reward` 约 **0 → 0.5–0.6**。
- 启动：`NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh`。TP=2，一只 vLLM engine，colocate，DP=1。两张卡合计约 **230 GB**。
- ROCm 上选卡用 **`HIP_VISIBLE_DEVICES`**，不要只写 `CUDA_VISIBLE_DEVICES`。权重转换带 `--no-gradient-accumulation-fusion --attention-backend flash`。
- **坑：** ROCm launcher 设了 `EVAL_ARGS=()`——训练集上的 `raw_reward` 不是 held-out eval。logprob ~0.012 和他们报的 NVIDIA 量级相当，**不是** bit-exact。AMD MoE 的 R3、完整 Router / PD、FP8 管线：路线图。
- 点名跑过、这里没另画图的：Qwen3-4B、Qwen3-8B dense、Qwen3-30B-A3B MoE。

## vime，按 ROCm 再说一遍

vime [六月那篇](vime.md) 发布。有了 ROCm，同一套 RL 后训练在 AMD Instinct 上原生能跑。

**Figure.** vime 架构（训练和 rollout 之间的 Data Buffer）。

slime 的三段式、训练–推理解耦。跟 slime 原生日径的差别：rollout 后端是 **vLLM**，不是 SGLang。

- **Training（Megatron）：** 参数更新；权重同步到 rollout。
- **Rollout（vLLM + Router）：** 采样；reward / verifier 信号。
- **Data Buffer：** prompt 注入和自定义 rollout 逻辑。

整条管线，他们说已经在 ROCm 上端到端核过。

## Why AMD Instinct

RL 后训练吃显存。每一步既要握住训练侧权重（Megatron 格式），又要握住推理侧 KV cache（vLLM rollout）。colocate 时它们抢同一池。页上说 MI300X / MI355X 合适，理由三条：

- **统一大 HBM。** MI300X 每卡 **192 GB** HBM3；MI355X **288 GB**。大模型不必为了摊内存就把张量并行拉得很凶——拓扑简单，集群更好喂满。
- **带宽。** MI300X 上 HBM3 合计 **>5 TB/s**；MI355X 上 HBM3E **8 TB/s**。RL rollout（大规模自回归 decode）本质是内存带宽绑定：每步 decode 都要从 HBM 拉 KV cache 和权重。带宽高，占大多数 RL 步时的 rollout 段才短得下来。
- **开源栈。** ROCm（HIP、LLVM、MIOpen）。vLLM 和 PyTorch 原生支持 ROCm，vime 继承 vLLM rollout 栈，**不另开一条代码路径**。名字相同，不是 kernel 相同的承诺。

## Training details（他们焊了什么）

- **Megatron-LM 后端。** ROCm 兼容 fork，外加一小块补丁：**在非 CUDA 构建上挡住 CUDA fused-kernel 初始化**。训练环：ROCm 兼容的 Megatron 补丁和 ROCm 专用启动 flag。梯度累积走 **原生 PyTorch** 路径（ROCm 支持）。HuggingFace → Megatron `torch_dist` 转换在 **单卡** 上跑，产出 Megatron 开训能 load 的布局。
- **Colocate 权重同步。** Megatron 和 vLLM 共用 GPU 池。每次 optimizer step 之后，Megatron 经 **IPC** 把新权重同步给 vLLM——没有网络来回。ROCm 上 `torch.cuda.get_device_properties(i).uuid` 给出稳定、跨进程一致的设备 UUID，所以 vime 按 UUID 键的 IPC 路由 **不用改**。
- **GPU 可见性和 Ray。** ROCm 用 `HIP_VISIBLE_DEVICES`。vime 启动脚本把它和 `CUDA_VISIBLE_DEVICES` **一起** 设上，让 Megatron actor 和 vLLM 子进程看见同一套序号。Ray 的 AMD GPU manager 配成 **不覆盖** 这些 mask。容器启动带 `--ulimit nofile=1048576:1048576`，因为 Ray 拉起全套 actor 时要这份 FD 上限。

## Getting started

预编译容器；镜像对得上宿主机，就不必从源码砌 ROCm 栈。

### 起容器

```bash
# Pull the ROCm image
docker pull vllm/vime-rocm
# Start the container
docker run -d --name vime --ulimit nofile=1048576:1048576 \
  --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --privileged \
  -e WANDB_API_KEY=$wandb_key vllm/vime-rocm
# The launch script enables W&B online mode, so a valid WANDB_API_KEY is required.

# Enter the container
docker exec -it vime bash
```

镜像里：vLLM、Megatron-LM、vime 在 `/root/vime`。

### 模型和数据

```bash
# Download model weights (Qwen3-8B)
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
# Download training dataset (dapo-math-17k)
hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir /root/dapo-math-17k
```

### 转到 Megatron `torch_dist`

先加载 Qwen3-8B 的模型配置，再转。**ROCm 上选卡用 `HIP_VISIBLE_DEVICES`**（不要只写 `CUDA_VISIBLE_DEVICES`）。

```bash
cd /root/vime && source scripts/models/qwen3-8B.sh
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/root/vime:/root/Megatron-LM \
  torchrun --nproc-per-node=1 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion --attention-backend flash \
  --hf-checkpoint /root/Qwen3-8B --save /root/Qwen3-8B_torch_dist
```

要留着的 flag：`--no-gradient-accumulation-fusion`、`--attention-backend flash`。

### 起 RL 训练

```bash
NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh
```

完整 colocate 管线：vLLM rollout worker、GRPO 环、on-policy rollout → train → weight-sync。

**配置备注：**

- `VISIBLE_GPUS` —— 两张空闲卡的序号；脚本 mask 到这两张。**TP=2**，一只 vLLM engine，**colocate**，**DP=1**。
- `NUM_ROLLOUT` —— 训练步数。默认 **3** 是冒烟；图上是 **100**。
- 两张选定卡合计约 **230 GB**。内存不够就别硬起。

换一个 `NUM_ROLLOUT` 再跑：清掉 save 目录，否则 checkpoint 对不上：

```bash
rm -rf /root/Qwen3-8B_vime/
```

## Performance results

这套 runbook 点名跑过：Qwen3-4B、Qwen3-8B（dense）、Qwen3-30B-A3B（MoE）。下面的图是 **Qwen3-8B** 那道例子。

**Figure.** MI355X 上 Qwen3-8B 的吞吐。

**100** 个训练 step 里，吞吐稳住约 **4,100** `tokens_per_gpu_per_second`，略往上。他们的读法：policy 学会更可预期的输出；更短或更齐的生成降低 decode 方差，vLLM 更好 batch。这是训练动力学，不是「只换 kernel 就加速」的宣称。

**Figure.** 同一 run 上的 `train_rollout_logprob_abs_diff`。

这支指标（训练侧 logprob 对 rollout 侧）钉在约 **0.012**，略往下。他们把功劳记在 Megatron → vLLM 的权重同步，免得 logprob 漂了把 policy gradient 带歪。GRPO 要的是稳住的低 diff；他们说这量级 **和报过的 NVIDIA 数字相当**。这 **不是** `kl_div == 0.0` 的逐 bit（那根杠在 [bitwise-rl](bitwise-rl.md)）。

**Figure.** 采样训练 prompt 上的 `raw_reward`。

step 0 靠近 **0**，到 step 100 爬到约 **0.5–0.6**。刚初始化的 policy 面对 dapo-math-17k 竞赛题，几乎解不出。训练奖励往上 = 在**采样到的训练 prompt** 上优化有进展。ROCm launcher 关了评估（`EVAL_ARGS=()`）；要泛化，得另做 held-out eval。

## Feature support roadmap on AMD

**今天（点名）：**

- GRPO
- Colocate 的训练和 rollout
- 异步（non-colocate）训练，actor 和 rollout GPU 池分开
- Megatron-LM 训练后端
- vLLM rollout 后端
- Qwen3 Dense 和 MoE

**往后（点名，不宣称已经活着）：**

- 完整 vLLM Router 和 PD 分离
- FP8 管线优化
- **AMD MoE 上的 R3（Rollout Routing Replay）** —— CUDA 发布（[vime](vime.md)）里 R3 是测过的约 **0.019 → ~0.013**；这里仍是路线图
- 异步管线性能（logprob 发散、内存泄漏）
- Agentic RL：多轮 tool calling、多 agent

页上的目标：跟着 vime 和 vLLM 的路线图往前赶。

## Acknowledgments

AMD contributors & vime community —— 页上这样署名（这篇没有个人名单；六月那篇列过 vime contributors）。

## References

- vime 仓库：[github.com/vllm-project/vime](https://github.com/vllm-project/vime)
- vime 发布：[vime.md](vime.md) / https://vllm.ai/blog/2026-06-09-announcing-vime
- AMD 教程：[docs/en/platform_support/amd_tutorial.md](https://github.com/vllm-project/vime/blob/main/docs/en/platform_support/amd_tutorial.md)
- slime：[github.com/THUDM/slime](https://github.com/THUDM/slime)
