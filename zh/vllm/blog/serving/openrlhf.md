---
source: https://vllm.ai/blog/2025-04-23-openrlhf-vllm
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# OpenRLHF：生成占 RLHF 墙钟的九成

英文对照：[en/vllm/blog/serving/openrlhf.md](../../../../en/vllm/blog/serving/openrlhf.md)  
原文：https://vllm.ai/blog/2025-04-23-openrlhf-vllm  
2025-04-23。署名 **The OpenRLHF Team**。仓库：[OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)。完整 colocated 例子：[rlhf_colocate](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html)。后来引擎侧的 pause / 标准权重同步：[native-rl](native-rl.md)；RDT 分片搬运：[rdt-weight-transfer](rdt-weight-transfer.md)；Ray 起集群：[ray-symmetric](ray-symmetric.md)。

这篇是 **训练框架怎么挂上引擎**。Native RL 是 **引擎怎么暂停、留 KV、换同一模型的新权重**。两篇不要并成一句。数字和 API 跟 2025-04 这篇走；`keep` pause / DPEP / `VLLM_SERVER_DEV_MODE` 跟 Native RL。

适用：PPO / RLHF 采样太慢、想用 Ray 把 vLLM 和 ZeRO-3 训程拼在一起。不适合：只想 `vllm serve` 对外；也不适合把这篇的 `WorkerExtension` 当成 2026 年仍推荐的唯一权重通道。

## 为什么生成会吃掉训练

要训会推理的模型，RLHF（尤其 PPO）算力税很重。OpenAI-o1、DeepSeek-R1 这类长 chain-of-thought 更明显：逐步推理可以拉到几千 token，**生成就能占到总训练时间的约 90%**——推理比参数更新还慢。vLLM 当时已经提供一套给 RLHF 用的接口：采样，以及把新权重灌回引擎。

## OpenRLHF 的设计

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 想在性能和能用之间站住，把四件叠在一起：

- **[Ray](https://github.com/ray-project/ray)：** 分布式骨架。调度复杂数据流，也包括把基于规则的 reward 模型铺到多机。
- **vLLM + Ray Executor + AutoTP：** 推理加速。Ray Executor、HuggingFace Transformers、AutoTP 做权重更新，吞吐和显存才压得住。
- **ZeRO-3 + [HuggingFace Transformers](https://github.com/huggingface/transformers)：** DeepSpeed 的显存切法，不必上 Megatron 才能训大模。加载和微调走 HF 习惯。

原文写它是第一只基于 Ray、vLLM、ZeRO-3 的开源 RLHF 框架；当时用过的名字包括 Google、ByteDance、Alibaba、Meituan、Berkeley Starling。同一范式后来也进了 [veRL](https://github.com/volcengine/verl)。

本地图（原文版权仍归原站；学习对照用）：

![ray](../../../../assets/vllm/blog/serving/openrlhf/01-ray.png)

Ray 的 [Placement Group](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) 排 vLLM engine、Actor、Critic、Reference、Reward。图画成分开的模块，实际可以 **colocate** 进同一组 GPU：hybrid engine 里全部挤在同一 GPU 组，或只把 Actor 和 Critic 捆在一起。中央 Ray Actor 管整个训练生命周期。Actor 和 vLLM 之间的权重同步走 **NCCL**，hybrid 场景也可以走 **CUDA IPC**。

## 用 vLLM Ray Executor 加速

自定义 `WorkerExtension` 做训练↔推理的权重同步。环境变量：

- `VLLM_RAY_PER_WORKER_GPUS`：每个 worker 占多少 GPU（可小于 1，才能和训程共卡）。
- `VLLM_RAY_BUNDLE_INDICES`：这个引擎对应 placement group 里的哪些 bundle。

`ColocateWorkerExtension` 要同时兼容当时的 vLLM V0 和 V1：

```python
# rlhf_utils.py
class ColocateWorkerExtension:
    """Extension class for vLLM workers to handle weight synchronization."""
    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id  # 换成当前进程的 device
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
```

起引擎时要先把顶层的 `CUDA_VISIBLE_DEVICES` 弹掉，免得 Ray 在父进程里改可见卡，子 worker 对不齐：

```python
# main.py
class MyLLM(LLM):
    def __init__(self, *args, bundle_indices: list, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        super().__init__(*args, **kwargs)

pg = placement_group([{"GPU": 1, "CPU": 0}] * 4)
ray.get(pg.ready())

inference_engines = []
for bundle_indices in [[0, 1], [2, 3]]:
    llm = ray.remote(
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg
        )
    )(MyLLM).remote(
        model="facebook/opt-125m",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.4,
        worker_extension_cls="rlhf_utils.ColocateWorkerExtension",
        bundle_indices=bundle_indices
    )
    inference_engines.append(llm)
```

读法：4 张卡、一个 placement group；两只推理引擎各 TP2（bundle `[0,1]` 和 `[2,3]`）；每只 worker 只申报 **0.4** GPU，剩下的给同卡上的训练 actor。`num_gpus=0` 是故意的——真正的卡从 bundle 来，不要让 Ray 再分配一层。演示模型是 `facebook/opt-125m`。权重相干靠 CUDA IPC 或 NCCL。

文档里的完整例子还会：按指定 GPU 数初始化 Ray、建 placement group、同时定义 **training actors**（初始化 + 推权重）和 **inference engines**（vLLM 侍候）。

## 致谢

vLLM 侧：[Kaichao You](https://github.com/youkaichao)（牵头 RLHF 集成）、[Cody Yu](https://github.com/comaniac)、[Rui Qiao](https://github.com/ruisearch42) 以及更多贡献者。OpenRLHF 侧：[Jian Hu](https://github.com/hijkzzz)（牵头）、[Songlin Jiang](https://github.com/HollowMan6)、[Zilin Zhu](https://github.com/zhuzilin)、[Xibin Wu](https://github.com/wuxibin89) 等，贡献在 Ray、vLLM Wrapper、Hybrid Engine。
