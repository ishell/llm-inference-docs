---
source: https://vllm.ai/blog/2025-04-23-openrlhf-vllm
lang: en
fetched: 2026-09-04
---

# Accelerating RLHF with vLLM, Best Practice from OpenRLHF

Chinese: [zh/vllm/blog/serving/openrlhf.md](../../../../zh/vllm/blog/serving/openrlhf.md)

2025-04-23. **The OpenRLHF Team**. Repo: [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Full colocated walkthrough: [rlhf_colocate](https://docs.vllm.ai/en/latest/getting_started/examples/rlhf_colocate.html). Later engine-side pause and standard weight sync: [native-rl.md](native-rl.md). Sharded transfer: [rdt-weight-transfer.md](rdt-weight-transfer.md). Cluster bring-up: [ray-symmetric.md](ray-symmetric.md).

This post is **how a trainer hangs off the engine**. Native RL is **how the engine pauses, keeps KV, and swaps weights for the same model**. Do not merge the two. Numbers and API names follow the April 2025 post; `keep` pause / DPEP / `VLLM_SERVER_DEV_MODE` follow Native RL.

For PPO / RLHF when sample generation dominates. Not a `vllm serve` recipe, and not the 2026-recommended-only weight path — that later moved into native APIs.

## Why generation eats training

RLHF, especially PPO, is expensive. Models that emit long chain-of-thought (OpenAI-o1, DeepSeek-R1 in the post) can spend **~90% of wall time on generation**: thousands of reasoning tokens, so inference outruns the parameter update. vLLM already exposed a generation + weight-update surface for RLHF sampling.

## Design of OpenRLHF

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) stacks four pieces:

- **[Ray](https://github.com/ray-project/ray):** distributed backbone. Schedules dataflow, including rule-based reward models across nodes.
- **vLLM + Ray Executor + AutoTP:** generation. Ray executors, HuggingFace Transformers, AutoTP for weight updates.
- **ZeRO-3 + [HuggingFace Transformers](https://github.com/huggingface/transformers):** DeepSpeed memory slicing without Megatron. Load and finetune the HF way.

The post calls it the first open-source RLHF framework on Ray + vLLM + ZeRO-3. Named users then: Google, ByteDance, Alibaba, Meituan, Berkeley Starling. Same shape later shows up in [veRL](https://github.com/volcengine/verl).

Local figures (copyright remains with the original site; study copies):

![ray](../../../../assets/vllm/blog/serving/openrlhf/01-ray.png)

Ray [Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) place the vLLM engine, Actor, Critic, Reference, and Reward. The diagram draws them apart; they can **colocate** in one GPU group (hybrid engine) or bind Actor+Critic only. A central Ray Actor owns the training lifecycle. Actor ↔ vLLM weight sync uses **NCCL**, or **CUDA IPC** in the hybrid case.

## Implementing RLHF acceleration with the vLLM Ray executor

A custom `WorkerExtension` moves weights between trainer and inference. Env:

- `VLLM_RAY_PER_WORKER_GPUS` — GPU fraction per worker (can be &lt; 1 so training colocates).
- `VLLM_RAY_BUNDLE_INDICES` — which placement-group bundles this engine owns.

`ColocateWorkerExtension` is written to work on both vLLM V0 and V1 of that vintage:

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
            list_args[6] = device_id  # current process device
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
```

Pop top-level `CUDA_VISIBLE_DEVICES` before constructing the LLM so Ray does not rewrite visibility in the parent while workers disagree:

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

Read: four GPUs, one placement group; two TP2 engines on bundles `[0,1]` and `[2,3]`; each worker claims **0.4** GPU so training actors can share the card. `num_gpus=0` is deliberate — cards come from the bundle, not a second Ray allocation. Demo model: `facebook/opt-125m`. Coherence via CUDA IPC or NCCL.

The docs example also initializes Ray with a GPU count, builds the placement group, and defines **training actors** (init + push weights) beside the **inference engines**.

## Acknowledgements

vLLM: [Kaichao You](https://github.com/youkaichao) (leads RLHF integration), [Cody Yu](https://github.com/comaniac), [Rui Qiao](https://github.com/ruisearch42), and others. OpenRLHF: [Jian Hu](https://github.com/hijkzzz) (leads), [Songlin Jiang](https://github.com/HollowMan6), [Zilin Zhu](https://github.com/zhuzilin), [Xibin Wu](https://github.com/wuxibin89), and others on Ray, the vLLM wrapper, and the hybrid engine.
