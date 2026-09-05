---
source: https://vllm.ai/blog/2026-05-28-native-rl-apis
lang: en
fetched: 2026-09-04
---

# Native RL APIs in vLLM

Chinese: [zh/vllm/blog/serving/native-rl.md](../../../../zh/vllm/blog/serving/native-rl.md)

2026-05-28. Aaron Hao, Sumanth Hegde, Kyle Sayers, Kourosh Hakhamaneshi, and the vLLM team. Docs: [Weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/), [Async RL](https://docs.vllm.ai/en/latest/training/async_rl/). Examples: [`examples/rl`](https://github.com/vllm-project/vllm/tree/main/examples/rl). HTTP weight-transfer and pause endpoints need `VLLM_SERVER_DEV_MODE=1`. Study note.

As post-training scales, two problems keep showing up:

1. Weight sync between training and inference is ad-hoc and duplicated across RL frameworks.
2. Asynchronous RL gets fragile at scale, especially in P/D and DPEP deployments.

This post adds two things in vLLM:

1. Native weight-syncing APIs — a standard interface for RL frameworks.
2. Better async RL: a new pause mode, and deadlock fixes for DPEP.

Sleep Mode ([sleep-mode.md](../../architecture/sleep-mode.md)) keeps the process and swaps **models**. These APIs keep the process and swap **new weights of the same model**.

## Native Weight Syncing APIs

### Background

In online RL, vLLM weights must be synced so rollouts come from the latest (or a recent) checkpoint.

![rl system overview](../../../../assets/vllm/blog/serving/native-rl/01-rl_system_overview.png)

Figure 1: RL system overview.

Weight loading used to live in each RL framework, usually by extending vLLM workers with custom receive/load logic. That works, and it hurts:

- **Added complexity:** authors maintain worker extensions; popular transports should be native.
- **Duplicated effort:** most frameworks reinvent packed tensor transfer and RPC endpoints.
- **Version locking:** ad-hoc pre/post-processing of received weights so vLLM can load them, then locked to a vLLM version.

### New APIs

Four phases, pluggable backend (`WeightTransferEngine`):

1. **Initialization** (`init_weight_transfer_engine`): open the trainer ↔ inference channel. Once, before the training loop.
2. **Start weight update** (`start_weight_update`): after each step (or batch of steps). Prepare workers to receive.
3. **Update weights** (`update_weights`): all or a subset. Callable multiple times for chunked transfer.
4. **Finish weight update** (`finish_weight_update`): post-process (e.g. quantization).

Matching APIs exist at the API server and the engine.

Then-current backends:

1. **NCCL:** NCCL broadcast between trainer and inference workers on **separate** GPUs.
2. **IPC:** CUDA IPC, same-device transfer via shared-memory handles.

Both support a packed implementation to cut serialization. Init and update usually carry **transport** logic (framework-custom). Start and finish are control messages: transport-agnostic pre/post-processing inside vLLM.

HTTP weight-transfer endpoints require `VLLM_SERVER_DEV_MODE=1`.

Register a custom engine with `WeightTransferEngineFactory.register_engine(...)`.

### Example: NCCL + FP8 quantization

![weight transfer nccl](../../../../assets/vllm/blog/serving/native-rl/02-weight_transfer_nccl.svg)

Figure 2: Weight transfer via NCCL with FP8 quantization on vLLM.

**1. Configure the engine**

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)
```

**2. Initialize communication.** Trainer rank 0 and all inference workers join one NCCL process group.

```python
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# Inference side
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,          # offset accounts for trainer rank 0
            world_size=world_size,  # trainer + all inference workers
        )
    )
)

# Training side
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

**3. Send from the trainer.** `WeightTransferEngine.trainer_send_weights` takes an iterable of parameters and starts transfer for all or a subset. Packed tensor broadcasting batches small tensors into a larger buffer.

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=group,
    packed=True,  # packed broadcasting
)

# send weights from an AutoModelForCausalLM instance
NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

**4. Receive in the inference engine** (async, while the trainer sends):

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

llm.start_weight_update()
llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)
llm.finish_weight_update()
```

### Customizing Weight Transfer

Implement and register a `WeightTransferEngine`:

```python
from dataclasses import dataclass
from typing import Iterator, Callable, Any
from torch import Tensor
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

@dataclass
class MyInitInfo(WeightTransferInitInfo):
    """Custom initialization info."""
    ...

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    """Custom update info."""
    ...

class MyWeightTransferEngine(WeightTransferEngine):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo):
        ...

    def receive_weights(
        self,
        update_info: MyUpdateInfo,
        load_weights: Callable[[list[tuple[str, Tensor]]], None],
    ):
        ...

    @classmethod
    def trainer_send_weights(
        cls,
        iterator: Iterator[tuple[str, Tensor]],
        trainer_args: dict[str, Any] | Any,
    ):
        ...

from vllm.distributed.weight_transfer import WeightTransferEngineFactory
WeightTransferEngineFactory.register_engine(
    "my_weight_transfer", MyWeightTransferEngine
)
```

`trainer_send_weights` is optional: it encodes trainer-side send logic; you do not have to structure sends that way.

As a then-current prototype, sharded transfer in the style of [Etha](https://github.com/cmriat/Etha) is sketched [here](https://github.com/hao-aaron/vllm/blob/89c951b3296578c60cbb82e05ca3d1734364ba8c/examples/rl/sharded_reloading/README.md). Later large-scale cousin: [rdt-weight-transfer.md](rdt-weight-transfer.md).

## Improved Pause/Resume for Asynchronous RL

In async RL, weights update while inference requests are still in flight. The usual three beats: pause generation, transfer weights, resume. The user chooses what to do with in-flight requests (abort, or continue from already generated tokens) and whether to keep or discard the KV cache.

![async rl](../../../../assets/vllm/blog/serving/native-rl/03-async_rl.svg)

Figure 3: Async RL diagram, inspired by [AReaL](https://arxiv.org/pdf/2505.24298v3). Training and generation overlap; training uses **4** samples per step. After a step: pause all engines, update weights, **discard KV**, resume. KV is recomputed on resume; generation continues.

### Keep Mode

To update weights while the engine is running: `pause_generation` / `resume_generation`. HTTP: `POST /pause`, `POST /resume` (still needs `VLLM_SERVER_DEV_MODE=1`). Previously `AsyncLLMEngine.pause_generation` had two modes: abort all requests, or wait for them to finish. The post adds a third: **keep**.

| Mode | Explanation | Client-side impact | Asynchronous RL possible? |
| --- | --- | --- | --- |
| `abort` | Abort all ongoing requests | Client must retry | Yes |
| `wait` | Wait for all ongoing requests | Client need not retry | No — generation must finish before the weight update |
| `keep` | Pause ongoing requests | Client need not retry | Yes |

```python
await engine.pause_generation(mode="keep")
# update weights here
await engine.resume_generation()
```

In `keep`: ongoing requests are paused, not discarded. The scheduler stops; state is preserved.

### Fixing Deadlocks in DPEP

Large-scale async RL needs coordinated in-flight weight updates in DPEP. `DPCoordinator` makes sure generation across vLLM ranks does not deadlock: each DP rank runs a forward while **any** DP rank has active scheduled requests.

![dp generate](../../../../assets/vllm/blog/serving/native-rl/04-dp_generate.svg)

Figure 4: DP-coordinated generation across vLLM ranks.

Async RL on DP used to deadlock: some engines had already paused, others were still serving and waiting for everyone to join. Pause state lived in `AsyncLLM`; DP messages flew between `EngineCore` processes and `DPCoordinator`. For DP world size **2**:

1. API Server DP Rank 0 gets a generation request, forwards it to `EngineCore`, sends `FIRST_REQ` to `DPCoordinator` to start a wave. Rank 0 `EngineCore` begins a scheduler step.
2. Controller `POST /pause` to both engines. Pause is set on `AsyncLLM`; new requests are not forwarded to `EngineCore`. API servers on all DP ranks return immediately.
3. Trainer issues weight updates. Rank 0 `EngineCore` is already in the forward, waiting for other DP ranks (the writeup skips `start_weight_update` / `finish_weight_update` here for brevity).
4. Weight update reaches Rank 1 `EngineCore`; that replica enters an NCCL broadcast and waits. The update is queued on Rank 0 `EngineCore`.
5. `DPCoordinator` sends `START_DP_WAVE` to Rank 1 `EngineCore`; the message is queued.
6. Ranks sit in different collectives. Deadlock.

![vllm deadlock](../../../../assets/vllm/blog/serving/native-rl/05-vllm_deadlock.svg)

Figure 5: A deadlock that was possible in DPEP deployments.

Two changes:

**1. Move pause into `EngineCore`.** Pause is tracked in the scheduler, not at the `AsyncLLM` entrypoint. Fewer races between pause and generation.

**2. Two-phase pause/resume.**

- **Phase 1 (local pause):** each engine stops scheduling but still steps if inbound `START_DP_WAVE` arrives, so it can join required forwards.
- **Phase 2 (global pause):** every **32** steps, all ranks all-reduce to see if any DP rank has pending requests. The same all-reduce now also checks that every engine is in local pause. If all agree, they stop together.

So: no rank stuck waiting; `START_DP_WAVE` is honored even after a pause request; workers transition together.

The same DP=2 story, after the fix:

1. Rank 0 API server gets a generation request, forwards to `EngineCore`, `FIRST_REQ` to `DPCoordinator`. Rank 0 `EngineCore` starts a scheduler step.
2. Controller `/pause` both engines. Pause is forwarded to both `EngineCore`s. Rank 0 queues pause until the step finishes. **API servers do not return yet.**
3. Rank 0 `EngineCore` starts a forward; workers wait on all-to-all.
4. Rank 1 `EngineCore` gets pause, enters **local pause**.
5. `DPCoordinator` sends `START_DP_WAVE` to Rank 1 `EngineCore`.
6. Rank 1 `EngineCore` joins the forward. Forward completes because both DP ranks joined.
7. Rank 0 `EngineCore` processes pause, enters local pause.
8. Both ranks hit the periodic all-reduce, see local pause everywhere, enter **global pause**.
9. API servers return from `/pause`.
10. Trainer issues weight updates. API servers forward them to `EngineCore` (again omitting start/finish in the writeup).
11. All vLLM workers enter the NCCL broadcast; trainer starts NCCL broadcast.
12. Weight update finishes.

![vllm no deadlock](../../../../assets/vllm/blog/serving/native-rl/06-vllm_no_deadlock.svg)

Figure 6: Deadlock-free pause/resume in DPEP with the two-phase protocol.

## Validation

### SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL) talks to inference over HTTP: native weight-sync APIs plus native `/pause` and `/resume`. Integration: [SkyRL inference architecture](https://docs.skyrl.ai/docs/getting-started/inference_architecture). Demo: async training of Qwen3-1.7B on the original DAPO recipe ([example script](https://github.com/NovaSky-AI/SkyRL/blob/dec7137d9c57db59458a677de09add0b24413f26/examples/train/algorithms/dapo/run_dapo_qwen3_1.7b_aime_fully_async_onestep.sh)).

![skyrl validation](../../../../assets/vllm/blog/serving/native-rl/07-skyrl_validation.svg)

Figure 7: Async training of Qwen3-1.7B on the DAPO recipe in SkyRL using the native RL APIs.

### Prime-RL at scale (Wide-EP, fully async)

Prime-RL ran `zai-org/GLM-5.1-FP8` with P/D disaggregation across **16** 8×H200 nodes: **2** replicas of **4P+4D**, **DPEP32** for both Prefill and Decode. CPU KV offload **1 TB per node**. Routing: `vllm-router` with cache-aware sticky sessions. Trainer: BF16 `zai-org/GLM-5.1` on another **16** 8×H200 nodes, custom math env, [IcePop](https://arxiv.org/abs/2510.18855). Stable for **100+** steps: eval going up, RL curve up, KL mismatch stable, weight updates progressing.

![prime rl](../../../../assets/vllm/blog/serving/native-rl/08-prime_rl.svg)

Figure 8: Prime-RL fully async RL with `zai-org/GLM-5.1-FP8` across 16 8×H200 nodes.

## Conclusion

Then-current follow-ons in the vLLM RL community: a [K8s-native weight transfer engine](https://github.com/vllm-project/vllm/pull/40828), and [sharding-aware, RDMA-native weight transfer](https://github.com/vllm-project/vllm/issues/40822) in a generic form. Tracker: [vLLM RL Roadmap](https://github.com/vllm-project/vllm/issues/41733).

Docs: [Weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/), [Async RL](https://docs.vllm.ai/en/latest/training/async_rl/). Try: [`examples/rl`](https://github.com/vllm-project/vllm/tree/main/examples/rl).

## Acknowledgements

- **Prime-RL** (especially [Matej Sirovatka](https://github.com/S1ro1)) and [Junjie Zhang](https://github.com/junjzhang) — large-scale validation and debugging.
- **NemoRL** — optimized packed tensor implementation.
- [Robert Shaw](https://github.com/robertgshaw2-redhat) — organizing RL work.
- [Kyle Sayers](https://github.com/kylesayrs) — quantized weight reloading via layerwise reloading.
