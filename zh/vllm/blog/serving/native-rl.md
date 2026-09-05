---
source: https://vllm.ai/blog/2026-05-28-native-rl-apis
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Native RL APIs：权重同步别再每家写一套 worker 补丁

英文对照：[en/vllm/blog/serving/native-rl.md](../../../../en/vllm/blog/serving/native-rl.md)  
原文：https://vllm.ai/blog/2026-05-28-native-rl-apis  
2026-05-28。Aaron Hao、Sumanth Hegde、Kyle Sayers、Kourosh Hakhamaneshi，以及 vLLM 团队。文档：[Weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/)、[Async RL](https://docs.vllm.ai/en/latest/training/async_rl/)。例子在 vLLM [`examples/rl`](https://github.com/vllm-project/vllm/tree/main/examples/rl)。HTTP 上的权重传送和 pause 端点要 `VLLM_SERVER_DEV_MODE=1`。

后训练一旦铺开，两件疼会反复来敲门：

1. 训练和推理之间的权重同步，每家框架自己补一套，重复劳动。
2. 异步 RL 在规模上发脆，尤其是 P/D 和 DPEP。

这篇给 vLLM 两样东西：

1. 原生权重同步 API——给 RL 框架一条标准接口。
2. 更好的异步 RL：多一种 pause，以及 DPEP 上的死锁修补。

Sleep Mode（[sleep-mode.md](../../architecture/sleep-mode.md)）让进程活着换**模型**；这套 API 让进程活着换**同一模型的新权重**。

## Native Weight Syncing APIs

### Background

在线 RL 里，vLLM 的权重必须定期对齐，rollout 才是最新（或最近一版）checkpoint 吐出来的。

![rl system overview](../../../../assets/vllm/blog/serving/native-rl/01-rl_system_overview.png)

Figure 1：RL 系统总览。

以前权重加载住在各家 RL 框架里，常见做法是给 vLLM worker 加一套自己的接收/加载。能跑，也疼：

- **复杂度：** 作者要维护 worker 扩展；常用传输本该是原生的。
- **重复劳动：** packed tensor 传送、RPC 端点，几乎每家写一遍。
- **版本锁死：** 收到的权重要先做一套临时的前/后处理，vLLM 才能 load，然后就锁在某一版 vLLM 上。

### New APIs

四拍，后端可插拔（`WeightTransferEngine`）：

1. **Initialization**（`init_weight_transfer_engine`）：训练和推理之间把通道建起来。训练循环开始前一次。
2. **Start weight update**（`start_weight_update`）：每步（或每几步）之后。让 worker 准备接收。
3. **Update weights**（`update_weights`）：全部或一部分。可多次调用，做分块传送。
4. **Finish weight update**（`finish_weight_update`）：收尾（量化一类后处理）。

API server 和 engine 两侧都有对应接口。

当时支持的后端：

1. **NCCL：** 训练和推理在**不同 GPU** 上，用 NCCL broadcast。
2. **IPC：** CUDA IPC，同设备，走共享内存句柄。

两边都支持 packed，少付序列化。init 和 update 通常带着**传输**逻辑（框架自己定制）。start 和 finish 是控制消息：与传输无关的前/后处理，住在 vLLM 里。

HTTP 权重传送端点要 `VLLM_SERVER_DEV_MODE=1`。

自己的引擎用 `WeightTransferEngineFactory.register_engine(...)` 注册。

### Example：NCCL + FP8 量化

![weight transfer nccl](../../../../assets/vllm/blog/serving/native-rl/02-weight_transfer_nccl.svg)

Figure 2：NCCL 传权重，推理侧做 FP8 量化。

**1. 给 engine 配传送**

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)
```

**2. 初始化通信。** trainer 的 rank 0 和所有推理 worker 加入同一个 NCCL process group。

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

**3. 从 trainer 发送。** `WeightTransferEngine.trainer_send_weights` 吃一份参数迭代器，给全部或一部分开传送。packed tensor broadcast 把许多小 tensor 打进更大的 buffer。

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

**4. 推理 engine 接收**（异步，trainer 一边发）：

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

实现并注册一个 `WeightTransferEngine`：

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

`trainer_send_weights` 不是必须用的：它只是把 trainer 侧的发送逻辑写成一种形状，你不必按这个形状发。

当时有一份 Etha 风格的分片传送原型：[Etha](https://github.com/cmriat/Etha)，草稿在 [这里](https://github.com/hao-aaron/vllm/blob/89c951b3296578c60cbb82e05ca3d1734364ba8c/examples/rl/sharded_reloading/README.md)。后来更大规模的亲戚：[rdt-weight-transfer.md](rdt-weight-transfer.md)。

## Improved Pause/Resume for Asynchronous RL

异步 RL 要在请求还在飞的时候换权重。通常三拍：停下生成、传送新权重、再恢复。在飞的请求怎么处置（abort，或从已经生成的 token 接着写），KV cache 留不留，都由使用者选。

![async rl](../../../../assets/vllm/blog/serving/native-rl/03-async_rl.svg)

Figure 3：异步 RL 示意，灵感来自 [AReaL](https://arxiv.org/pdf/2505.24298v3)。训练和生成重叠；每步训练用 **4** 条 sample。一步结束：所有 engine pause，更新权重，**丢掉 KV**，再 resume。恢复时重算 KV，生成接着往前。

### Keep Mode

引擎还在跑、又要安全换权重：`pause_generation` / `resume_generation`。HTTP：`POST /pause`、`POST /resume`（同样要 `VLLM_SERVER_DEV_MODE=1`）。以前 `AsyncLLMEngine.pause_generation` 只有两种：abort 全部请求，或等它们说完。这篇加上第三种：**keep**。

| Mode | 含义 | 客户端 | 异步 RL 做不做得到？ |
| --- | --- | --- | --- |
| `abort` | 中止所有在飞请求 | 客户端必须重试 | 可以 |
| `wait` | 等所有在飞请求结束 | 客户端不必重试 | 不行——权重更新前生成必须说完 |
| `keep` | 暂停在飞请求 | 客户端不必重试 | 可以 |

```python
await engine.pause_generation(mode="keep")
# update weights here
await engine.resume_generation()
```

`keep` 里：在飞请求被暂停，不丢掉。scheduler 停住，状态还在。

### Fixing Deadlocks in DPEP

大规模异步 RL 要在 DPEP 上协调「请求还在飞、权重已经要换」。`DPCoordinator` 保证各 vLLM rank 的生成不会把自己卡死：只要**任何一个** DP rank 还有已调度的活，每个 DP rank 都要跑一次 forward。

![dp generate](../../../../assets/vllm/blog/serving/native-rl/04-dp_generate.svg)

Figure 4：跨 vLLM rank 的 DP 协调生成。

以前 DP 上的异步 RL 容易死锁：有人已经 pause，有人还在干活、等所有人入场。pause 状态记在 `AsyncLLM` 里；DP 消息在 `EngineCore` 进程和 `DPCoordinator` 之间飞。DP world size **2** 时可以走出这一幕：

1. API Server DP Rank 0 接到生成请求，转给 `EngineCore`，向 `DPCoordinator` 发 `FIRST_REQ` 开一波。Rank 0 的 `EngineCore` 开始新的 scheduler step。
2. Controller 对两个 engine `POST /pause`。pause 记在 `AsyncLLM` 上，新请求不再转给 `EngineCore`。各 DP rank 的 API server **立刻返回**。
3. Trainer 发出权重更新。Rank 0 的 `EngineCore` 已经进了 forward，在等别的 DP rank（原文这里故意不写 `start_weight_update` / `finish_weight_update`）。
4. 权重更新到了 Rank 1 的 `EngineCore`，这个 replica 进入 NCCL broadcast 等待。更新在 Rank 0 的 `EngineCore` 上排队。
5. `DPCoordinator` 给 Rank 1 的 `EngineCore` 发 `START_DP_WAVE`，消息被排住。
6. 各 rank 坐在不同的 collective 里。死锁。

![vllm deadlock](../../../../assets/vllm/blog/serving/native-rl/05-vllm_deadlock.svg)

Figure 5：当时 DPEP 上可能走出的死锁。

两处改动：

**1. 把 pause 下沉到 `EngineCore`。** pause 记在 scheduler 里，不再记在 `AsyncLLM` 入口。pause 和生成之间少一层赛跑。

**2. 两阶段 pause/resume。**

- **Phase 1（local pause）：** 每个 engine 停调度，但仍响应进来的 `START_DP_WAVE`，把这一拍该参加的 forward 做完。
- **Phase 2（global pause）：** 每 **32** 步，所有 rank 做一次全局 all-reduce，看有没有 DP rank 还握着未完成的请求。同一拍 all-reduce 现在也问：是不是大家都已经 local pause。齐了，就一起停。

于是：没有人卡在等待里；即便已经收到 pause，`START_DP_WAVE` 仍被尊重；workers 一起换状态。

同一套 DP=2 的故事，修好之后：

1. Rank 0 的 API server 接到生成请求，转给 `EngineCore`，`FIRST_REQ` 给 `DPCoordinator`。Rank 0 的 `EngineCore` 开始 scheduler step。
2. Controller 对两个 engine `/pause`。pause 转到两边的 `EngineCore`。Rank 0 把 pause 排到这一 step 做完。**API server 这时还不返回。**
3. Rank 0 的 `EngineCore` 开始 forward；workers 在 all-to-all 上等待。
4. Rank 1 的 `EngineCore` 收到 pause，进入 **local pause**。
5. `DPCoordinator` 给 Rank 1 的 `EngineCore` 发 `START_DP_WAVE`。
6. Rank 1 的 `EngineCore` 加入 forward。两边都到了，forward 做完。
7. Rank 0 的 `EngineCore` 处理 pause，进入 local pause。
8. 两边碰上周期性 all-reduce，看见大家都是 local pause，进入 **global pause**。
9. API server 从 `/pause` 返回。
10. Trainer 发出权重更新。API server 转到 `EngineCore`（原文同样略过 start/finish）。
11. 所有 vLLM worker 进入 NCCL broadcast；trainer 开始 NCCL broadcast。
12. 权重更新顺利结束。

![vllm no deadlock](../../../../assets/vllm/blog/serving/native-rl/06-vllm_no_deadlock.svg)

Figure 6：DPEP 上两阶段协议之后，pause/resume 不再死锁。

## Validation

### SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL) 用 HTTP 和推理说话：原生权重同步 API，再加上原生 `/pause`、`/resume`。接入见 [SkyRL inference architecture](https://docs.skyrl.ai/docs/getting-started/inference_architecture)。演示：Qwen3-1.7B 按原来的 DAPO 配方做异步训练（[示例脚本](https://github.com/NovaSky-AI/SkyRL/blob/dec7137d9c57db59458a677de09add0b24413f26/examples/train/algorithms/dapo/run_dapo_qwen3_1.7b_aime_fully_async_onestep.sh)）。

![skyrl validation](../../../../assets/vllm/blog/serving/native-rl/07-skyrl_validation.svg)

Figure 7：SkyRL 用原生 RL API，按 DAPO 配方异步训练 Qwen3-1.7B。

### Prime-RL 大规模（Wide-EP，全异步）

Prime-RL 用 `zai-org/GLM-5.1-FP8`，P/D 分离，铺在 **16** 台 8×H200 上：**2** 组 **4P+4D**，Prefill 和 Decode 都是 **DPEP32**。CPU KV offload 每机 **1 TB**。路由：`vllm-router`，cache-aware 粘会话。Trainer：另一边 **16** 台 8×H200 上跑 BF16 的 `zai-org/GLM-5.1`，自定义数学环境，算法是 [IcePop](https://arxiv.org/abs/2510.18855)。**100+** step 稳定：eval 往上、RL 曲线往上、KL mismatch 稳住、权重更新正常往前。

![prime rl](../../../../assets/vllm/blog/serving/native-rl/08-prime_rl.svg)

Figure 8：Prime-RL 在 16 台 8×H200 上，用 `zai-org/GLM-5.1-FP8` 做全异步 RL。

## Conclusion

当时 vLLM RL 社区还在往下做：一份 [K8s-native weight transfer engine](https://github.com/vllm-project/vllm/pull/40828)，以及通用的 [sharding-aware、RDMA-native 权重传送](https://github.com/vllm-project/vllm/issues/40822)。跟踪：[vLLM RL Roadmap](https://github.com/vllm-project/vllm/issues/41733)。

文档：[Weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/)、[Async RL](https://docs.vllm.ai/en/latest/training/async_rl/)。上手：[`examples/rl`](https://github.com/vllm-project/vllm/tree/main/examples/rl)。

## Acknowledgements

- **Prime-RL**（尤其 [Matej Sirovatka](https://github.com/S1ro1)）和 [Junjie Zhang](https://github.com/junjzhang)——大规模验证和排错。
- **NemoRL**——优化过的 packed tensor 实现。
- [Robert Shaw](https://github.com/robertgshaw2-redhat)——把 RL 相关的事组织起来。
- [Kyle Sayers](https://github.com/kylesayrs)——用 layerwise reloading 让量化权重也能重新装上。
