---
source: https://vllm.ai/blog/2026-07-14-vllm-tilert-pd
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# TileRT：换一只 Decode，不必养 vLLM fork

英文对照：[en/vllm/blog/serving/tilert.md](../../../../en/vllm/blog/serving/tilert.md)  
原文：https://vllm.ai/blog/2026-07-14-vllm-tilert-pd  
2026-07-14。署名 **TileRT team**。TileRT **0.1.5**：[PyPI](https://pypi.org/project/tilert/)（`pip install tilert`；Python 3.12，CUDA 13 wheels）、仓库 [tile-ai/TileRT](https://github.com/tile-ai/TileRT)。接头是 vLLM V1 的公开 connector：`KVConnectorBase_V1`，落在 `MultiConnector` 下，用 `kv_connector_module_path` 加载。**零改 vLLM**：不 fork、不打补丁、不包一层内部 worker。当时演示：**GLM-5 / 5.1**、**DeepSeek-V3.2**。Prefill 侧要开 MTP；一台 TileRT decode 节点一次只有 **一只** in-flight 请求。

和别的 connector P/D 是同一扇门，不是再写一套协议：[Mooncake](mooncake.md) / NIXL 搬字节；[Router](router.md) 是库存 vLLM 的 P/D 网关（这篇的入口是 TileRT 自己的 `pd_router`）；集群级拆分见 [large-scale.md](large-scale.md)；另一只「只加 connector、不动引擎」的拆法是 [moriio.md](moriio.md)。页上的柱子不是你的 SLA。

适用：延迟绑死、per-user 速度比集群吞吐更要紧，模型又在 TileRT 名单里。不适合：把这篇当成通用 Decode 的替代默认；也不适合指望一台 TileRT 节点高并发 batched Decode。

P/D 分离把 compute-bound 的 Prefill 和 memory-bandwidth-bound 的 Decode 拆开，vLLM 已经用一等公民的 connector 做这件事。相位一分，**Decode 就可以插拔**。Prefill 池、调度、缓存、对外 API 留在原地；decode 池变成一道选择题。

这篇交的就是那道题：**vLLM Prefill + TileRT Decode**，跟 TileRT 0.1.5 一起出。延迟敏感的流量拿到 TileRT 原生的 per-user decode 速度；其余部署仍是库存 vLLM。

## 为什么还要第二只 Decode？

vLLM 自带的 Decode 仍是正确的 **默认**：高吞吐、batched serving，模型和硬件覆盖面很大。可有一类负载在涨——agent 循环、交互式编程助手、实时语音——它们在意的不是集群合计吞吐，而是 token 到达 **每一个** 用户有多快。这些活是延迟绑死的。原生 Decode 和 TileRT 落在同一条吞吐–延迟前沿的不同点上，所以能拼在一起。

TileRT 是一只新的推理运行时，目标只有一件：把 per-user decode 速度往硬件极限推。他们另文写过 [速度正在变成一种 scaling 维度](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html)。这篇不是引擎说明书。更实际的问题：能不能换一只专门的 decode 引擎，**又不丢掉** 你已经依赖的生态——OpenAI 兼容 API、调度、prefix caching、tool calling、vLLM 的运维成熟度？

集成想把这道税压小：

- **Prefill 仍是 vLLM。** 调度、chunked prefill、prefix caching——不动。
- **对外表面仍是 vLLM。** 同一套 API、同一请求格式、同一套工具。
- **只有 Decode 变，而且只变你派过去的那路流量。** TileRT 这套可以和现有 vLLM 部署并排；每路负载选自己的入口。

## 架构：并排放，是故意的

核心原则：**零改 vLLM**。不 fork、不打补丁、不包内部 worker。东西全部活在 V1 的公开扩展面上：一个 `KVConnectorBase_V1` 实现，用 `MultiConnector` 组合，经标准 `kv_connector_module_path` 加载。这不只是好看：加一只 TileRT decode 池，不该把你已经在跑的 vLLM 弄不稳；升级 vLLM，也不该再移植一座 fork。

本地图（原文版权仍归原站；学习对照用）：

![pd arch](../../../../assets/vllm/blog/serving/tilert/01-pd_arch.png)

**架构图。** 延迟敏感流量由 TileRT PD router 打标，TileRT connector 认领。一般流量走原生 P/D。两边共用 **一份库存 vLLM Prefill 池**，在 `MultiConnector` 下拼起来。

**Routing。** 一只轻量 router 挡在 TileRT 池前面。每个请求设 `max_tokens=1`（vLLM 做 Prefill、吐第一个 token），并把目标 decode 节点写进透传字段：`kv_transfer_params = {"tilert_host": ..., "tilert_ctrl_port": ...}`。原生池的流量仍走原来的 disaggregation proxy，不改。

**Claim filtering。** TileRT connector **只** 认带标记的请求；其余严格 no-op。两只 decode 池可以共享一台 Prefill——甚至 **同一轮 forward batch**。给一部分流量上 TileRT，其余不受影响。

**纯 producer。** Connector 只当 `kv_producer`。不碰调度、不碰采样；Prefill 之后抽出状态、送走。其余方面，这台 Prefill 就是库存 vLLM 服务器。

## 交接怎么发生

跨引擎 P/D 要能落地，三件事得同时真：传输要快，不能拖慢 Prefill，Decode 必须从 Prefill 停下的地方接着说。

**数据面。** Prefill 之后，请求的 attention 状态——压缩 KV、sparse-attention 的 index cache、一小撮 metadata——以 **RDMA 单边写** 进预先登记好的 GPU buffer，落到 decode 节点。搬运引擎是 **Mooncake 或 NIXL**。没有中间序列化，不在主机内存垫一层。交接 **协议** 不依赖底下是谁在搬字节。

**和 Prefill 完全重叠。** 抽状态发生在前向窗口里：在 cache block 还能被回收之前，先把状态拷到 staging buffer；真正的网络传输交给后台 sender。派给 TileRT 的请求不挡住下一轮 Prefill，包括同一 batch 里那些走原生池的请求。

**注入正在跑的引擎。** 状态到达后转成 TileRT 的原生布局，直接打进一台 **已经在跑** 的引擎。Decode 立刻开始，第一步就带着多 token 投机（MTP）。

## 评估

![glm5 tilert mtp](../../../../assets/vllm/blog/serving/tilert/02-glm5_tilert_mtp.png)

**Decode 速度图。** GLM-5.1-FP8，**8× NVIDIA B200**，TileRT **v0.1.5**。输出长度 **1K**，输入长度 **1K–192K**。三根柱：TileRT **不开 MTP**；开 MTP、平均 acceptance length **3.2**；最好情况下 MTP acceptance **4.0** 的峰值。

原文没有把 tok/s / TPS 写成表——柱高只在图里。

## 选哪一只 decode 池

per-user token 速度是硬约束时，把流量派到 **TileRT Decode**——交互式 agent、实时助手、有延迟 SLO 的推理——**并且** 模型在 TileRT 支持名单里。

要最大合计吞吐、高并发 batch、以及通用 Decode 才能覆盖的长尾模型和功能，留在 **原生 vLLM Decode**。

两边都是 OpenAI 兼容表面。搬负载是 **改路由**，不是改客户端。

**这一版的限制：** 一台 TileRT decode 节点同一时刻只伺候 **一只** in-flight 请求；router 做 gated dispatch 和反压。当时模型覆盖：**GLM-5/5.1**、**DeepSeek-V3.2**，后面还会加。

## 上手

Prefill 和 Decode 节点都要装 TileRT；Prefill 侧需要 connector 插件。

```bash
# 0. One-time: convert the HF checkpoint to TileRT's weight format
python -m tilert.models.preprocess.weight_converter \
    --model_type glm-5 \
    --model_dir /path/to/GLM-5.1 \
    --save_dir /path/to/tilert-glm5.1-weights

# 1. TileRT decode node
python -m tilert.pd_vllm.decode_server \
    --engine tilert --model glm5 \
    --model-weights-dir /path/to/tilert-glm5.1-weights \
    --with-mtp --max-seq-len 202752 \
    --kv-cache-dtype fp8 \
    --ctrl-port 5556 --http-port 5557

# 2. vLLM prefill (stock vLLM; the connector loads as a plugin).
#    The MTP speculative config is required: prefill populates the
#    draft-layer KV that decode-side speculation resumes from.
vllm serve /path/to/GLM-5.1 \
    --served-model-name glm5.1 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --trust-remote-code \
    --return-tokens-as-token-ids \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8_ds_mla \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --kv-transfer-config '{
        "kv_connector": "TileRTConnector",
        "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config":{
            "tilert_host":"[TILERT_DECODE_SERVER_IP]",
            "tilert_ctrl_port":5556,
            "tilert_model":"glm5",
            "tilert_max_seq_len":202752
        }
    }'

# 3. Router: OpenAI-compatible ingress for the TileRT pool
python -m tilert.pd_vllm.pd_router \
    --vllm-url http://prefill-node:8000 \
    --decode decode-node:5556:5557 \
    --model-path /path/to/GLM-5.1 \
    --port 23333
```

Prefill 侧 `--speculative-config` 里 `"method": "mtp"` 是 **必须** 的：Prefill 要填好 draft 层 KV，Decode 侧的投机才接得上。Decode CLI 开 `--with-mtp`。示例序列上限 **202752**。Decode 的 KV dtype 是 `fp8`；vLLM Prefill 是 `--kv-cache-dtype fp8_ds_mla`。Router 的 `--decode` 格式：`host:ctrl-port:http-port`。入口端口 **23333**。

要让 TileRT 池和原生 vLLM decode 池共用 **一台** Prefill，把两只 connector 写进 `MultiConnector`。他们验证过的配置是 **全程 NIXL**（原生池用 vLLM 标准 `NixlConnector`，TileRT 池用 NIXL 模式的 TileRT connector），共享 Prefill 只依赖 **一份** 传输库。改的只是 Prefill 的 `--kv-transfer-config`。

## 往后看

P/D 分离正在悄悄改推理栈的形状：不再是一台引擎包打天下，而是共享 serving 层后面 **几只各专一事的引擎**。vLLM 的 connector 让这种拼法今天就能做；这次集成是其中一个例子。TileRT 敢在一个维度上挖这么深，也是因为 serving 层是共享的、接口是开的——不必为了加速 Decode 把其余一切重造一遍。

他们想听社区的话：集成面、哪些负载真的受益、下一步该支持哪些模型。

## 致谢

vLLM 社区设计了 V1 connector，才有这次 **零修改** 集成。Mooncake 与 NIXL 提供 RDMA 传输引擎。[Inferact Inc.](https://inferact.ai/) 协作改进 vLLM–TileRT 对接。
