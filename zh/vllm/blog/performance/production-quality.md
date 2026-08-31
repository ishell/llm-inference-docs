---
source: https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
lang: zh
voice: literary-study
fetched: 2026-08-31
---

# 怎样把 vLLM 保持在「能上线」的质量

英文对照：`en/vllm/blog/performance/production-quality.md`  
原文：https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality  
2026-07-16。数字会过时：当时 86k+ stars、每月 5.6M pip、2.5M 镜像、1000+ 模型架构、600+ 加速器。2026 年 6 月 main 合入 **1918** 个 commit（日均 64），CI **1300 万** job-minute，峰值 **1400** 并发 runner。H100 上干净的改动，可能在 AMD 编不过、在 B200 变慢、在某一个 backend 把输出拧歪一点。

三层门：CI 抓住会响的崩；夜间性能/精度抓住不响的慢和错；发布流程决定哪一颗 commit 能出门。

## Layer 1：CI

GitHub Actions 做 lint；committer 点头后 Buildkite 按 diff **动态**组流水线——改文档可能几条，动 kernel 可能 **100+** 并行。当时全套 **37** 组、**266** 个 job。

环境：多数 job 共用同一张分阶段 Docker（`base` CUDA → `build` 编 wheel → `runtime`；再分叉成发布镜像和 `test` 镜像）。依赖用 `pip-compile` **锁死传递依赖**——FlashInfer / nixl / transformers 某周静默升级，曾经让同一条测试周一过、周三炸。

硬件：当时 **58** 个 runner 队列，多家厂商捐卡。Buildkite agent **出站**连 HTTPS，捐助方不必开入站、VPN。K8s 上用 Agent Stack，一 job 一 Pod。H200 切 MIG：一张卡 7 片 18GB。按小时租的队列 **scale to zero**；sccache 进 S3（只有 main 的 builder 能写）；权重下到集群共享盘。看板：ci.vllm.ai。夜间 CI-analyzer 读日志、点名罪魁、丢 Slack，附带回滚 PR——大约每天 1.5 个，罪名找对约 **70%**。

绿灯仍不够：为了快和便宜，CI 会跳过很多 e2e。v0.20.0 过了 CI，几天内连打两张补丁——gpt-oss 在 Blackwell 上 TP>1 挂了，DeepSeek V4 在 GB200 上吞吐塌了。都不崩，只是慢或错。

## Layer 2：夜间性能与精度

仓库：https://github.com/vllm-project/perf-eval。每份配置：怎么起 server、什么模型、什么卡、跑哪些任务。三件事：`vllm-bench` 的 TTFT/TPOT；`lm-eval` 的 GSM8K/GPQA/AIME；Berkeley Function-Calling Leaderboard。当时每夜 + 每个 release candidate：DeepSeek V4、gpt-oss、Kimi K2.5、MiniMax、Qwen3.5、GLM 5.1、Gemma 4、Nemotron 3 Super × H200/B200/MI300X/MI355X，**17** 条配方。计划加 GB200/GB300 和 P/D 分离。结果进同一块 CI HUD，可对两张镜像拍头。

## Layer 3：两周一发

2025 年 11 月起两周节奏。周一从 main 上最绿的一次全量 CI **切 branch**；周一到周三 cherry-pick 打成 RC；每个 RC 过三道门（全量 CI + 性能 + 精度）。周三后只收修复。三道门不过就不发——「done when it's done」，但不花十年。过门之后编 7 个 wheel、11 张 Docker（CUDA 12.9/13.0、CPU、ROCm、x86/arm），先对产物做 smoke。

路线图（当时）：自动选测、CI 从 1–2 小时打到 30 分钟内、真正的单元测试而不是每次拉起整台 server、修错的 exit code、自动隔离 flaky、坏机器自动踢出、覆盖率。Slack `#sig-ci`。

读这一篇，是为了看见评论里那句「去读 NVIDIA 和 vLLM 的测试文档」在开源这边对应什么：绿灯、夜间秒表、发布门。数字会过时，三层门不会。
