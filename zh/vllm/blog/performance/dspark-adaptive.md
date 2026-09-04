---
source: https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# DSpark 自适应验收：按信心和负载改预算

英文对照：[en/vllm/blog/performance/dspark-adaptive.md](../../../../en/vllm/blog/performance/dspark-adaptive.md)  
原文：https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification  
2026-08-14。署名 **vLLM Team**。学习笔记。进树是 [PR #47808](https://github.com/vllm-project/vllm/pull/47808)，开关 `enable_adaptive_verification`。页上的演示：DeepSeek-V4-Pro-0813，**TP=8**，**8×B300**（SM100）。并行草稿家族：[parallel-drafting](parallel-drafting.md)。验收数学：[spec-decode](spec-decode.md)。DSpark 论文：[arXiv 2607.05147](https://arxiv.org/abs/2607.05147)。

投机解码用更多计算换更少 Decode 步。Batch size **1** 时这笔买卖很香：GPU 还 memory-bound，算力有空，多出来的 draft token 几乎免费。到 batch size **256** 就细了。Draft token 和真 token 抢同一份算力；每个被拒的都是浪费；拒得够多，吞吐会掉。

**原文 TL;DR。** [DSpark](https://arxiv.org/abs/2607.05147) 的 confidence head 给每个 draft token 打「活过验收」的分。不必为一次部署选定长投机长度，vLLM 可以 **每一步** 决定这一轮验多少。打开自适应验收（`num_speculative_tokens: 7`），投机解码一直到 **concurrency 256** 仍有好处，低并发时又能保住长草稿的那截。这样就少拧 `num_speculative_tokens`。原文把 DSpark 说成更容易「默认打开」的那种赢。

改的是 **每步验多少**，不是草稿结构。

## The problem

按位置的接受率掉得很快。DeepSeek-V4-Pro-0813 上，**7 token** 一块里 **最后一个** draft 活下来不到 **10%**，第一个则超过 **70%**。那个低概率 token 仍要在每一轮 verification batch 里占一个槽。GPU 还 memory-bound 时这个槽几乎免费，值得赌；一旦饱和，这场赌就有真吞吐成本。

交叉点会随负载、随工作负载自己的接受率移动，所以 **没有** 一个静态 `num_speculative_tokens` 能在所有并发上都最优。DSpark 的办法：一份自适应草稿 **预算**，同时看系统负载，以及 DSpark head 有多确信 target 会接受每个 draft token。

## Scheduling the budget

DSpark 每轮草稿一块 *k* 个 token（`num_speculative_tokens`），用学出来的 confidence head 给每个位置打分。调度器把分数收成存活概率：沿每条请求的前缀连乘

$$
S(r, i) = \prod_{j \le i} \mathrm{confidence}(r, j)
$$

存活只随位置 *i* 往下掉。给定草稿 token 预算 *B*，把它分给最可能的草稿序列，就是对存活分数做一次全局 top-*B*。结果自然是每条请求草稿的 **连续前缀**，不必再加约束。槽在请求 **之间** 竞争：一条很有把握的请求的位置 5，可以压过一条没把握的请求的位置 1。

![fig1 policy](../../../../assets/vllm/blog/performance/dspark-adaptive/01-fig1-policy.svg)

**Figure 1。** 同一 batch，两种策略。固定验收为全部 **21** 个槽付钱，包括存活近零的。自适应验收只验最好的 **B=11**。

*B* 最大化「单位 step 时间里期望 token 数」：

$$
B^* = \arg\max_B \frac{N_\mathrm{sampling} + \sum_{j < B} S_\mathrm{sorted}[j]}{\mathrm{draft\_cost}[\mathrm{num\_reqs}] + \mathrm{verify\_cost}[T + B]}
$$

分子：每个会采样的请求一个 bonus token，再加上 *B* 个最好草稿槽的存活。*N*<sub>sampling</sub> 只数 **这一步真正会 sample** 的请求——还在 chunked Prefill 里的请求贡献为零。分母：一张 profile 出来的成本表，按下标看这一步的 token 数。*T* 是已经排上、但不是 draft 的 token，所以 *T* + *B* 是整步。两边都是数组；选择就是对前缀和做 `np.argmax`。成本单位是 **microseconds**。

定 *B* 在 **CPU** 上跑，此时 GPU 还在做上一步；用的是双缓冲里 **旧一步** 的 confidence。把这 *B* 个槽分给各条请求则在 **GPU** 上、对着 **当前** confidence。选择写成 PyTorch，经 `torch.compile` 降到 Triton，**不读回 host**。

## Varlen decode CUDA graphs

变长验收需要 **varlen decode CUDA graphs**。这要求 attention kernel 支持：sparse MLA 天然 varlen（每个 query token 自己的 top-k）。DeepSeek 在 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 开源了 varlen indexer kernel，作为 [PR #47808](https://github.com/vllm-project/vllm/pull/47808) 的一部分接进来。

Decode graph 按 `num_reqs = min(num_tokens, max_num_seqs)` 捕获，并承诺 `max_query_len = num_speculative_tokens + 1`。一张图就能伺候每条请求 **1** 到 `num_speculative_tokens + 1` 个 token 的任意搭配。

## The cost model

预算公式要除以一步的成本，所以这成本必须查找便宜、还得像真的。启动时引擎对一套固定形状跑 **dummy step**（CUDA graph 形状，再加上比 max cudagraph size 稍大的几个），每个形状取 **五次的中位数**。变成两张扁查找表：

- verification 表，按下标是 **token 数**
- drafter 表，按下标是 **请求数**（验多少 token 都不改草稿成本）

两张相加。

![fig2 costcurve](../../../../assets/vllm/blog/performance/dspark-adaptive/02-fig2-costcurve.svg)

**Figure 2。** 一次真实启动 profile 里的两张成本表；成本是 5 次采样的中位数。原文没有把表项写成数字。

落在已捕获的 CUDA graph 里，成本是 **台阶** 不是直线：cudagraph padding 意味着 **121** 个 token 的 batch 会跑 **128** token 那张图，并且（大体上）为全部 128 付钱。过了 capture 上限，台阶结束，成本才连续。离开 cudagraph 区有一截明显的跳变；跳得够狠，预算算法会被 **强烈推回 cudagraph 区里面**。

Profile 噪声：曲线被强制 **单调**。真实 step 成本完全可能随 batch 变大而下降（kernel tile size），所以单调是平滑，不是声称硬件单调。Dummy step 对着合成 KV 上下文，默认 **8192** token，可用 `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` 调。

## Results

页上的配置：**DeepSeek-V4-Pro-0813**，**TP=8**，**8×B300**（SM100），expert parallel，FP8 KV cache，`max_model_len` **16384**，`max_cudagraph_capture_size` **4096**，vLLM `main` 在 `73b8394`。基准：**880** 条 prompt，temperature **1.0**，最多 **2048** 个输出 token，并发从 **1 扫到 256**。

![fig3 pareto](../../../../assets/vllm/blog/performance/dspark-adaptive/03-fig3-pareto.svg)

**Figure 3。** 不同投机方案的吞吐对 interactivity。自适应验收整段都贴着 Pareto 前沿。

原文判断：自适应验收在 **整段扫描上都贴着 Pareto 曲线的边**，两端都明显好过不开投机。读图的方式：低并发像长固定块，高并发像短固定块——两头都要，又不必预先知道负载长什么样。TL;DR 把同一句话说到 **c=256**。原文 **没有** 把各点的 tok/s 或 ITL 写成表；他们画的是结果 JSON 里的 `output_throughput`。

## Limitations

- **FULL** varlen decode graph 要求 `AttentionCGSupport.ALWAYS`。DSV4 的 sparse-MLA、sparse-SWA、indexer 后端在 **SM100** 上会报这个。别处自适应验收会在 **启动时拒绝**，不会退回 PIECEWISE。
- `--enforce-eager`（step 成本是从捕获的 graph 上 profile 的）、**LoRA**、**pipeline parallelism** 当时都不支持。
- 打开自适应验收时拒绝输出 **logprobs**：验收会在前向之后 **压紧 logits**。

## Appendix: reproducing

下面命令走 [PR #47808](https://github.com/vllm-project/vllm/pull/47808)，当时已合进 vLLM `main`。上面的数字测在 `73b8394`。

**Server**（所有测量；消融只改 `--speculative-config`）：

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 --max-model-len 16384 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.8 \
  --compilation-config '{"max_cudagraph_capture_size":4096}' \
  --speculative-config '{"method":"dspark","attention_backend":"FLASH_ATTN","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}'
```

Draft 默认就是 **target checkpoint**，所以 `"model"` 可以省略。`--kv-cache-dtype fp8` **必须有**：`fp8_ds_mla` 布局拒掉其他 KV dtype。`--max-num-seqs` 也要紧：默认是 **128**，会把 batch 卡在并发扫描的顶端以下。他们把 `max_cudagraph_capture_size` 提到 `(num_speculative_tokens + 1) * max_num_seq`，好让每一轮 verification batch 都落在 cudagraph 里。更大的 capture 更吃显存，所以 `--gpu-memory-utilization 0.8`；默认值会在 **capture 时 OOM**。

消融：

- 固定 k：`"enable_adaptive_verification": false`，`"num_speculative_tokens": k`，且 k ≥ `dspark_block_size`（这个 checkpoint 上是 **5**）
- 不开投机：去掉 `--speculative-config`

**吞吐扫描**，每个并发 `c ∈ {1, 16, 32, 64, 128, 256}`，先做一轮 warmup（`--speed-bench-output-len 256 --num-prompts 64 --max-concurrency 32`）：

```bash
MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
for c in 256 128 64 32 16 1; do
  n=880; [ "$c" = 1 ] && n=240
  vllm bench serve \
    --backend openai-chat --base-url http://127.0.0.1:8000 \
    --endpoint /v1/chat/completions --model "$MODEL" \
    --tokenizer "$MODEL" --tokenizer-mode deepseek_v4 \
    --dataset-name speed_bench --dataset-path <speed-bench-dir> \
    --speed-bench-dataset-subset qualitative --speed-bench-output-len 2048 \
    --num-prompts $n --max-concurrency $c --request-rate inf \
    --skip-chat-template --disable-shuffle --temperature 1.0 --seed 0 \
    --save-result --result-filename adaptive_on_c${c}.json
done
```

`--disable-shuffle` 加上固定 prompt 集：每条臂拿到相同顺序的相同 prompt。结果 JSON 里的 `output_throughput` 就是 Figure 3 上的 tok/s。`--speed-bench-output-len` 是 **上限** 不是目标——请求在 EOS 停，真实平均远低于 2048。**c=1** 用 `n=240` 条 prompt，其余 **880**。

## Acknowledgments

Lucas Wilkinson（Red Hat）和 Benjamin Chislett（NVIDIA）。感谢 [DSpark](https://arxiv.org/abs/2607.05147) 作者提供草稿算法和 confidence head，以及 DeepSeek 的 DeepSeek-V4 checkpoint。
