---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Micro-agent：一个 model 名后面的有界协作

英文对照：[en/vllm/blog/serving/semantic-router-micro-agent.md](../../../../en/vllm/blog/serving/semantic-router-micro-agent.md)  
原文：https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models  
2026-06-29。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。MoM 专章：[mom](semantic-router-mom.md)。AMD 现场池：[mom-amd](semantic-router-mom-amd.md)。Fusion 原语：[fusion](semantic-router-fusion.md)。合同发版：[themis](semantic-router-themis.md)。Session 层：[session](semantic-router-session.md)。不要和引擎里的 [Router](router.md) 混。分数是他们 closed/hybrid 配方的 scorecard，不是「每个请求都该上全套闭源模型」。

同目录还有：[athena](semantic-router-athena.md)、[amd](semantic-router-amd.md)、[vision](semantic-router-vision.md)、[modular](semantic-router-modular.md)。

人人盯下一只 frontier checkpoint。这篇把赌注放在它**前面那一层**。Router 已经能砍成本（何时值得上 frontier）、把安全做成可执行策略（敏感域走更严的模型 / 过滤 / 审查）、协调 cloud 和 edge。下一份工作：

> A router can make the model better.

不是改权重。不是让每个应用自己搭一套 agent 图。是把 **一次 model API 调用** 收成 serving 层里的有界协作。

本地图（原文版权仍归原站；学习对照用）：

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

**Figure 1.** Router 从选模型，走到造能力。

[Sakana Fugu](https://sakana.ai/fugu/) 被当成把「一只 model 可以是表面，后面是团队」做成商品的例子。点名的研究语言：[Fugu technical report](https://arxiv.org/abs/2606.21228)、[Conductor](https://arxiv.org/abs/2512.04388)、[Trinity](https://arxiv.org/abs/2512.04695)。vLLM-SR 把抽象放在别处：协作该是 **开放的 serving 原语**，不是一只托管端点，也不是应用侧的一张图。

用户仍调一只模型：

```json
{
  "model": "vllm-sr/auto",
  "messages": [{"role": "user", "content": "..."}]
}
```

这个身份后面，router 可以选配方、扇出、收 quorum、核分歧、合成、修输出合同，再交回一条 OpenAI 兼容响应。重点不是把复杂度摊开。重点是让协作 **摸起来像一只模型**。

## Looper 就是 runtime

请求仍以普通 chat completion 进来。信号 → projection（任务形 / 风险带）→ 决策 → 算法。算法可以是普通单模路由，也可以是 **looper** 路由。

这篇点名的主 looper：

- **Confidence**：顺序升级。先便宜；分数太低才升级。
- **Ratings**：有界扇出。硬 `max_concurrent`；按 rating 加权。
- **ReMoM**：反复的 mixture-of-model 推理。广度采样、minimum-success quorum、合成轮。
- **Fusion**：panel–judge–final。独立答案当证据。专篇：[fusion](semantic-router-fusion.md)。
- **Workflows**：micro-agent 工作流 runtime。静态角色或动态 planner；有界 worker 步；再合成。

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

**Figure 2.** Looper 算法跑在 router 里；对外仍是一个 model 名。

Looper 不是口号「再问几只模型」。它是一只小 runtime：**budget、topology、trace、failure policy**。

### Confidence：升级费只花在难例上

先小/便宜。够自信就停。置信可以来自 token 级 log probability、logprob margin、hybrid score、self-verification，或 AutoMix 式 entailment verifier。

过阈值 → 立刻返回。太低 → 下一只候选。升级是显式路由策略：阈值、失败行为、停止条件，看得见、拧得动。

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

**Figure 3.** Confidence 把升级收成可度量的停止策略。

### Ratings：硬帽下的并行质量

并行拉几只候选，但不超过配置的 `max_concurrent`。收成功的、按 rating 聚合、按路由策略处理失败。他们点的用途：A/B 评测、ensemble、已经有 per-candidate 质量信号的路由。

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

**Figure 4.** Ratings 把多候选执行封在帽里，并且按 rating 加权。

### ReMoM：有合同的广度

推理方差大，但答案格式必须活过协作。扇出推理尝试，等 minimum-success quorum，再让合成模型把证据并进规定的输出合同。

合成失败、前面 worker 已有合法证据：退回最好的合法证据；仍是正常响应，不是 API error。

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

**Figure 5.** ReMoM 把广度、quorum、合成、回退当成 serving 时的旋钮。

### Fusion：把分歧当信号

有时有用的对象不是平均答案，是 **分歧的结构**。独立 panel 答案当证据。Judge 看见同意、矛盾、独特洞察；finalizer 交回一答，trace 收在 API 后面。

适合有竞争路径的时候：难的选择题、长文专家判断、单次自信回答容易脆的精确答案。策略细节：[fusion](semantic-router-fusion.md)。

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

**Figure 6.** Fusion 不藏分歧。它把分歧收成证据。

### Workflows：预算下的角色

最像 agent 的一种，也最需要边界。Planner 只能选 **允许的** worker 模型。Plan 要校验。步数、并行、超时、错误策略都有上限。最终响应仍要满足输出合同。

页上的 SWE 例：planner、patcher、verifier、finalizer——应用不用自己养一套 agent 栈。能力在，治理也在基础设施里。

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

**Figure 7.** Workflows 给 router 一套有界角色，不是无界自治 agent。

### Auto recipes：一个 model 名，多种 loop

对外仍是 `vllm-sr/auto`。对内，信号和 projection 选 loop。难度、风险、合同压力、延迟、成本是路由事实，不是 prompt 里的注释。它们可以选 Confidence、Ratings、ReMoM、Fusion、Workflows，或回退。

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

**Figure 8.** Auto recipe 让信号选协作形态，同时保住一个 model 身份。

他们咬住的差别：「agent 当应用逻辑」对「micro-agent 当 serving runtime」。Budget、policy、topology、trace、失败模式，归 router。

## 配方赢过一只万能 loop

评测课不是某算法永远赢。相反：

> The best loop is task-shaped.

- GPQA-Diamond 要保住严格的选择题答案
- LiveCodeBench 要能跑的代码和隐藏测试稳健
- Humanity's Last Exam 要解分歧、精确答案格式
- SWE 类任务要 planner / patcher / verifier / finalizer

所以 `vllm-sr/auto` 不该等于「永远跑最大 loop」。它该等于：选贴合这个任务的配方。

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

**Figure 9.** 信号和 projection 选 benchmark 形的协作。

他们配方里写死的形状：

- GPQA-Diamond：难的科学选择题 → ReMoM，严格保 `ANSWER: X`
- LiveCodeBench：看约束、starter code、标准输入、float tolerance、timeout 风险、hidden-test 风险，再选 code 形 loop
- HLE：形式推理、分歧风险、长上下文、精确答案压力 → 更深的 ReMoM、更小的 Fusion，或回退

配方不只是 prompt。它还定义模型池、角色、reasoning effort、concurrency、quorum、timeout、合成模型、回退策略、输出合同、可观测标签。

## Scorecard 是证明，不是整部故事

闭源配方，三道难题。**VSR Closed** = 只用闭源 backend。**VSR Hybrid** = 开源加闭源；配方需要高风险 judging / repair / synthesis / fallback 时才上更强闭源。

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)

**Figure 10.** VSR Closed 和 VSR Hybrid 在 LiveCodeBench、GPQA-Diamond、Humanity's Last Exam 上的 scorecard。

| Benchmark | VSR scorecard row | Score | Reference rows |
| --- | --- | ---: | --- |
| LiveCodeBench, January-April 2025 | VSR Closed | 92.6 | Fugu Ultra 92.0, Fugu 90.3, GPT-5.5 90.7, Opus 4.8 90.3 |
| GPQA-Diamond | VSR Closed | 96.0 | Fugu Ultra 95.5, Fugu 95.5, Gemini 3.1 Pro 94.3, GPT-5.5 93.6 |
| Humanity's Last Exam | VSR Closed | 50.0 | Fugu Ultra 50.0, Fugu 48.5, Gemini 3.1 Pro 45.0 |
| Humanity's Last Exam | VSR Hybrid | 47.1 | GLM-5.2 40.5, Qwen3.7 Max 41.4, GPT-5.5 41.4 |

仔细读。不是主张每条请求都该上全套闭源。主张是：router 自己管的协作，可以造出比底下单次调用更强的 **model 身份**。压过或打平 frontier 单模基线，同时保住一个 API 面。

页上的产品形状：

- 用户看见一个 model 名
- 运维管配方
- 系统可以变好，客户端集成不用改
- 开源和闭源在同一套 serving 抽象下参与

## 这对 model serving 意味着什么

旧栈：收下 model 名，转给 backend。下一栈要问：

- 这条请求有什么证据？
- 质量、成本、延迟、安全落在哪条带？
- 一只模型够不够？
- 不够，该跑哪种协作？
- 哪份答案合同必须保住？
- 一家 provider 慢了或错了怎么办？
- 怎么交回一条干净响应，同时留住完整 trace？

这是基础设施，不是应用胶水。Micro-agent 该住在 router 里，因为 router 已经有 alias、provider 策略、凭证、成本元数据、信号、决策、重试、超时、trace、OpenAI 兼容的响应语义。

## 收束

「Frontier model」开始有两层意思：一只 checkpoint，和一条 **系统边界**。编排浪潮把方向照亮。vLLM-SR 的赌注：在 serving 层做成可编程、可观测、开放。

下一场竞赛仍有更好的模型。也有更好的 router：何时省钱、何时执法、何时留在 edge、何时上云、何时把一条请求收成一支有纪律的小队。

## 致谢

研究合作：[MBZUAI](https://mbzuai.ac.ae/)、[McGill University](https://www.mcgill.ca/)、[Mila](https://mila.quebec/)、[Agentic Intelligence Lab](https://agentic-in.ai/)，尤其 [Prof. Xue Liu](https://www.linkedin.com/in/xueliu) 和 [Dr. Bowei He](https://www.linkedin.com/in/bowei-he-8a9450199/)。

个人贡献：[Huamin Chen](https://www.linkedin.com/in/huaminchen/)、[Yincheng Ren](https://www.linkedin.com/in/yincheng-ren/)。

AMD GPU 评测支持：[Andy Luo](https://www.linkedin.com/in/andyluo77/)、[Haichen Zhang](https://www.linkedin.com/in/haichen-zhang-9010b6382/)。
