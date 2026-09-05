---
source: https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Micro-agent：一个 model 名后面的有界协作

英文对照：[en/vllm/blog/serving/semantic-router-micro-agent.md](../../../../en/vllm/blog/serving/semantic-router-micro-agent.md)  
原文：https://vllm.ai/blog/2026-06-29-micro-agent-frontier-models  
2026-06-29。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。面板–法官：[fusion](semantic-router-fusion.md)。MoM 当系统：[mom](semantic-router-mom.md)。会话连续：[session](semantic-router-session.md)。可运营合同：[themis](semantic-router-themis.md)。不要和引擎里的 [Router](router.md) 混。分数是他们 closed/hybrid 配方的 scorecard，不是「每个请求都该上全套闭源模型」。

同目录还有：[athena](semantic-router-athena.md)、[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[modular](semantic-router-modular.md)、[vision](semantic-router-vision.md)、[halugate](halugate.md)。

人人盯下一只前沿 checkpoint。更有意思的一层可能坐在它前面。

Router 正在变成 AI 推理的控制面。第一岗很实际：对的请求送到对的模型。生产已经不是一只模型的世界，这一岗已经要紧。

Router 可以省：何时配得上前沿模型，何时开源或本地就够。可以把安全做成可执行：敏感域送到更严的模型、更严的过滤、更强的复核。可以协调云和边：私密或低延迟意图留在本地，更难的活再升到云。

这些都重要。页上下一岗更有意思：

> Router 能把模型变好。

不是改权重。不是让每只应用自己搭一套 agent 图。是把 **一次模型 API 调用** 变成 serving 层里的 **有界协作**。

本地图（原文版权仍归原站；学习对照用）：

![router capability layer](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/01-router-capability-layer.png)

**Figure 1.** 从选模型到造能力。

[Sakana Fugu](https://sakana.ai/fugu/) 把一个简单想法做成商业产品：「模型」可以是一层皮，皮后面是团队。有用的语言：[Fugu 技术报告](https://arxiv.org/abs/2606.21228)、[Conductor](https://arxiv.org/abs/2512.04388)、[Trinity](https://arxiv.org/abs/2512.04695)。vLLM-SR 的赌注在 **抽象落在哪**：协作不该只住在一家托管端点里，也不该只住在应用自己的 agent 图里。它该变成 **开放的 serving 原语**。

用户仍调一只模型：

```json
{
  "model": "vllm-sr/auto",
  "messages": [{"role": "user", "content": "..."}]
}
```

这个稳定身份后面，router 可以选配方、散到 worker、收 quorum、核分歧、合成、修输出合同，收回一条普通的 OpenAI 兼容响应。目的不是把复杂摊开。协作该 **摸起来像一只模型**。

## Looper 就是 runtime

在 vLLM Semantic Router 里，looper 是有界 micro-agent 的执行 runtime。

请求当普通 chat completion 进来。信号 → projection（任务形状或风险带）→ 决策 → 算法。算法可以是单模路由，也可以是 looper 路由。

今天的主模式：

- **Confidence**——顺序升级：便宜先试，量置信，分太低才升级
- **Ratings**——`max_concurrent` 封顶的并行；按评分加权聚合
- **ReMoM**——重复的 mixture-of-model 推理：广度采样，等成功 quorum，再合成一轮
- **Fusion**——面板 → 法官 → 收束（[fusion](semantic-router-fusion.md)）
- **Workflows**——micro-agent 工作流 runtime：静态角色或动态 planner；有界 worker 步；再合成

![looper micro agents](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/02-looper-micro-agents.png)

**Figure 2.** Looper 算法在 router 里跑；模型 API 面不动。

实现细节要紧。Looper 不是口号「再问几只模型」。它是一只小 runtime，带着 **预算、拓扑、迹、失败政策**。

### Confidence：升级税只花在难的上

成本感知的环。先小/便宜，再问这答够不够自信可以停。置信可以来自 token 级 logprob、logprob margin、混合分、自核、或 AutoMix 那种蕴含核验器。

分过阈值，立刻返回。分太低，升到下一只候选。要紧的不是「有升级」，是升级变成显式的 router 政策：阈值、失败行为、停止条件，看得见、拧得动。

![confidence loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/03-confidence-loop.png)

**Figure 3.** 升级变成可量的停止政策。

### Ratings：硬帽下的并行质量

受控的 ensemble 环。几只候选并行，但只到配置的 `max_concurrent`。一条路由想从多只模型视角受益，又不许每条请求都变成无界扇出。

收成功的、按评分聚合、失败按路由政策处理。适合 A/B 评测、ensemble、运营已经有每候选质量信号的路。

![ratings loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/04-ratings-loop.png)

**Figure 4.** 多候选执行有界，并且认得评分。

### ReMoM：带着合同的广度

推理方差大，但答案格式必须活下来。散出多次尝试，等最低成功 quorum，合成模型把证据收进规定输出合同。

合成失败、前面 worker 却交出过合法证据：不必整段塌成 API 错误。回退到最好的合法证据，仍交一条普通响应。

![remom loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/05-remom-loop.png)

**Figure 5.** 广度、quorum、合成、回退，都是 serving 时的控制。

### Fusion：分歧当信号

另一套赌注。有时有用的不是平均答案，是 **分歧的结构**。独立面板答案变成证据。法官看见共识、矛盾、独到洞察；收束器交一答，迹收在 API 后面。

难的多选、长文专家判断、一只自信答案就脆的精确答题。

![fusion loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/06-fusion-loop.png)

**Figure 6.** 分歧是证据，不是要藏的东西。

### Workflows：预算里的角色

最像 agent，因此边界最严。Planner 只能挑 **允许的** worker 模型。计划要校验。步数、并行、超时、错误政策都有顶。最终响应仍要满足输出合同。

SWE 向：planner、patcher、verifier、finalizer——应用不必自己拥有一套 agent 栈。对生产 serving，这个分别要紧：环有力量，但仍受基础设施治理。

![workflows loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/07-workflows-loop.png)

**Figure 7.** 有界角色系统，不是无界自治 agent。

### Auto 配方：一个名字，许多环

对外仍是 `vllm-sr/auto`。对内，信号和 projection 选环。难度、风险、合同压力、延迟、成本不是 prompt 里的注释。它们是路由事实，可以选 Confidence、Ratings、ReMoM、Fusion、Workflows，或一条回退路。

![auto recipe loop](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/08-auto-recipe-loop.png)

**Figure 8.** 信号挑协作模式；模型身份仍是一个。

差别：「agent 当应用逻辑」对「micro-agent 当 serving runtime」。Router 拥有预算、政策、拓扑、迹、失败模式。

## 配方打败一只万能环

页上评测课：不是某一算法永远赢。正好相反：

> 最好的环是任务形状的。

GPQA-Diamond 要严格保住多选。LiveCodeBench 要能跑的代码和隐藏测试鲁棒。Humanity’s Last Exam 要化解分歧、精确格式。SWE 向要 planner / patcher / verifier / finalizer。

所以 `vllm-sr/auto` 不该等于「永远跑最大的环」。该等于：选配得上这个任务的配方。

![benchmark shaped recipes](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/09-benchmark-shaped-recipes.png)

**Figure 9.** 用信号和 projection 选基准形状的协作。

他们的配方把形状写死：

- GPQA-Diamond：难的科学多选 → ReMoM，严格保住 `ANSWER: X`
- LiveCodeBench：约束、starter code、标准输入、浮点容差、超时风险、隐藏测试风险 → 代码形状的环
- HLE：形式推理、分歧风险、长上下文、精确答案压力 → 更深 ReMoM、更小 Fusion、或回退

Prompt 只是一块。配方还定义模型池、角色、reasoning effort、并发、quorum、超时、合成模型、回退、输出合同、可观测标签。这就是为什么 router 侧协作不只是改 prompt。

## Scorecard 是证明，不是全故事

闭源配方过三道硬基准。数字有用，因为想法不只审美。

![three eval scorecard](../../../../assets/vllm/blog/serving/semantic-router-micro-agent/10-three-eval-scorecard.png)

**Figure 10.** LiveCodeBench、GPQA-Diamond、HLE 上的 VSR Closed / Hybrid。

**VSR Closed** = 只用闭源 backend。**VSR Hybrid** = 开闭混用；判 / 修 / 合成 / 回退风险更高处用更强闭源。

| Benchmark | VSR scorecard row | Score | 页上对照行 |
| --- | --- | ---: | --- |
| LiveCodeBench, 2025-01–04 | VSR Closed | 92.6 | Fugu Ultra 92.0, Fugu 90.3, GPT-5.5 90.7, Opus 4.8 90.3 |
| GPQA-Diamond | VSR Closed | 96.0 | Fugu Ultra 95.5, Fugu 95.5, Gemini 3.1 Pro 94.3, GPT-5.5 93.6 |
| Humanity’s Last Exam | VSR Closed | 50.0 | Fugu Ultra 50.0, Fugu 48.5, Gemini 3.1 Pro 45.0 |
| Humanity’s Last Exam | VSR Hybrid | 47.1 | GLM-5.2 40.5, Qwen3.7 Max 41.4, GPT-5.5 41.4 |

小心读。不是主张每条请求都该动用全部闭源模型。那会是错的产品。主张：router 自管的协作可以造出比底下单次调用 **更强的模型身份**，打平或压过前沿单模基线，同时保住一个 API。

产品形状：

- 用户看见一个模型名
- 运营管配方
- 系统可以变好而不改客户端集成
- 开源和闭源在同一套 serving 抽象下参与

## 对模型 serving 意味着什么

旧栈是被动的：收下模型名，送到 backend。

下一栈是主动的。它要问：

- 这条请求有什么证据？
- 落在哪条质量 / 成本 / 延迟 / 安全带？
- 一只模型够不够？
- 不够走哪套协作？
- 哪份答案合同必须保住？
- provider 慢或错怎么办？
- 怎么交出一条干净响应，又把整段迹留下？

这是基础设施，不是应用胶水。Micro-agent 该住在 router 里，因为 router 已经拥有 micro-agent 需要的东西：别名、provider 政策、凭证、成本 metadata、信号、决策、重试、超时、迹、OpenAI 兼容响应语义。

## 收束

「前沿模型」开始指两件事。一只是 checkpoint。另一只是 **系统边界**。

编排浪潮把方向照亮。vLLM-SR 赌：这能力该在 serving 层可编程、可观测、开放。

下一场竞赛仍会有更好的模型。也会有更好的 router：何时省、何时执法、何时留在边上、何时上云、何时把一条请求收成一支有纪律的小队。

这是 Model API 里面 micro-agent 的承诺。

## 致谢

研究合作：[MBZUAI](https://mbzuai.ac.ae/)、[McGill University](https://www.mcgill.ca/)、[Mila](https://mila.quebec/)、[Agentic Intelligence Lab](https://agentic-in.ai/)；点名 [Prof. Xue Liu](https://www.linkedin.com/in/xueliu)、[Dr. Bowei He](https://www.linkedin.com/in/bowei-he-8a9450199/)，谈 router 侧模型协作。

个人贡献者：[Huamin Chen](https://www.linkedin.com/in/huaminchen/)、[Yincheng Ren](https://www.linkedin.com/in/yincheng-ren/)。

AMD GPU 评测：[Andy Luo](https://www.linkedin.com/in/andyluo77/)、[Haichen Zhang](https://www.linkedin.com/in/haichen-zhang-9010b6382/)。
