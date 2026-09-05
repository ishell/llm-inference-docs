---
source: https://vllm.ai/blog/2025-01-14-struct-decode-intro
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Structured decoding：给会说话的模型一套不会说错格式的栅栏

英文对照：[en/vllm/blog/performance/struct-decode.md](../../../../en/vllm/blog/performance/struct-decode.md)  
原文：https://vllm.ai/blog/2025-01-14-struct-decode-intro

2025-01-14。客座：BentoML 与 Red Hat。学习译文，不是官方译本。文中 V1「即将发布」是**当时时态**——那些路线图后来大部分进了 V1。JSON mode、function calling、agent 工具参数，底下往往就是这件事。落地 [PR #10785](https://github.com/vllm-project/vllm/pull/10785)。当时的路线图 [#8779](https://github.com/vllm-project/vllm/issues/8779)。

**TL/DR：**

- Structured decoding 管的是输出**格式**，采样仍是采样
- 当时 vLLM 同时接 [outlines](https://github.com/dottxt-ai/outlines) 和 [XGrammar](https://github.com/mlc-ai/xgrammar)
- 刚接上的 XGrammar：负载下 TPOT 最多大约 **5×**
- 当时计划的 V1：性能，以及 **scheduler 级** mask 广播，好让混合 batch 里的普通人不受阻

[vLLM](https://blog.vllm.ai/2023/06/20/vllm.html) 是高吞吐推理引擎。这篇顺着语言模型的注释史，写到当时 vLLM 里 structured decoding 的现状、刚落地的 [XGrammar](https://github.com/vllm-project/vllm/pull/10785)，以及一份当时还 tentative 的改进路线图。

作者还请人用哲学的方式读：structured decoding 是对「模型吐出来的东西」怎么负责的一次改口，也是搭复杂 agent 系统的一块砖。

更多见当时的 [vLLM 文档](https://docs.vllm.ai/en/latest/)。

本地图（原文版权仍归原站；学习对照用）：

![shogoth gpt](../../../../assets/vllm/blog/performance/struct-decode/01-shogoth-gpt.png)

**图注（原文）。** Shogoth as GPTs。某种意义上，RLHF 或任何 post-training，都是往大复合 AI 系统里注射规则（一次 GOFAI 动作）。

![mermaid intro](../../../../assets/vllm/blog/performance/struct-decode/02-mermaid-intro.svg)

**图注（原文）。** Structured decoding 的顶层视图。

![constrained json fsm](../../../../assets/vllm/blog/performance/struct-decode/03-constrained-json-fsm.webp)

**图注（原文）。** Constrained JSON 的 FSM。转引自 [LMSys, 2024](https://lmsys.org/blog/2024-02-05-compressed-fsm/)。

![vllm new xgrammar](../../../../assets/vllm/blog/performance/struct-decode/04-vllm-new-xgrammar.png)

![vllm xgrammar decode time per output token](../../../../assets/vllm/blog/performance/struct-decode/05-vllm-xgrammar-decode-time-per-output-token.png)

**图注（原文）。** XGrammar 对 Outlines；每个输出 token 的 Decode 时间。courtesy of Michael Goin (Red Hat)。

## Language models: A brief historical context

1950 年，Alan Turing 提出：一台高速数字计算机，若用规则编好，可以表现出智能的涌现行为（Turing, 1950）。后来大致两条路：

1. **Good Old-Fashioned AI (GOFAI)：** 1950 年代很快长出一套范式：专家系统，想复刻人类专家的决策（符号推理）。Haugeland 称之为 Good Old-Fashioned AI（Haugeland, 1997）。语义表征撑不到通用任务，经费跟着塌——也就是「AI 冬天」（Hendler, 2008）。脚注里：Allen Newell 与 Herbert Simon 在 RAND 先证明计算机能模拟智能的重要侧面。医学上更有名的是 Stanford 1970 年代的 MYCIN，给血液感染开建议（Shortliffe, 1974），并用 “rule traces” 把推理讲给人听。

2. **New-Fangled AI (NFAI)：** 同时，Donald Norman 的 Parallel Distributed Processing 小组（Rumelhart et al., 1986）在 Rosenblatt 的 perceptron（Rosenblatt, 1958）上加 **hidden layers**，想从训练里外推出合适的响应。这些联结网络常常建在统计方法上。数据够多，再加上 Moore’s Law 带来的算力，联结网络后来在研究和生产里全面占上风——尤其是做 **text generation** 的 **decoder-only** transformer。所以多数现代 transformer 变体被看成 **NFAI**。

脚注里的统计线：1990 年代 IBM alignment models 做机器翻译；2001 年前后 Bag of words 变体在 0.3B token 上当时算 SOTA（Mikolov et al., 2013）——统计模型比符号系统更能抓住大语料的通式。2017 年 “Attention Is All You Need”（Vaswani et al., 2023）把 attention（Bahdanau et al., 2016）收成 Transformer；OpenAI 的 scaling law（Kaplan et al., 2020）把竞赛推到基础语言模型上。Attention 之前，seq-to-seq 靠 RNN / LSTM（Hochreiter & Schmidhuber, 1997）扛长上下文，但梯度难、远期记忆差。Attention 把位置信息编进输入；原文还有 encoder–decoder，今天多数文本生成却是 decoder-only，零样本更好。Transformer 比 LSTM 可扩展、更认硬件——不能指望多叠几块 LSTM 就换来长期记忆。

收束：

- GOFAI **确定**、基于规则：意图写进程序
- NFAI 常被看成黑盒（进：输入 — 出：某种输出），数据驱动，内部表征是网络长出来的

## Why do we need structured decoding?

LLM 擅长这条启发式：给它一坨文本，它会吐出它认为最可能的后续 token。例如一篇维基，它该写出像那篇文章剩下部分的东西。

前提是：输入 prompt 得干净、结构清楚，围着用户真正要的问题。换一种说法：你要特定格式时，LLM 可以很不听话。请它写 JSON——没有栅栏，它可能吐出读得通、却坏掉 JSON 规格的文本。

Few-shot（「给我这样的 JSON……」）仍是采样，非法 JSON 仍然被允许。为 JSON 单独微调，训练、盯进度、评测都贵，不是人人付得起。

Structured decoding 就在这里：让 LLM 按想要的结构吐字，同时**保住系统的非确定性**——采样仍是采样。

OpenAI 一类公司看见了这件事，做了 [JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode) 来约束输出格式。做过 agent、function calling、编程助手，多半已经在用 structured decoding，只是没看见名字。

**Structured / constrained / guided decoding** 是同一件事：用一份格式，让模型**按结构去采样**。

> Guided decoding 对 LLM，就像校验对 API——保证吐出来的东西对得上你的预期。结构完整了，开发者才敢把 LLM 接进应用。

Dottxt 还写过：有时它甚至能[改善](https://blog.dottxt.co/coalescence.html)原生 Decode 性能。

## Structured decoding and vLLM

简单说，structured decoding 给 LLM 一份「模板」。用户给 schema，去「影响」模型的输出，让它守住结构。

技术上，推理引擎可以改下一 token 的概率分布：按 schema 给 token 加 bias（常常是 logit mask）。[outlines](https://github.com/dottxt-ai/outlines) 提出用有限状态机（FSM）做 guided generation（Willard & Louf, 2023）：解码时跟踪当前状态，对非法 token 加 logit bias，滤掉。

在 vLLM 里：把 JSON schema 塞进 **sampling params**（Python SDK 或 HTTP）。

### Previous limitations in vLLM

当时 Outlines backend 几处疼：

1. **Decode 慢。** FSM 按 token 构造，一步只能转一个状态，因此一步只能解一个 token。
2. **组 batch 的瓶颈。** 实现严重依赖 logit processor（当时的 [`outlines_logits_processors.py`](https://github.com/vllm-project/vllm/blob/80c751e7f68ade3d4c6391a0f3fce9ce970ddad0/vllm/model_executor/guided_decoding/outlines_logits_processors.py)），落在采样热路径上。组 batch 时，每条请求编 FSM、同步算 mask，**同一 batch 里所有人**都得等 → TTFT 被拖高，吞吐下降。编 FSM 本身就贵，是 TTFT 的大头。HuggingFace 的 [logits-processor zoo](https://huggingface.co/blog/logits-processor-zoo) 是更一般的阀门。
3. **CFG 模式的性能。** JSON mode 还算快；CFG 慢很多，偶尔还能[把引擎弄崩](https://github.com/vllm-project/vllm/issues/10081)。
4. **高级能力接不上。** [Jump-forward decoding](https://lmsys.org/blog/2024-02-05-compressed-fsm/) 当时做不到：它要一次填好已经确定的 k 个 token，logit processor 只看得见**下一个**。

### Integration with XGrammar

[XGrammar](https://github.com/mlc-ai/xgrammar) 用下推自动机（PDA）做 batch constrained decoding。可以把 PDA 想成「一堆 FSM，每坨是一份 context-free grammar (CFG)」。PDA 能递归，一次可以跳多步状态。语法编译还有额外[优化](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)。

这直接对着**疼点 (1)**：语法编译从 Python 挪到 C，走 `pthread`。也为后来对着**疼点 (4)** 铺路。下面是 XGrammar 对 Outlines 的性能对照（Figure 4–5）：负载下 TPOT 最多大约 **5×**。

V0 架构里，XGrammar 仍是 [logit processor](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/xgrammar_decoding.py)，只是 tokenizer 数据有 cache。成绩令人鼓舞，他们仍觉得还能挖。

当时 XGrammar v0 要对齐全部用例，还缺几块（保持「文中当时」的标签）：

- 还不会非 **GBNF** 语法（vLLM [PR](https://github.com/vllm-project/vllm/pull/10870)）
- 还不会 **regex**
- 还不会带 regex pattern 或数值范围的复杂 JSON（[vLLM #10899](https://github.com/vllm-project/vllm/pull/10899)，上游 [xgrammar #106](https://github.com/mlc-ai/xgrammar/pull/106)）

> vLLM 当时默认有一份基本的 XGrammar。知道它伺候不了这条请求，就回落到 Outlines。
>
> 仓库里还有 lm-format-enforcer。他们测过：某些长上下文测例约束会漏，性能也不如 Outlines 稳。

## Tentative plans for v1

当时 [v1](https://github.com/vllm-project/vllm/issues/8779) 将至，structured decoding 的 tentative 计划：

1. Guided decoding 升到 **scheduler 级**：
   - 调度器认得谁在用 structured decoding，就不该挡住同一 batch 里的普通人（对着疼点 **(2)**）。离开热路径。
   - Jump-forward 也更自然（对着疼点 **(4)**）。
2. Bitmask **在一个进程里算**，再 **broadcast** 给每个 GPU worker，而不是每个 worker 算一遍。
   - 每条 sample、每个走 guided decoding 的请求，广播 mask 的带宽，当时说要仔细量。
3. 给**投机解码**和 **tool-use** 同一套底座：
   - XGrammar 计划接 tool-use，好离开 Python [tool parser](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)。
   - 投机解码的 tree scoring 可以和 jump-forward 共用 API（取决于 scheduler 级 guided decoding）。

_NOTE：有建议欢迎收。当时 Slack：`#feat-structured-output`（入口见原文里的 [vLLM slack](https://www.notion.so/bentoml/slack.vllm.ai)）。_

## Acknowledgements

vLLM 团队、XGrammar 团队，以及 [Aaron Pham (BentoML)](https://github.com/aarnphm)、[Michael Goin (Red Hat)](https://github.com/mgoin)、[Chendi Xue (Intel)](https://github.com/xuechendi)、[Russell Bryant (Red Hat)](https://github.com/russellb) —— 把 XGrammar 带进 vLLM，并继续改进 structured decoding。

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2016). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv preprint arXiv:1409.0473
- Haugeland, J. (1997). *Mind Design II: Philosophy, Psychology, and Artificial Intelligence*. The MIT Press. <https://doi.org/10.7551/mitpress/4626.001.0001>
- Hendler, J. (2008). Avoiding Another AI Winter. *IEEE Intelligent Systems*, *23*(2), 2–4. <https://doi.org/10.1109/MIS.2008.20>
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models*. arXiv preprint arXiv:2001.08361
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, *65*(6), 386–408. <https://doi.org/10.1037/h0042519>
- Rumelhart, D. E., McClelland, J. L., & Group, P. R. (1986). *Parallel Distributed Processing, Volume 1: Explorations in the Microstructure of Cognition: Foundations*. The MIT Press. <https://doi.org/10.7551/mitpress/5236.001.0001>
- Shortliffe, E. H. (1974). *MYCIN: A Rule-Based Computer Program for Advising Physicians Regarding Antimicrobial Therapy Selection* (Technical Report STAN-CS-74-465). Stanford University.
- Statistical Machine Translation. (n.d.). *IBM Models*. <http://www2.statmt.org/survey/Topic/IBMModels>
- Turing, A. M. (1950). i.—Computing Machinery And Intelligence. *Mind*, *LIX*(236), 433–460. <https://doi.org/10.1093/mind/LIX.236.433>
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). *Attention Is All You Need*. arXiv preprint arXiv:1706.03762
- Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. arXiv preprint arXiv:2307.09702

Anatomy 里 Structured Output Manager 就是这间房间。功能页：[speculative-decoding.md](../../features/speculative-decoding.md) 管的是「猜字」；这篇管的是「猜的字还得长对形状」。
