---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 视觉信号：不是换更大的 encoder，是 Candle 对不齐

英文对照：[en/vllm/blog/serving/semantic-router-vision.md](../../../../en/vllm/blog/serving/semantic-router-vision.md)  
原文：https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening  
2026-05-28。署名 **David Shrader, Huamin Chen, Xunzhuo Liu, Bowei He, and the vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。接 Athena 的 `multi-modal-embed-small`：[athena](semantic-router-athena.md)。信号合同：[themis](semantic-router-themis.md)。不要和引擎里的 [Router](router.md) 混。Cosine 和探针分数是**他们的** PR 分支测量。没合入前当校验轨迹，不当生产保证。

同目录还有：[modular](semantic-router-modular.md)、[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[session](semantic-router-session.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)、[mom](semantic-router-mom.md)。

多数路由器从 prompt 出发，选一个端点。VSR 的赌注：先抽信号，再合成决策，路径可观测，**然后**才到 serving 模型。Iris 把它带离固定的 domain 分类器。Athena 把它推向 MoM 和 agentic 部署的系统级智力层。

本地图（原文版权仍归原站；学习对照用）：

![hero](../../../../assets/vllm/blog/serving/semantic-router-vision/01-hero.png)

**Figure 1.** 下一道边界是多模态：图可以是决定性证据。

请求里一旦有图、截屏、扫描件、文档页，router 不再只对着 prompt 想。只看文本，就是在路由一份 **残缺请求**。要紧的一步不是「加一只 image encoder」。是把视觉证据收成可信的 VSR **信号**，和文本进同一张决策织物。

围着 `multi-modal-embed-small` 的部署多模态路径，看起来 **错得很自信**。第一反应：紧凑 encoder 不够强。真正的问题：**同一只模型，Rust/Candle 路径对不齐 PyTorch 参考路径**。

## 多模态路由不是图像分类

文本路由已经不止话题匹配。信号是独立观察；决策用优先级和布尔把它们合成；插件 / model ref 决定下一步。VSR 才能说「安全敏感的 code review 走更强推理模型并开 jailbreak 检查」，而不是「computer science 就去 coding 模型」。

多模态保持这个形状。分析单元变成 **整份请求**。文本可以很泛，图带着决定：

| Request evidence | Text-only router sees | Multimodal router should see |
|---|---|---|
| "Summarize this" + passport image | Generic summarization | Identifier document, PII risk, restricted handling |
| "What does this show?" + chest X-ray | Vague visual question | Clinical image, medical-domain policy, capable VLM target |
| "Find the bug" + code screenshot | Coding request | Code artifact, possible secret leakage, security review path |
| Medical prompt + unrelated car image | Medical text | Out-of-domain visual evidence, clarification or rejection path |

创新不是 VSR 能算 image embedding。是 embedding 变成 **带类型的信号**，和文本 intent、PII、jailbreak、domain、semantic similarity、插件、选模并排。Prompt 级路由变成 **请求级策略**。

![policy layer](../../../../assets/vllm/blog/serving/semantic-router-vision/02-policy-layer.png)

**Figure 2.** 图像证据进同一张 Signal-Decision 织物。

文本信号错了，策略会选错模型或跳过插件。视觉信号若 **反相关**，router 可以错得很自信，审计轨迹还干净——记的是错决策。参考对齐是 **控制面不变量**，不是模型质量卫生。

## 视觉信号错得很自信的时候

不是小幅掉点。**11** 张探针，三个垂直类，**21** 个候选标签：部署的 `multi-modal-embed-small`（mmes）路径，**11 张里 9 张把错误垂直类排第一**。医学 X 光更靠近半导体候选，而不是医学候选。证件不一定落在证件锚点附近。

这是 **82%** 倒置。反相关，不只是噪。

![inversion heatmap](../../../../assets/vllm/blog/serving/semantic-router-vision/03-inversion-heatmap.png)

**Figure 3.** 倒置热图：部署路径把错误垂直类排第一。

弱分类器通常显得不确定。倒置的分类器在错误方向上显得确定。对多模态策略层，这可能比 **没有** 图像信号更糟。

暴露它的面：围着 `multi-modal-embed-small` 的图像模态路由，包括 [PR #1881](https://github.com/vllm-project/semantic-router/pull/1881) 的 E2E routing profile。真图走进 Candle binding，缺口才看得见。

## 诱人的解释：换更大的 encoder

假说：紧凑 encoder 不够强。当时已经在看 SigLIP2 和更大的 `multi-modal-embed-large`（mmEL）。同一套 21 候选探针直接测：

- SigLIP2-base：**10/10**
- Hugging Face Transformers 上的 SigLIP-base：**10/10**
- mmEL（vision tower 基于 SigLIP2）：**10/10**
- mmes model card 走 **PyTorch 参考** 路径：**10/10**

![encoder eliminated](../../../../assets/vllm/blog/serving/semantic-router-vision/04-encoder-eliminated.png)

**Figure 4.** Encoder 家族排除：同一只 mmes，PyTorch 参考路径是好的。

所以 encoder 家族不是根因。连「失败」的 mmes，参考加载也表现正常。

他们留下的旁证：更大的 SigLIP2-so400m 在这套探针上 OOD 拒绝更强（一张误入的汽车引擎图）。以后若显存允许更大 vision tower，防御路由也许用得上。**不是** 这次生产倒置的 bug。

## 改写调查方向的参考对照

同一只 mmes，同一张护照夹具，两条路径。

PyTorch 参考：对着相关护照锚点 cosine **0.7204**。部署的 Candle-binding 路径：**0.1576**。同一只模型、同一张夹具，差 **5–8×**。

![diagnostic gap](../../../../assets/vllm/blog/serving/semantic-router-vision/05-diagnostic-gap.png)

**Figure 5.** 同一 checkpoint、同一张护照：0.7204 对 0.1576。

此后别再问「换哪只 encoder」。问：生产路径在哪和参考分叉？多模态路由里，**参考对照该是第一诊断**。Embedding 是策略证据。方向反了 → 下游每一层都可以逻辑正确、运营错误。

## 真正坏的是什么

漂在 **Candle 路径** 上，不在权重里。三刀：

1. **Pooling 头错了。** `candle-binding/src/model_architectures/embedding/multimodal_embedding.rs` 里的 `SigLIPVisionEncoder::forward` 在做 BERT 式 mean + Linear + tanh。SigLIP 用 attentional probe pooling。[PR #1927](https://github.com/vllm-project/semantic-router/pull/1927) 在 Candle binding 里对齐 SigLIP 的 multi-head attention pooling。

2. **归一化不完整。** Go 图像加载器给出 `[0, 1]` 的 CHW float32。SigLIP 要的是逐通道 `(x - 0.5) / 0.5`。[PR #1928](https://github.com/vllm-project/semantic-router/pull/1928) 把这步放进 Rust encoder 路径。

3. **预处理残留。** 旧的 Go 侧 resize：4-tap bilinear。PyTorch 参考：经 `SiglipProcessor` 的 PIL 风格。[PR #1943](https://github.com/vllm-project/semantic-router/pull/1943) 把 decode、resize、CHW float32 搬进 Rust（`image` crate，Catmull-Rom），近似 PIL bicubic + antialias。

![hardening arc](../../../../assets/vllm/blog/serving/semantic-router-vision/06-hardening-arc.png)

**Figure 6.** 三只 PR：pooling、归一化、预处理进 Rust。

跨语言 serving 栈里很容易漏。Go、Rust FFI、Candle、PyTorch 各自看起来都合理，端到端仍能把路由弄断。

## 校验状态

下面的数字：[#1927](https://github.com/vllm-project/semantic-router/pull/1927)、[#1928](https://github.com/vllm-project/semantic-router/pull/1928)、[#1943](https://github.com/vllm-project/semantic-router/pull/1943) 的 PR 分支栈。三只都合入之前：当 **分支栈校验**，不当已发布的生产行为。

规范护照夹具（`inrule_identifier_passport.jpg`）上的三向量隔离：

| Comparison | Cosine | Max abs diff | What it isolates |
|---|---:|---:|---|
| Python vs Candle-PIL | **0.999989** | 0.000911 | Model-forward only |
| Candle-PIL vs Candle-Go | **0.999916** | 0.001992 | Preprocessing only |
| Python vs Candle-Go | **0.999902** | 0.002120 | Full branch-stack pipeline |

第一行：Rust model-forward 和 PyTorch 对齐到 fp32 噪声级。前两刀之后剩下的漂在预处理——所以预处理要跨过 FFI 边界。

**20** 图语料（证件、环境、代码、对抗、OOD）：

- Cosine：最低 **0.999557**，均值 **0.999919**，最高 **0.999978**
- **20 / 20** 张相对 PyTorch 参考 cosine **>= 0.999**
- 修之前，规范夹具上预处理 cosine：**0.990145**

![corpus alignment](../../../../assets/vllm/blog/serving/semantic-router-vision/07-corpus-alignment.png)

**Figure 7.** 分支栈语料：相对 Python 参考，最低 cosine 0.999557。

方法和最终 cosine 一样要紧：生产和参考对照，把 model-forward 漂和预处理漂拆开，然后让 serving 和测试用同一套预处理语义。

## 这解锁什么

视觉路径可信之后，图是一等证据，不是边信道元数据。不只是「带图的请求去图像模型」。文本和图进同一张织物：

| Combined signal pattern | Example decision |
|---|---|
| Clinical text + clinical image + PHI/PII signal | Route to a protected medical VLM path with privacy plugins enabled |
| Generic text + identifier image | Block, redact, or route to an identity-document handling policy before model invocation |
| Code/security prompt + code screenshot | Route to a security-specialized model and keep jailbreak checks on the original request |
| In-domain text + out-of-domain image | Ask for clarification or reject the image evidence instead of forcing a bad route |

Iris 让决策可组合。Athena 换更强模型栈、选模、记忆、replay、更富信号。多模态把同一套架构从只控语言，扩成 **请求级** 控制。

点名的公开 demo：[shrader.dev](https://shrader.dev)。发文时展示的是 **文本路由** 版策略形：域相关、隐私敏感路由、模型调用前的拦截。图像加上去之前，先看清策略形状。

![cyclotron demo](../../../../assets/vllm/blog/serving/semantic-router-vision/08-cyclotron-demo.png)

**Figure 8.** 同一策略形的文本路由 demo（shrader.dev）。

文本路径还露出一条延迟性质，图进来以后仍要紧：分类信号可以经 `runSignalDispatchers` 并发，墙钟吃 **最慢那只已启用的分类器**，不是求和。页上代表性轨迹：完整分类决策在 CPU 上约 **1.3s**。

![parallel dispatch](../../../../assets/vllm/blog/serving/semantic-router-vision/09-parallel-dispatch.png)

**Figure 9.** 并行信号分发：墙钟跟最慢那只分类器（他们 CPU 轨迹约 1.3s）。

多模态不是另一条产品线。同一只策略引擎，证据面更大。图和文本该被抽取、校验、合成、replay、审计，语义同一套。VSR 若要按视觉证据路由，视觉路径必须无聊地可靠：对齐参考、活过 Go/Rust/Candle、策略更会表达时仍可测。

## 下一步

把 hardening PR 合进去、审完；校验语料跟着多模态路由一起活。让参考驱动的检查成为多模态 serving 的常态。然后是架构：

- 把图像派生信号露在和文本同一层决策里
- 让多模态决策在 replay、指标、调试里看得见
- 选模同时看策略契合 **和** 模态能力
- 安全关键信号（PII、jailbreak）保住高保真检查
- 把织物伸向 agentic 工作流：tool call、写记忆、调模型，过同一层决策

文本路由是第一块控制面。多模态是下一块。目标：不是在 router 旁边另养一只视觉分类器，而是让请求里每个有意义的部分，都进同一颗可编程的路由脑。

![next steps](../../../../assets/vllm/blog/serving/semantic-router-vision/10-next-steps.png)

**Figure 10.** 下一步：图像信号进同一层决策、replay、模态感知选模。

上手：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)，现场 demo [shrader.dev](https://shrader.dev)。

## 致谢

Huamin Chen 的 mmEL 指针，拆掉「换更大 encoder」的误诊；[#1927](https://github.com/vllm-project/semantic-router/pull/1927)、[#1928](https://github.com/vllm-project/semantic-router/pull/1928)、[#1943](https://github.com/vllm-project/semantic-router/pull/1943) 的维护者审阅；邀请写成这篇。更广的维护者团队：这条弧接上的多模态分类器工作、`multi-modal-embed-small` model card、以及底下的 Candle-binding 集成。
