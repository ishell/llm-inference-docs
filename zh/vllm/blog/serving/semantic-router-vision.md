---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# 视觉信号：不是换更大的 encoder，是 Candle 对不齐

英文对照：[en/vllm/blog/serving/semantic-router-vision.md](../../../../en/vllm/blog/serving/semantic-router-vision.md)  
原文：https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening  
2026-05-28。署名 **David Shrader, Huamin Chen, Xunzhuo Liu, Bowei He, and the vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[Iris](semantic-router-iris.md) / [signal-decision](semantic-router-signal.md)。接 Athena 的 [`multi-modal-embed-small`](semantic-router-athena.md)。Candle 核一家：[modular](semantic-router-modular.md)。后来：[themis](semantic-router-themis.md)、[session](semantic-router-session.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)。不要和引擎里的 [Router](router.md) 混。余弦和倒置率是**他们的**探针 / PR 分支数字。没合入前当 PR 验证，不当线上保证。

同目录还有：[amd](semantic-router-amd.md)、[mom-amd](semantic-router-mom-amd.md)、[mom](semantic-router-mom.md)、[halugate](halugate.md)。

多数 router 拿一句 prompt 选端点。VSR 的赌注：在 serving 模型之前抽信号、合成决策、让路径可观测可编程。从文本起手。下一道边界是多模态：图 / 截屏 / 扫描 / 文档页进了请求，router 在推理 **请求证据**，不只一句 prompt。图可能才是让请求变成临床、受监管、安全敏感、出域、或该上更强 VLM 的那一块。只看文本，是在路由一份 **残请求**。

要紧的一步不是「加一只图像 encoder」。是把视觉证据收成同一张布上的 **可信 VSR 信号**。围着 `multi-modal-embed-small` 的部署路径看起来 **自信地错了**。第一猜：紧凑 encoder 不够强。真问题：**同一份权重，Rust/Candle 路径对不齐 PyTorch 参考路径**。

本地图（原文版权仍归原站；学习对照用）：

![hero](../../../../assets/vllm/blog/serving/semantic-router-vision/01-hero.png)

**Figure 1.** 视觉证据要变成带类型的信号，不是边上的 embedding。

## 多模态路由不是图像分类

文本路由已经不只话题匹配。信号是独立观察；决策用优先级和布尔把它们合成；插件 / 模型引用说下一步干什么。所以才写得出「安全敏感的代码审查走更强模型并过 jailbreak」，而不是「计算机 → 编码模型」。

多模态形状不变；分析单位变成 **整条请求**。文本可以很泛，图带着决定性证据：

| 请求证据 | 只看文本看见 | 多模态该看见 |
| --- | --- | --- |
| "Summarize this" + 护照图 | 泛泛摘要 | 身份证件、PII 风险、受限处理 |
| "What does this show?" + 胸片 | 含糊的看图问 | 临床图、医疗域政策、够格的 VLM |
| "Find the bug" + 代码截屏 | 编码请求 | 代码制品、可能泄密、安全审查 |
| 医疗 prompt + 不相干的车图 | 医疗文本 | 出域视觉证据，澄清或拒绝 |

图像 embedding 变成 **带类型的信号**，和文本意图、PII、jailbreak、域、相似、插件、选模并列。从 prompt 级路由走到 **请求级政策**。

![policy layer](../../../../assets/vllm/blog/serving/semantic-router-vision/02-policy-layer.png)

**Figure 2.** 同一张 Signal-Decision 布；证据面更大。

文本信号错了，会走错模型或跳过插件。视觉信号 **反相关**，router 可以自信地错，还留下一条干净的、记错决策的审计迹。参考对齐是 **控制面不变量**。

## 视觉信号自信地错了

11 张探针、三个垂直、21 个候选标签：部署的 `multi-modal-embed-small`（mmes）在 **11 张里 9 张** 把 **错误垂直** 排最高。医疗 X 光更贴半导体候选而不是医疗。身份证件贴不近身份锚点。**82% 倒置**——反相关，不只是吵。

![inversion heatmap](../../../../assets/vllm/blog/serving/semantic-router-vision/03-inversion-heatmap.png)

**Figure 3.** 排反了：信心朝错的方向。

弱分类器通常看起来不确定。倒置的看起来很确定。对政策层来说，这可能比 **没有** 图像信号更糟。暴露它的面：围着 mmes 的图像模态路由，含 [PR #1881](https://github.com/vllm-project/semantic-router/pull/1881) 的 E2E profile。真图走过 Candle binding，缝就看见了。

## 诱人的解释：换更大的 encoder

自然的第一假说：紧凑 encoder 不够强。当时已经在看 SigLIP2 和更大的 `multi-modal-embed-large`（mmEL）。同一 21 候选探针直接测：

- SigLIP2-base：**10/10**
- Hugging Face Transformers 上的 SigLIP-base：**10/10**
- mmEL（视觉塔基于 SigLIP2）：**10/10**
- 经 **PyTorch 参考** 路径的 mmes：**10/10**

![encoder eliminated](../../../../assets/vllm/blog/serving/semantic-router-vision/04-encoder-eliminated.png)

**Figure 4.** 这一家没坏；连「失败」的 mmes 在参考加载器上也没坏。

顺带：更大的 SigLIP2-so400m 在这组探针里 OOD 拒绝更狠（误入的发动机图压得更死）。以后内存够、视觉塔可以更大时也许有用。**不是** 线上倒置的 bug。

## 改了调查形状的参考对照

同一只 mmes、同一张护照夹具、两条路径。PyTorch 参考对护照锚点余弦 **0.7204**。部署的 Candle binding：**0.1576**。同一模型同一夹具，量级差 **5–8×**。

![diagnostic gap](../../../../assets/vllm/blog/serving/semantic-router-vision/05-diagnostic-gap.png)

**Figure 5.** 同一份权重，两套加载器：生产路径漂了。

此后别再问「换哪只 encoder？」问：生产在哪和模型卡参考分岔？多模态路由里，**参考对照该当第一诊断**。这个 embedding 是政策证据，不只是检索。

## 真正坏的是什么

Candle 实现漂了，不是权重。三刀：

1. **Pooling 头。** `candle-binding/src/model_architectures/embedding/multimodal_embedding.rs` 里的 `SigLIPVisionEncoder::forward` 在做 BERT 式 mean + Linear + tanh。SigLIP 用的是 attentional probe。[PR #1927](https://github.com/vllm-project/semantic-router/pull/1927) 在 Candle binding 里对齐 SigLIP 多头注意力 pooling。
2. **归一化。** Go 图像加载器给出 `[0, 1]` 的 CHW float32。SigLIP 要 `(x - 0.5) / 0.5`。[PR #1928](https://github.com/vllm-project/semantic-router/pull/1928) 在 Rust encoder 路径里做这步。
3. **预处理残差。** 旧 Go resize：4-tap bilinear。PyTorch 参考：经 `SiglipProcessor` 的 PIL 风格。[PR #1943](https://github.com/vllm-project/semantic-router/pull/1943) 把 decode、resize、CHW float32 搬进 Rust（`image` crate，Catmull-Rom ≈ PIL bicubic + antialias）。

![hardening arc](../../../../assets/vllm/blog/serving/semantic-router-vision/06-hardening-arc.png)

**Figure 6.** Pooling、归一化，再把预处理搬过 FFI。

跨语言栈里好漏：Go、Rust FFI、Candle、PyTorch 各自看起来都合理，端到端路由已经断了。

## 验证状态

下面数字来自 #1927 / #1928 / #1943 的 **PR 分支栈**。三只都没合入前，当分支验证读，不当已发版生产行为。

护照夹具（`inrule_identifier_passport.jpg`）上的三向量隔离：

| Comparison | Cosine | Max abs diff | 隔离什么 |
| --- | ---: | ---: | --- |
| Python vs Candle-PIL | **0.999989** | 0.000911 | 只看模型前向 |
| Candle-PIL vs Candle-Go | **0.999916** | 0.001992 | 只看预处理 |
| Python vs Candle-Go | **0.999902** | 0.002120 | 整条分支栈 |

第一行：Rust 模型前向能对到 PyTorch 的 fp32 噪声量级。前两刀之后剩下的漂在预处理——所以预处理要搬过 FFI。

20 图语料（证件、环境、代码、对抗、OOD）：

- Cosine：最低 **0.999557**，均值 **0.999919**，最高 **0.999978**
- **20 / 20** 图相对 PyTorch cosine ≥ 0.999
- 修之前，夹具上预处理 cosine：**0.990145**

![corpus alignment](../../../../assets/vllm/blog/serving/semantic-router-vision/07-corpus-alignment.png)

**Figure 7.** 隔离方法：把模型前向的漂和预处理的漂拆开。

## 这把什么解开

视觉路径可信之后，图是一等证据，不是边上的 metadata。不只「有图的请求 → 图像模型」：

| 组合信号 | 决策例子 |
| --- | --- |
| 临床文本 + 临床图 + PHI/PII | 受保护的医疗 VLM 路；隐私插件打开 |
| 泛文本 + 证件图 | 拦、脱敏、或证件处理政策，**调用前** |
| 代码/安全 prompt + 代码截屏 | 安全专门模型；jailbreak 仍看原始请求 |
| 域内文本 + 出域图 | 澄清或拒绝图像证据，不硬走一条坏路 |

Iris 让决策可组合。Athena 让 router 更战略。多模态把只控语言扩成 **请求级** 控制。

公开 demo：[shrader.dev](https://shrader.dev)——今天是政策形状的 **文本路由** 版（域相关、隐私敏感路由、调用前拦住）。先把政策形状亮出来，再加图。

![cyclotron demo](../../../../assets/vllm/blog/serving/semantic-router-vision/08-cyclotron-demo.png)

**Figure 8.** 同一控制形状的文本政策 demo。

分类信号经 `runSignalDispatchers` 可并发；墙钟吃 **最慢** 那只启用的分类器，不是加总。代表轨迹：完整分类决策在 CPU 上约 **1.3 s**（他们测的）。

![parallel dispatch](../../../../assets/vllm/blog/serving/semantic-router-vision/09-parallel-dispatch.png)

**Figure 9.** 并行派信号；墙钟吃最慢那只分类器。

多模态是同一台政策引擎，证据面更大。图和文的信号该经同一套路由语义抽取、校验、合成、replay、审计。VSR 若要按视觉证据路由，视觉路径必须无聊地可靠。

## 下一步

把硬化 PR 合进去、审完；验证语料跟着多模态路由一起转。然后：图像派生信号进同一决策层；多模态决策在 replay / 指标 / 调试里可见；选模同时认得政策契合 **和** 模态能力；PII 和 jailbreak 仍全保真检查；往 agent 工作流伸，工具调用、记忆写入、模型调用共用一层决策。

![next steps](../../../../assets/vllm/blog/serving/semantic-router-vision/10-next-steps.png)

**Figure 10.** 文本是第一块控制面；多模态是下一块——不是 router 旁边另挂一只视觉分类器。

- 仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- 现场 demo：[shrader.dev](https://shrader.dev)

## 致谢

Huamin Chen 的 mmEL 指点，拆掉「换 encoder」的误诊；#1927 / #1928 / #1943 的维护者审阅；邀稿；更广的多模态分类器工作、`multi-modal-embed-small` 模型卡、Candle-binding。
