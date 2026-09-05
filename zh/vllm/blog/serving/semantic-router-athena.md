---
source: https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release
lang: zh
voice: literary-study
fetched: 2026-09-05
---

# Semantic Router v0.2 Athena：换底座，当系统脑

英文对照：[en/vllm/blog/serving/semantic-router-athena.md](../../../../en/vllm/blog/serving/semantic-router-athena.md)  
原文：https://vllm.ai/blog/2026-03-10-v0.2-vllm-sr-athena-release  
2026-03-10。署名 **vLLM Semantic Router Team**。仓库：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)。立项：[semantic-router](semantic-router.md)。脊柱：[semantic-router-signal](semantic-router-signal.md)。v0.1：[iris](semantic-router-iris.md)。分类核 LoRA：[modular](semantic-router-modular.md)。HaluGate：[halugate](halugate.md)。后来视觉路径的坑：[vision](semantic-router-vision.md)。合作愿景：[amd](semantic-router-amd.md)。现场池：[mom-amd](semantic-router-mom-amd.md)。下一版：[themis](semantic-router-themis.md)。不要和引擎里的 [Router](router.md) 混。MI300X 延迟和社区数字是发版快照。

同目录还有：[session](semantic-router-session.md)、[fusion](semantic-router-fusion.md)、[micro-agent](semantic-router-micro-agent.md)、[mom](semantic-router-mom.md)。

[Iris](semantic-router-iris.md) 之后这一轮：模型栈重砌，路由伸进安全、semantic cache、记忆、检索、长上下文信号，并往更大的赌注走——语义路由当 MoM 和多 agent 部署的 **系统脑**。

v0.2 代号 **Athena**。完整换模型、runtime 更硬，还有 **ClawOS**：实验性操作层，用路由、记忆、安全、聊天管团队，编排多套 OpenClaw。Iris 是用户和模型之间的桥；Athena 开始把桥变成模型团队的操作面。

本地图（原文版权仍归原站；学习对照用）：

![athena 0](../../../../assets/vllm/blog/serving/semantic-router-athena/01-athena-0.png)

**Figure 1.** 换模型、更硬的 runtime，ClawOS 当实验操作层。

## 为什么叫 Athena

神话里 Athena 是智慧、战略、有纪律的手艺。v0.2 不只更快，也不只更多插件。它要 **更战略**：选哪只模型、编排 OpenClaw worker、跨轮记住该记的、用工具把决策摊开、让团队真能运营。

![athena 1](../../../../assets/vllm/blog/serving/semantic-router-athena/02-athena-1.png)

**Figure 2.** 战略，不只是插件变多：选模、团队、记忆、工具。

## v0.2 新了什么

### 1. 换模型栈，重砌 MoM 底座

要紧的改动在 UI 和 DSL 下面：**模型栈重砌了**。

中心：[`mmbert-embed-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka) 和分类家族 [`mom-multilingual-class`](https://huggingface.co/collections/llm-semantic-router/mom-multilingual-class)。embedding、intent、jailbreak、PII、feedback、fact-check 及相关面，迁到同一套 mmBERT 派生底座，并对齐同一条 ONNX + Flash Attention 加速路。

另有 [`multi-modal-embed-small`](https://huggingface.co/llm-semantic-router/multi-modal-embed-small)：文、图、音频进 **同一 384d 空间**。跨模态检索（用文本搜图、用描述找音频）。宣称用 `transformers` + `torch` 就能加载，不另要自定义 runtime。后来这只模型的 Candle 路径，是 [vision](semantic-router-vision.md) 那篇硬化故事。

![athena 1b](../../../../assets/vllm/blog/serving/semantic-router-athena/03-athena-1b.png)

**Figure 3.** 共用的 mmBERT 底座，加上一只紧凑跨模态 embed。

立刻能用的三件（发布页数字）：

- **Multi-Modal Embed Small：** 约 **120M**、共用 **384d**、图文对齐、2D Matryoshka、sub-100ms 推理目标、宣称 **Audio-Text Retrieval R@1 = 36.4%**
- **mmBERT-Embed-32K-2D-Matryoshka：** **32K** 上下文、**1800+** 语言、**307M**、**STS 80.5**、**768d → 256d** 截断约留 **99%**、**22L → 6L** 早退约 **3.3×**
- **mom-multilingual-class** 把这根骨头收成一套分类器，共享 ONNX 加速

五件核心任务，各有 **merged** 和 **LoRA**：

| Task | Merged model | LoRA model |
| --- | --- | --- |
| Intent | `mmbert32k-intent-classifier-merged` | `mmbert32k-intent-classifier-lora` |
| Jailbreak | `mmbert32k-jailbreak-detector-merged` | `mmbert32k-jailbreak-detector-lora` |
| PII | `mmbert32k-pii-detector-merged` | `mmbert32k-pii-detector-lora` |
| Fact-check | `mmbert32k-factcheck-classifier-merged` | `mmbert32k-factcheck-classifier-lora` |
| Feedback | `mmbert32k-feedback-detector-merged` | `mmbert32k-feedback-detector-lora` |

| 新底座 | Athena 改什么 |
| --- | --- |
| `multi-modal-embed-small` | 文图音频同一 384d |
| `mmbert-embed-32k-2d-matryoshka` | 32K、1800+ 语言、2D Matryoshka 运行时控制 |
| ONNX + CK Flash Attention | 新栈在生产里更快，不只纸上更新 |

![athena 2](../../../../assets/vllm/blog/serving/semantic-router-athena/04-athena-2.png)

**Figure 4.** 换模型也是换 runtime：ONNX、ROCm、CK Flash Attention。

三路对照，**AMD Instinct MI300X**，真实路径 **Envoy (:8801) → ext_proc → SR (:50051)**：

| Request size | ONNX + GPU avg | ONNX + CPU avg | Candle + CPU avg |
| --- | --- | --- | --- |
| ~500 tokens | 22 ms | 853 ms | 1053 ms |
| ~2000 tokens | 31 ms | 1814 ms | 1805 ms |
| ~8000 tokens | 128 ms | 4796 ms | 1830 ms |

**Domain extraction：** ONNX+GPU 在这三档 **10.2 / 16.3 / 36.1 ms**，对 ONNX+CPU **630.4 / 833.3 / 743.9 ms**，对 Candle+CPU **849.0 / 1304.9 / 1311.5 ms**。**PII extraction：** ONNX+GPU **8.4 / 19.0 / 118.8 ms**，对 ONNX+CPU **729.5 / 1781.8 / 4783.9 ms**，对 Candle+CPU **854.2 / 1299.8 / 1327.8 ms**。

三只分类器同时载在 MI300X 上；旧 SDPA 撞内存墙：

| Sequence length | SDPA | CK Flash Attention | Result |
| --- | --- | --- | --- |
| 4096 | 167 ms | 51 ms | **3.3× faster** |
| 8192 | OOM | 105 ms | SDPA 挂，FA 还能跑 |
| 16384 | OOM | 259 ms | FA 到 16K |
| 32768 | OOM | 756 ms | FA 到满 32K |

FA 怎么接：`onnx-binding/ort-ck-flash-attn` 下一只独立的 **ONNX Runtime custom-op 库**，在 ROCm 上注册 `com.ck::CKFlashAttention`，直接调 AMD Composable Kernel 的 tiled FMHA。图改写把密 SDPA 子图换成单个 CK Flash Attention 节点。不再物化密 **`[1, 1, S, S]`** attention mask，从 `attention_mask` 推出轻量 **`[B, 1, 1, S]`** padding bias，把 sliding-window 设进 kernel。局部注意力层用 CK 窗口参数；全局层切回满注意力。**认得模型的 ONNX 改写 + 自定义 ROCm kernel**，不是拨一下 backend。

更重：CK Flash Attention 跑完 **20** 路并发 **32K-token** 请求，median **9872 ms** / p95 **14862 ms**，**零 OOM**，校验查询上分类结果相同。

### 2. 选模变成一等原语

不再是路线图。可训的 ML 选择器，加上运行时策略。管线位置写死：抽信号 → 匹配决策 → **决策命中之后**，**按决策的算法** 在它的 `modelRefs` 里挑。选模是「这条请求属于这条决策」和「该哪只模型伺候」之间的最后一步。

| Family | Method | 干什么 |
| --- | --- | --- |
| ML-based | **KNN** | 近邻历史查询投票 |
| ML-based | **KMeans** | 簇级质量 / 效率 |
| ML-based | **SVM** | RBF 边界切模型偏好 |
| ML-based | **MLP** | embedding 进神经 router；Candle 推理 |
| Advanced | **Static** | 可预期比适应更重要时钉死默认 |
| Advanced | **Latency-Aware** | 用 TPOT 和 TTFT 分位数挑最快 |
| Advanced | **Elo** | 反馈 / 成对偏好上的 Bradley-Terry |
| Advanced | **RouterDC** | 双对比去贴模型描述 |
| Advanced | **AutoMix** | 便宜先试，自核再升级 |
| Advanced | **Hybrid** | 质量 / 相似 / 成本可配权重 |
| Advanced | **Thompson Sampling** | serving 时边探边用 |
| Advanced | **GMTRouter** | 多轮历史上的图路由 |
| Advanced | **Router-R1** | 外面一只 router 模型先想，再选下游 |

![athena 3](../../../../assets/vllm/blog/serving/semantic-router-athena/05-athena-3.png)

**Figure 5.** 决策命中之后才选模，不是拿选模替换信号。

运营层：ML 训练和配 config 的 setup wizard、CLI 和 runtime 接入、指标、E2E、dashboard 上的 Elo 反馈。

### 3. ClawOS：OpenClaw 的操作层

**OpenClaw** 是底下的 agent 平台。**ClawOS** 是 Athena 在 Semantic Router 里叠上去的编排和操作体验。实验，但已经摸得着：MCP 工具和房间式聊天，拉起 OpenClaw 团队和 worker，在共享房间里协调，看整机运行时。

Dashboard 想塞进这套的：

- **Intelligent Routing：** 成本–质量选模
- **Safety Guardrails：** jailbreak、PII、幻觉
- **Hierarchical Memory Storage：** 长程、多步执行
- **Knowledge Sharing：** 跨 agent
- **Isolation & Team Management：** 多 agent 操作收在一块编排面

![athena 7](../../../../assets/vllm/blog/serving/semantic-router-athena/06-athena-7.png)

**Figure 6.** ClawOS：路由、安全、记忆、团队控制，一块面。

v0.2 落地的产品面：自然语言 MCP 控制；带 leader–worker 组成的团队；共享房间聊天；leader 协调 worker；dashboard 里配 worker；健康 / 组成 / 状态；更安全演示用的只读房间；共享 runtime，让 Claw worker 能和 router 住在同一操作环境。

不是成品平台。早早的一问：语义路由若不只选模型，而是 **给建在 OpenClaw 上的多 agent 操作层供电**，会怎样？

### 4. 记忆、RAG、响应状态进核

**Agentic Memory** 走 Milvus、混合记忆检索、记忆打分、Llama Stack 向量后端、记忆指标。OpenAI **Responses API** 加 Redis 持久化、会话链、更硬的测试。**Router Replay**：可插拔存储、按决策隔离、dashboard 可视化。

混合检索：**向量 + BM25 + n-gram**，加权融合或 **RRF**。内存后端能原生跑混合；Milvus 一类先拉宽候选再混合重排。

![athena 4](../../../../assets/vllm/blog/serving/semantic-router-athena/07-athena-4.png)

**Figure 7.** 记忆、RAG、Responses API、replay 进核，不再是边上的功能。

更可信：**MINJA** 挡记忆注入；写入记忆前做响应侧 jailbreak 闸；跨模型 cache 共享；Demand RAG 和向量库摄入。路由从无状态决策点，往记住 / 检索 / 核验 / replay 走。

### 5. 信号更富、更快、更安全

Iris 引入 Signal-Decision。Athena 把它撑开。

| 信号面 | Athena 加什么 | 为什么要紧 |
| --- | --- | --- |
| 请求核 | language、latency、context、complexity（含 few-shot 变体） | 不只看话题 |
| 控制上下文 | modality、authz | 媒体意图和访问更早上闸 |
| 反馈环 | feedback、preference 分类器 | 用户侧信号升一等 |
| 语义匹配 | 多模态 embedding、软 embedding 规则、HNSW | 检索面变大时仍快 |
| 确定性快路 | BM25、n-gram fuzzy、regex | 可审计，少脆 |
| 运行时置信 | 动态置信打分 | 质量，不只二值命中 |

安全更靠近主信号路：

| 安全面 | Athena 加什么 | 为什么要紧 |
| --- | --- | --- |
| Jailbreak | 并行信号；分类器 + **对比多轮** | 单轮和慢慢升级都抓 |
| PII | 并行信号；政策和揭示控制 | 同一套路由 / 执行层 |
| 工具安全 | 置信闸的工具过滤重排 | 不必把每个边写死 |
| 幻觉 | 更灵活的 **多层** 响应处理 | 警告 / 标注 / 把风险摊开 |

![athena 5](../../../../assets/vllm/blog/serving/semantic-router-athena/08-athena-5.png)

**Figure 8.** 信号更宽；keyword 路不再只认字面。

Keyword：**BM25** 在更大词集上做话题式路由；**n-gram** 容错近邻；**regex** 给合规 / 结构化检测。快路仍可审计；措辞吵也不必立刻漏掉。

### 6. NLP prompt 压缩当长上下文原语

**抽信号之前**压缩，不再另跳一只 LLM。

| 压缩层 | Athena 做什么 | 为什么要紧 |
| --- | --- | --- |
| 方法 | TextRank、位置加权、TF-IDF、novelty | 压长 prompt，不加 LLM 一跳 |
| 位置 | 压缩文本 **只给抽信号** | 原始请求仍送给 serving 模型 |
| 安全 | `skip_signals` 让 jailbreak 和 PII 看原文 | 该全保真的地方全保真 |
| 端到端 | Envoy STREAMED body + 快 JSON | 生产延迟，不只架构图好看 |

![athena 5b](../../../../assets/vllm/blog/serving/semantic-router-athena/09-athena-5b.png)

**Figure 9.** 信号路上确定性 NLP 压缩；serving 模型仍看原始 prompt。

MI300X buffered vs streamed：STREAMED（快 JSON、半流式 chunk、prompt 压缩）约 16K token 上 e2e **143 ms → 103 ms**；信号路把 **16K → 512** token 时，**jailbreak** 抽取 **127 ms → 10 ms**。

### 7. 可编程神经–符号配置语言

白皮书：一份类型化配置语言，当路由引擎的指令集——神经抽信号 + 符号评决策。路由配置往 **程序合成** 走：自然语言规格 → 合法路由程序。用 LLM coding agent 从自然语言合成策略，是论文里点名的主张。

落地：完整 DSL 编译器；可视化 builder；信号和决策的 dashboard CRUD；各配置面更好收敛；面向 Kubernetes 的部署时翻译更硬。

![athena 6](../../../../assets/vllm/blog/serving/semantic-router-athena/10-athena-6.png)

**Figure 10.** runtime config、dashboard、CLI、Kubernetes 表示往一块收。

还有：config reload 修好；deploy reload 之后 apiserver 分类服务刷新。能编译、能校验、能 round-trip，越来越能让 agent 写。

### 8. 零配置上手

安装和首跑连成一条。macOS / Linux：

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

安装器：侦测 Python，把 `vllm-sr` 装进隔离本地环境，启动器写到 `~/.local/bin/vllm-sr`，除非退出否则准备 Docker 或 Podman，自动跑第一次 `vllm-sr serve`，能开 dashboard 就开；远端机器给访问 / SSH 隧道提示，不默默失败。

之后，或任何时候从空目录再跑：

```bash
vllm-sr serve
```

可以：自动 bootstrap 最小工作区；背后写 `.vllm-sr/router-defaults.yaml`；dashboard 开在 setup 模式；带第一只模型和路由起步；**激活之后**才写 `config.yaml`。

![athena 8](../../../../assets/vllm/blog/serving/semantic-router-athena/11-athena-8.png)

**Figure 11.** 首跑 dashboard-first；`vllm-sr init` 变成可选。

高级用户仍可写 YAML。CLI-only、跳过自动启动、钉 runtime、或 `--platform amd` 走第一次 AMD 启动。最短路径：装、自动起、开 dashboard、配一只模型、激活。

### 9. Dashboard 变成系统脑

带测试查询的 topology；Router Replay 可视化；评测 API 和 dashboard 评测面；监控；认得 reasoning 的 playground；公测用只读模式；dashboard 里的 MCP 工具；布局 / 移动 / 落地页 / manager / 监控修一圈。

![athena 9](../../../../assets/vllm/blog/serving/semantic-router-athena/12-athena-9.png)

**Figure 12.** 从 dashboard 观察、调试、评测、演示——不只 YAML 和日志。

### 10. AMD ROCm 变成一等 `vllm-sr` 路径

正经流程，不是边上的实验。`vllm-sr` 的 ROCm 版镜像、AMD 剧本、CLI：

```bash
vllm-sr serve --platform amd
```

选 ROCm 镜像默认、把 AMD 平台传进容器 runtime、GPU-first 配置（除非显式关掉，`use_cpu` → `false`）、主机有 `/dev/kfd` 和 `/dev/dri` 就挂上。

![athena 10](../../../../assets/vllm/blog/serving/semantic-router-athena/13-athena-10.png)

**Figure 13.** `--platform amd`：ROCm 镜像、GPU 默认、ONNX + CK Flash Attention。

ROCm 镜像编 ONNX 后端的 router，装 ROCm ONNX Runtime，能加载 CK Flash Attention custom op。参考 AMD profile：对着 ROCm vLLM backend 做别名路由。端到端：专用镜像、写明的 serve、GPU 透传、ONNX + FA 进预定的运营体验。现场 demo 专篇：[mom-amd](semantic-router-mom-amd.md)。

### 11. 也是一轮研究 / 模型系统周期

- 白皮书：[Signal Driven Decision Routing for Mixture-of-Modality Models](https://vllm-semantic-router.com/white-paper/)
- 多模态 / 模态感知训练，含跨模态 embedding 和 mmBERT 分类器 / 模态 router
- 经 CK Flash Attention、ONNX 图改写、ROCm 向推理，把更长上下文加速推进核
- 研究制品到可部署 runtime 的桥拧紧

![athena 11](../../../../assets/vllm/blog/serving/semantic-router-athena/14-athena-11.png)

**Figure 14.** 研究、训练、生产一起动。

## 往前看：Athena 之后

Athena 把战略路由做成能运营的。下一步他们列：能从自然语言写 / 改 DSL 的训练 coding agent；用反向信号和路由结果迭代规则的自学习环；更深的多轮记忆和 agent 工具；更多运营自动化；更宽的多模态和工具安全；研究原型继续收进可部署 runtime。下一号发版是 [Themis](semantic-router-themis.md)。

## 致谢

`v0.1.0`（**2026-01-05**）到 `main`（**2026-03-09**）：**304** 个 commit，**43** 位 contributor。点名感谢：Red Hat、IBM、AMD、NVIDIA、DaoCloud，以及更广的开源社区。

## 上手

托管：[play.vllm-semantic-router.com](http://play.vllm-semantic-router.com)。

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

装 CLI，给 `vllm-sr serve` 准备本地 Docker / Podman，自动第一次启动，能开 dashboard 就开。

手工 / Windows：

```bash
pip install vllm-sr
vllm-sr serve
```

还没有 `config.yaml`：bootstrap + dashboard setup。YAML-first：仍可 `vllm-sr init` 再 `vllm-sr serve`。

```bash
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router
```

- 文档：[vllm-semantic-router.com](https://vllm-semantic-router.com)
- GitHub：[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
- 模型：[Hugging Face](https://huggingface.co/LLM-Semantic-Router)
- Slack：[vLLM Slack](https://vllm-dev.slack.com/archives/C09CTGF8KCN)

原文收束：*The bridge can now reason strategically. Welcome to Athena.*
