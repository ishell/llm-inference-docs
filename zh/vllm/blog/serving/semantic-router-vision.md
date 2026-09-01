---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# 视觉信号：不是换更大的 encoder，是 Candle 对不齐

英文对照：`en/vllm/blog/serving/semantic-router-vision.md`  
原文：https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening  
图在原网页。接 [Athena](semantic-router-athena.md) 的 `multi-modal-embed-small`。

11 张探针、21 标签：部署路径 9/11 把垂直类反了（82% 倒置）。同一只 mmes，PyTorch 护照锚点 cosine 0.7204，Candle 0.1576。SigLIP2 / HF SigLIP / mmEL / PyTorch mmes 都是 10/10——模型没坏。三刀：pooling 从 BERT mean+Linear+tanh 改成 SigLIP attentional probe（PR#1927）；`(x-0.5)/0.5` 归一化（#1928）；预处理搬进 Rust，Catmull-Rom 近似 PIL bicubic（#1943）。分支栈 20 图 vs Python cosine 最低 0.999557。没合入前当 PR 验证，不当线上保证。文本分类可并发，墙钟吃最慢那只（他们 CPU 轨迹约 1.3s）。
