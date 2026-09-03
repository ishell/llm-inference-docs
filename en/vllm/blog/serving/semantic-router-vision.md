---
source: https://vllm.ai/blog/2026-05-28-vllm-sr-vision-encoder-hardening
lang: en
fetched: 2026-09-01
---

# Vision signals: not a bigger encoder — Candle missed the reference

Chinese: [zh/vllm/blog/serving/semantic-router-vision.md](../../../../zh/vllm/blog/serving/semantic-router-vision.md)  
Follows Athena’s `multi-modal-embed-small`.

11-image probe, 21 labels: deployed path inverted the vertical on 9/11 (82%). Same mmes checkpoint: PyTorch passport-anchor cosine 0.7204 vs Candle 0.1576. SigLIP2 / HF SigLIP / mmEL / PyTorch mmes all 10/10 — the weights were fine. Three cuts: pooling BERT-mean+Linear+tanh → SigLIP attentional probe (PR#1927); `(x-0.5)/0.5` norm (#1928); preprocess in Rust, Catmull-Rom ≈ PIL bicubic (#1943). Branch-stack 20 images vs Python: min cosine 0.999557. Until merge, PR validation, not production. Text classifiers can run concurrent; wall clock is the slowest (their CPU trace ~1.3s).

Local figures (copyright remains with the original site; study copies):

![hero](../../../../assets/vllm/blog/serving/semantic-router-vision/01-hero.png)

![policy layer](../../../../assets/vllm/blog/serving/semantic-router-vision/02-policy-layer.png)

![inversion heatmap](../../../../assets/vllm/blog/serving/semantic-router-vision/03-inversion-heatmap.png)

![encoder eliminated](../../../../assets/vllm/blog/serving/semantic-router-vision/04-encoder-eliminated.png)

![diagnostic gap](../../../../assets/vllm/blog/serving/semantic-router-vision/05-diagnostic-gap.png)

![hardening arc](../../../../assets/vllm/blog/serving/semantic-router-vision/06-hardening-arc.png)

![corpus alignment](../../../../assets/vllm/blog/serving/semantic-router-vision/07-corpus-alignment.png)

![cyclotron demo](../../../../assets/vllm/blog/serving/semantic-router-vision/08-cyclotron-demo.png)

![parallel dispatch](../../../../assets/vllm/blog/serving/semantic-router-vision/09-parallel-dispatch.png)

![next steps](../../../../assets/vllm/blog/serving/semantic-router-vision/10-next-steps.png)
