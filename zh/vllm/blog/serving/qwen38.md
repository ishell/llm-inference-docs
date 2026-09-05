---
source: https://vllm.ai/blog/2026-08-12-qwen3.8
lang: zh
voice: literary-study
fetched: 2026-09-04
---

# Qwen3.8-2.4T：Max 级开权，引擎不用换骨架

英文对照：[en/vllm/blog/serving/qwen38.md](../../../../en/vllm/blog/serving/qwen38.md)  
原文：https://vllm.ai/blog/2026-08-12-qwen3.8  
2026-08-12。署名 **vLLM Team and Inferact**。数字是演示。Qwen 家第一次把 Qwen-Max 级开出来：`Qwen3.8-2.4T-A95B`。骨架仍是 Qwen 3.5——512 expert 的稀疏 MoE，92 层里每 4 层一次 full attention，其余 69 层 linear attention。**不是新引擎。** 前身 hybrid：[qwen3-next.md](qwen3-next.md)。GDN + P/D 后续：[qwen35-25k-tps.md](qwen35-25k-tps.md)。GSM8K / AIME25 是页上的核对，不是你的 SLA。

**Figure（social preview；未抓图；按页上路径）。** 原文 `/assets/figures/2026-08-12-qwen3.8/social-preview.png`。

**原文 TL;DR：**

- **Day-0：** 复用 Qwen 3.5 架构；第一天就能在 vLLM 上跑；不用改架构。
- **精度：** 官方 FP8 / BF16；Inferact 另放 NVFP4 / MXFP4。
- **硬件：** 至少两台 NVIDIA B300 / AMD MI355X；FP4 单机可试。
- NVIDIA 和 AMD 的 kernel 活叠在已有 Qwen 3.5 路径上。

## 为什么要这只

Qwen3.8-2.4T-A95B 被写成 Qwen 家第一次把 Qwen-Max 级权重开出来。骨架是 Qwen 3.5；vLLM 开箱就能跑。Inferact 的 MXFP4 / NVFP4：声称满精度质量，内存和带宽下去。

2.4T 稀疏 MoE，**512** expert。92 层 hybrid：每 4 层一次 full attention；其余 **69** 层 linear attention。推理：≥2× B300 / MI355X，FP4 可以单机。

## Quick start

NVFP4（`--linear-backend flashinfer_cutedsl`，`--tensor-parallel-size 8`，MTP 3）：

```bash
# See recipes for the exact docker run command
vllm serve Inferact/Qwen3.8-2.4T-A95B-NVFP4 \
  --linear-backend flashinfer_cutedsl \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

MXFP4（同一套 parser / MTP；**不必** `flashinfer_cutedsl`）：

```bash
vllm serve Inferact/Qwen3.8-2.4T-A95B-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

完整起服指南：[recipes.vllm.ai](https://recipes.vllm.ai/)。

## FP4 量化：质量还在，成本下去

Inferact 把选中的层——包括 routed expert——用 Round-to-Nearest (RTN) 加 activation calibration 量化到 FP4（4-bit activation）。初步核对：量化精度还在。**要复现评测，得把 reasoning budget 开大。**

| Benchmark | FP8 | NVFP4 |
| :--- | :--- | :--- |
| GSM8K (strict / flexible) | 89.61% / 90.52% | 90.37% / 91.05% |
| AIME25 @3 (avg / pass) | 87.78% / 93.33% | 92.22% / 96.67% |

量化不是这篇的精度故事，预算才是。

## Optimizations

Kernel 叠在已有 Qwen 3.5 支持上。

**NVIDIA。** NVIDIA 和 Inferact：Linear Attention（Gated Delta Rule）、Attention（GQA）、Dense GEMM、MoE routing。新的 fused kernel 砍通信。怎么切活：Attention 走 DP+TP，MoE 走 Expert Parallelism。

**AMD Instinct。** AITER-fused Gated DeltaNet decode、attention、MoE——少 kernel-launch，少搬数据。共享 expert 路径：hipBLASLt GEMM。Routed expert：AITER FusedMoE。AMD Quark：MXFP4，内存下去，声称精度仍强。

## Deployment tips

Model card 上的采样：

```
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
```

推理模：agent 工作流要把 `max_tokens` 开大。页上的客户端（`timeout=3600`，`max_tokens=128_000`；`model="Qwen/Qwen3.8-2.4T-A95B"`，serve 用的是 Inferact FP4 slug）：

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.8-2.4T-A95B",
    messages=[{"role": "user", "content": "Give me three primes above 100."}],
    temperature=1.0, top_p=0.95, max_tokens=128_000,
)
print(resp.choices[0].message.content)
```

## Acknowledgements

Qwen 团队（权重 + 持续合作）。NVIDIA 和 AMD（共研 kernel）。Inferact（量化 checkpoint + 端到端 vLLM）。更广的 vLLM 社区。早期测试点名的推理伙伴：DigitalOcean、Together AI。
