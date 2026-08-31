---
source: https://docs.nvidia.com/nim/benchmarking/llm/latest/benchmarking-lora.html
lang: zh
fetched: 2026-08-30
---

# LoRA 模型怎么测

PEFT（参数高效微调）可以低成本微调大模型。NIM 支持 LoRA，并能同时加载多个 adapter。按 PEFT 指南加载 Hugging Face 或 NeMo adapter，用环境变量把 adapter 目录传给 NIM。

加载后，把 model ID 换成 LoRA 名字即可查询：

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/v1/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"model": "llama3-8b-instruct-lora_vhf-math-v1",
"prompt": "John buys 10 packs of magic cards. Each pack has 20 cards and 1/4 of those cards are uncommon. How many uncommon cards did he get?",
"max_tokens": 128
}'
```

用 AIPerf 压测时，`-m` 传 adapter ID。`--model-selection-strategy {round_robin,random}` 控制按轮询还是随机选 adapter。

测多个 adapter 时：`-m adapter1 adapter2`，并加上 `--model-selection-strategy`。

## 多 LoRA 压测实践

取决于基座大小、adapter 配置和负载：

- **基座：** 8B 和 70B 都能当 LoRA 基座。小模型更适合分类等传统 NLP；大模型更适合复杂推理。LoRA 可以在单卡 H100 上用 4-bit 量化微调 70B。
- **Adapter rank：** 精度和组 batch 效率之间权衡。常用 8 / 16 / 32 / 64。运维侧统一 rank 更容易组 batch。
- **输出长度：** 设 `ignore_eos`，生成到 `max_tokens`，OSL 才稳定。
- **系统负载：** 并发应贴近真实用量，且别超出有效 batch。NVIDIA 的经验：单卡 8B，并发上限大约 250。
- **任务类型：** 生成和非生成都要覆盖。ISL 200–2000、OSL 1–2000 能覆盖分类、摘要、翻译、代码生成。
