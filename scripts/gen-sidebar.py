#!/usr/bin/env python3
"""Regenerate /_sidebar.md from note titles. Run from repo root."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def title(rel: str) -> str:
    p = ROOT / rel
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if not m:
        return p.stem
    t = m.group(1).strip()
    t = re.sub(r"（产品页）", "", t)
    if len(t) > 42:
        t = t[:40] + "…"
    return t


def line(rel: str) -> str:
    return f"  - [{title(rel)}]({rel})"


NVIDIA_BENCH = [
    "zh/nvidia/benchmarking/nim-index.md",
    "zh/nvidia/benchmarking/nim-01-overview.md",
    "zh/nvidia/benchmarking/nim-02-metrics.md",
    "zh/nvidia/benchmarking/nim-03-parameters.md",
    "zh/nvidia/benchmarking/nim-04-aiperf.md",
    "zh/nvidia/benchmarking/nim-05-lora.md",
    "zh/nvidia/benchmarking/nim-product-benchmarking.md",
    "zh/nvidia/benchmarking/blog-01-fundamental-concepts.md",
    "zh/nvidia/benchmarking/blog-02-genai-perf-and-nim.md",
    "zh/nvidia/benchmarking/blog-genai-perf-openai.md",
]
NVIDIA_TUNE = [
    "zh/nvidia/performance-tuning/mastering-llm-techniques.md",
    "zh/nvidia/performance-tuning/blog-03-tensorrt-llm.md",
    "zh/nvidia/performance-tuning/trtllm-product.md",
    "zh/nvidia/performance-tuning/trtllm-tuning-guide.md",
    "zh/nvidia/performance-tuning/trtllm-baseline.md",
    "zh/nvidia/performance-tuning/trtllm-build-flags.md",
    "zh/nvidia/performance-tuning/trtllm-max-batch.md",
    "zh/nvidia/performance-tuning/trtllm-sharding.md",
    "zh/nvidia/performance-tuning/trtllm-fp8.md",
    "zh/nvidia/performance-tuning/trtllm-runtime-flags.md",
    "zh/nvidia/performance-tuning/trtllm-kvcache.md",
    "zh/nvidia/performance-tuning/trtllm-paged-attention-ifb.md",
    "zh/nvidia/performance-tuning/trtllm-bench.md",
]
NVIDIA_COST = ["zh/nvidia/cost/blog-04-tco.md"]
NVIDIA_TOOLS = [
    "zh/nvidia/tools/aiperf.md",
    "zh/nvidia/tools/aiperf-load-generator.md",
    "zh/nvidia/tools/aiperf-metrics.md",
    "zh/nvidia/tools/aiperf-comprehensive.md",
    "zh/nvidia/tools/genai-perf.md",
    "zh/nvidia/tools/perf-analyzer.md",
    "zh/nvidia/tools/triton-performance-tuning.md",
]
VLLM_DOCS = [
    "zh/vllm/getting-started/index.md",
    "zh/vllm/getting-started/quickstart.md",
    "zh/vllm/getting-started/serve.md",
    "zh/vllm/optimization/optimization.md",
    "zh/vllm/benchmarking/cli.md",
    "zh/vllm/benchmarking/auto-tune.md",
    "zh/vllm/metrics/production-metrics.md",
    "zh/vllm/metrics/design-metrics.md",
    "zh/vllm/features/prefix-caching.md",
    "zh/vllm/features/prefix-caching-design.md",
    "zh/vllm/features/speculative-decoding.md",
    "zh/vllm/features/v1-guide.md",
]
SKIP_BLOG = {"README.md", "MUST-READ.md", "FLAG-MAP.md", "CATALOG.md"}


def folder_files(sub: str) -> list[str]:
    d = ROOT / "zh/vllm/blog" / sub
    return [
        str(p.relative_to(ROOT))
        for p in sorted(d.glob("*.md"))
        if p.name not in SKIP_BLOG
    ]


def main() -> None:
    out = [
        "- 开始",
        "  - [那条评论在指什么](zh/GUIDE.md)",
        "  - [总目录](README.md)",
        "  - [必读博客](zh/vllm/blog/MUST-READ.md)",
        "  - [旋钮对照](zh/vllm/blog/FLAG-MAP.md)",
        "- NVIDIA · 压测",
        *[line(p) for p in NVIDIA_BENCH],
        "- NVIDIA · 调优",
        *[line(p) for p in NVIDIA_TUNE],
        "- NVIDIA · 成本",
        *[line(p) for p in NVIDIA_COST],
        "- NVIDIA · 尺子",
        *[line(p) for p in NVIDIA_TOOLS],
        "- vLLM · 文档",
        *[line(p) for p in VLLM_DOCS],
        "- vLLM 博客 · 架构",
        *[line(p) for p in folder_files("architecture")],
        "- vLLM 博客 · 性能",
        *[line(p) for p in folder_files("performance")],
        "- vLLM 博客 · Serving",
        *[line(p) for p in folder_files("serving")],
        "- 英文对照",
        "  - [English README](README.md)",
        "  - [CATALOG](en/vllm/blog/CATALOG.md)",
    ]
    (ROOT / "_sidebar.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote _sidebar.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
