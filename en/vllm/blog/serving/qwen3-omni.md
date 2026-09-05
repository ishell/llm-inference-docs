---
source: https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
lang: en
fetched: 2026-09-05
---

# Experience and Lessons Learned from Serving Multi-Stage Qwen3-Omni in vLLM-Omni

Chinese: [zh/vllm/blog/serving/qwen3-omni.md](../../../../zh/vllm/blog/serving/qwen3-omni.md)

2026-07-01. **vLLM-Omni Team and Ant Group SCT Team**. Staged Thinker → Talker → Code2Wav, not one Decode loop. TTS engineering sibling: [omni-tts.md](omni-tts.md). Same Omni line: [vllm-omni.md](vllm-omni.md). Study note; Seed-TTS `en` sweep and DFX numbers on the page, not your SLA. Audio **TTFP** (time to first audio packet) is not the same stopwatch as text **TTFT**.

**TL;DR from the page:**

- Three stages: Thinker (multimodal + text), Talker (RVQ codec), Code2Wav (waveform).
- Primary endpoint: `/v1/chat/completions`. `modalities` in the body: `["text"]` or `["text", "audio"]`.
- Stage batching + per-stage CUDA Graphs; async chunk / async omni output; Talker/Code2Wav replicas; hot-path cleanup.
- Sweep at concurrency 64 (Seed-TTS `en`, `Qwen3-Omni-30B-A3B-Instruct`): Batch **2.2** req/s, audio TTFP **5884 ms**, RTF **1.15** → stacked **11.7** req/s, **632 ms**, RTF **0.47**. CUDA Graph is the ~**4×** throughput jump; async chunk is the largest TTFP cut (**2790 → 655 ms**).

## TL;DR

vLLM-Omni's Qwen3-Omni serving stack:

- **A three-stage pipeline:** Thinker for multimodal reasoning, Talker for speech codec generation, Code2Wav for waveform reconstruction.
- **OpenAI-compatible serving:** `/v1/chat/completions` is the primary endpoint for Qwen3-Omni text and audio generation.
- **Batching, CUDA Graphs, async chunk, async output, replicas, and hot-path cleanup:** stage-level batching and per-stage graph capture on Thinker, Talker, and Code2Wav improve high-concurrency throughput; async-chunk handoffs and async output keep the pipeline and Decode workers from stalling on full-payload barriers and synchronous payload construction; Talker/Code2Wav replicas scale the speech-generation stages; hot-path cleanup trims the per-step model-internal overhead that scales with utterance length.
- **Performance validation:** controlled benchmark sweeps and DFX perf runs show lower audio TTFP (time to first audio packet), lower audio RTF (real-time factor), and higher throughput as each optimization layer is enabled.

Local figures (copyright remains with the original site; study copies):

![qwen3 omni serving flow](../../../../assets/vllm/blog/serving/qwen3-omni/01-qwen3-omni-serving-flow.svg)

![qwen3 omni optimization stack](../../../../assets/vllm/blog/serving/qwen3-omni/02-qwen3-omni-optimization-stack.svg)

![qwen3 omni cuda graph stages](../../../../assets/vllm/blog/serving/qwen3-omni/03-qwen3-omni-cuda-graph-stages.svg)

![qwen3 omni async chunk timeline](../../../../assets/vllm/blog/serving/qwen3-omni/04-qwen3-omni-async-chunk-timeline.svg)

![qwen3 omni async output step gap](../../../../assets/vllm/blog/serving/qwen3-omni/05-qwen3-omni-async-output-step-gap.svg)

![qwen3 omni async replica](../../../../assets/vllm/blog/serving/qwen3-omni/06-qwen3-omni-async-replica.svg)

![qwen3 omni bench reqps](../../../../assets/vllm/blog/serving/qwen3-omni/07-qwen3-omni-bench-reqps.svg)

![qwen3 omni bench rtf](../../../../assets/vllm/blog/serving/qwen3-omni/08-qwen3-omni-bench-rtf.svg)

![qwen3 omni bench ttfp](../../../../assets/vllm/blog/serving/qwen3-omni/09-qwen3-omni-bench-ttfp.svg)

Figure 7–9 **alt text** on the original page says concurrency `1, 8, 16, and 32`. Captions, the sweep paragraph, and the table use `1 / 16 / 32 / 64`. This note follows the captions and table.

## Quickstart

`--omni` resolves the default deploy profile:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

Explicit staged profile:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml
```

Bundled yaml has a `platforms:` section. Omni detects CUDA / NPU / ROCm / XPU and merges matching deltas — **no** extra CLI flag. Same launch across hardware.

Requests: `/v1/chat/completions`. Set `modalities` for output types. Deployment, async-chunk, multi-replica: [Qwen3-Omni online serving guide](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/qwen3_omni/README.md). The post does **not** inline a request JSON; `modalities` is the body field it names.

## Qwen3-Omni serving model

Text-only LLM serving is one loop: Prefill, Decode, detokenize. Qwen3-Omni adds two speech stages after multimodal reasoning, each with a different compute profile:

```text
Thinker   -> multimodal understanding + text generation
Talker    -> hidden states and embeddings to RVQ codec codes
Code2Wav  -> codec codes to waveform audio
```

**Figure 1.** Staged dataflow: Thinker emits text and hidden states; Talker emits codec codes; Code2Wav reconstructs audio.

## Optimization overview

No single recipe. Each lever targets a stage or a handoff. Walkthrough order is the validation order: **Why** / **Why it works** / **What you gain**.

| Technique | Target stage / path | Problem it addresses | Primary benefit |
|---|---|---|---|
| Stage decomposition | Thinker → Talker → Code2Wav | One loop, one batch/graph/device policy; slowest sub-path gates the rest | Independent runtime policy per stage |
| AR + Code2Wav batching | Talker MTP path, Code2Wav async chunks | Single-request micro-work leaves SMs idle at high concurrency | Higher occupancy and req/s |
| CUDA Graph | Thinker / Talker / Code2Wav Decode | Per-step CPU kernel dispatch inflates TPOT; audio RTF stays above real-time | Lower TPOT and audio RTF; ~4× throughput jump in the sweep |
| Async chunk | Thinker→Talker, Talker→Code2Wav | Full-payload barriers delay first audio | Pipelined handoffs; largest audio TTFP cut |
| Async omni output | Thinker connector payloads | Sync payload construction blocks Thinker Decode workers | Throughput recovery without audio TTFP regression |
| Stage replicas | Talker, Code2Wav | Speech stages saturate while Thinker still has headroom | Horizontal scale on bottleneck stages only |
| Hot-path cleanup | Talker code predictor, connector payloads | Per-step Python / alloc / sync scales with utterance length | Lower per-step latency; stacks with layers above |

Sweep: Seed-TTS `en`, `Qwen3-Omni-30B-A3B-Instruct`, **10 / 160 / 320 / 640** prompts at concurrency **1 / 16 / 32 / 64**, **5** warmups, three visible GPUs mapped `0/1/2`. Each row restarted the server with an isolated deploy profile and added one optimization. **Batch** through **Async output**: one stage per GPU (Thinker / Talker / Code2Wav on 0 / 1 / 2, single replica). **Stage replicas**: Thinker on GPU 0; **2×** Talker + **2×** Code2Wav on GPUs 1 and 2. Table is concurrency **64**; charts cover all four levels. GPU SKU, driver, and vLLM/Omni versions are **not** in the post.

| Step | Config added | Talker / Code2Wav replicas | Req/s | Mean audio TTFP | Mean audio RTF |
|---|---|---|---:|---:|---:|
| Baseline | Batch | 1 / 1 | 2.2 | 5884 ms | 1.15 |
| + CUDA Graph | Graph capture on Thinker, Talker, Code2Wav | 1 / 1 | 8.6 (+299%) | 2790 ms (−53%) | 0.59 (−49%) |
| + Async chunk | Async-chunk stage handoffs | 1 / 1 | 9.3 (+8%) | 655 ms (−77%) | 0.63 |
| + Async output | Async omni output path | 1 / 1 | 11.3 (+22%) | 631 ms (−4%) | 0.47 (−25%) |
| + Stage replicas | 2× Talker + 2× Code2Wav | 2 / 2 | 11.7 (+4%) | 632 ms | 0.47 |

**Figure 2.** Stack: staged execution, batching and replicas, CUDA Graph, async chunk, hot-path cleanup. Original caption: performance comes from the staged dataflow, stage runtime, and Decode hot path together.

## Optimization stack, stage by stage

Each step’s numbers assume every layer above is already on.

### 1. Stage decomposition and batching: the baseline

**Why.** Thinker is multimodal AR text; Talker is a codec-predictor AR path; Code2Wav is parallel vocoder Decode. One serving path forces one batching / graph / device policy. Separating stages exposes a second problem: speech still spends GPU time on single-request micro-work. Each Talker step is a short code-predictor forward; each Code2Wav chunk is a small vocoder forward. At concurrency 64, one-request-at-a-time leaves SMs idle and never amortizes the fixed per-step cost.

**Why it works.** Stage boundaries become first-class: connectors carry hidden states, embeddings, codec codes, chunk metadata; the scheduler batches and graphs each stage on its own critical path. Per-stage batching packs concurrent requests into one Talker MTP invocation and one Code2Wav forward.

**What you gain.** Independent `max_num_seqs`, sampling params, connectors, graph/eager policy, optional replicas — prerequisite for everything below. This batched, stage-decomposed config **is** the Batch baseline.

### 2. CUDA Graph: per-stage Decode capture

**Why.** Batching raised occupancy; each Decode step still paid CPU kernel dispatch. Talker may run hundreds of short steps per utterance. At concurrency 64 that launch tax dominated **TPOT** and kept audio RTF above real-time.

**Why it works.** Capture a fixed operator sequence once; replay with minimal CPU. Decode shapes bucket into stable `(batch, seq, frames)` profiles; record at warmup, reuse on the hot path. Each stage has a different capture point; the principle is the same.

**Figure 3.** Thinker and Talker under vLLM’s outer Decode graph; Talker’s inner code predictor is `torch.compile`d (not a second graph); Code2Wav uses an inner `CUDAGraphDecoderWrapper`.

#### Stage 0 — Thinker: vLLM outer decode graph

The Thinker is an autoregressive multimodal stage (`LLM_AR`). When `enforce_eager` is false, it uses vLLM's standard CUDA Graph capture on the Decode path — the same mechanism as text-only serving. That removes repeated CPU-side kernel dispatch during long Thinker generations.

#### Stage 1 — Talker: outer decode graph + compiled code predictor

The Talker also runs through vLLM's outer CUDA Graph path when `enforce_eager: false`. Each Talker Decode step additionally invokes the **code predictor** — a short re-prefill transformer that emits RVQ codec codes. That inner path is optimized separately:

- `torch.compile` fuses the 5-layer predictor (`dynamic=False`, `epilogue_fusion=False`) so RMSNorm/RoPE stay aligned with the reference while cutting kernel count.
- On CUDA the predictor does **not** enable a second manual CUDA Graph by default (`use_cuda_graphs=False`) — it would conflict with Talker’s `CUDAGraphWrapper`. Outer graph + compiled inner forward are complementary: one captures the AR stage loop, the other fuses the codec-prediction micro-forward.
- Optional prefix-graph buckets: `code_predictor_prefix_graphs` in connector config, when explicitly enabled.

#### Stage 2 — Code2Wav: inner vocoder graph

Code2Wav is a generation stage (`LLM_GENERATION`), not an AR loop. Its graph path is an **inner** `CUDAGraphDecoderWrapper` rather than vLLM's outer wrapper:

```python
# Enabled during weight load when stage enforce_eager is false
self.code2wav.enable_cudagraph(
    codec_chunk_frames=chunk_frames,
    codec_left_context_frames=left_frames,
)
```

Shape bucketing from connector: `codec_chunk_frames`, `codec_left_context_frames`. Capture enumerates `(batch, num_quantizers, frames)` buckets that async-chunk and full-payload Decode will hit, including the smaller first chunk from `initial_codec_chunk_frames`. Vocoder warmup: `precompute_snake_caches()` before capture so SnakeBeta activations do not pay setup inside the graph. Async-chunk: `chunked_decode_streaming` → `_cudagraph_wrapper.chunked_decode_with_cudagraph`; full-payload paths use batched decode when shapes match captured buckets.

**What you gain.** All three stages graphed: req/s **2.2 → 8.6** (+299%), mean audio TTFP **5884 → 2790 ms**, mean audio RTF **1.15 → 0.59**. The page attributes most of that win to launch overhead coming off Thinker text generation, Talker codec Decode, and Code2Wav vocoder forwards **together**.

### 3. Async chunk: pipelined inter-stage handoffs

**Why.** Each stage was faster; the pipeline was still **barrier-synchronized**. Talker could not start until Thinker finished; Code2Wav could not emit until Talker had a full payload. First audio tracked full Thinker generation plus full Talker Prefill — even when a few codec frames would have been enough for the first audible chunk.

**Why it works.** Partial handoffs: Thinker emits embedding rows incrementally; Talker slices on `initial_codec_chunk_frames` / `codec_chunk_frames`; the async scheduler overlaps chunk transfer with compute.

**Figure 4.** Barrier path waits for full Thinker + Talker payloads; async chunk overlaps so Code2Wav starts after a few codec frames.

**What you gain.** Largest audio TTFP win: **2790 → 655 ms**.

### 4. Async output: non-blocking payload construction

**Why.** Handoffs are incremental, but **synchronous** payload construction (copy embeddings and hidden states on every chunk boundary) can still stall Thinker Decode workers. GPU time then goes to Python scheduling even though stage handoffs are already incremental.

**Why it works.** `async_omni_output`: Thinker hands Decode state to a non-blocking output path and returns to the next token; the connector assembles chunks asynchronously.

**Figure 5.** Sync payload: GPU idle ~**2.8 ms** between Talker steps. After: inter-step gap ~**41 µs**.

**What you gain.** On top of async chunk at concurrency 64: mean audio TTFP near **631 ms**; mean audio RTF **0.63 → 0.47**. TTFP barely moves; RTF and req/s do (**9.3 → 11.3**).

### 5. Stage replicas: scaling Talker and Code2Wav

**Why.** Thinker generates text once; Talker and Code2Wav then run hundreds of short steps. At concurrency 64 a single speech replica becomes the tail while Thinker still has headroom. Cloning the whole pipeline would duplicate the large multimodal Thinker.

**Why it works.** Replicate only saturating stages. One Thinker on GPU 0 feeds 2× Talker and 2× Code2Wav on GPUs 1 and 2:

```json
{
  "stage_overrides": {
    "1": {"num_replicas": 2, "devices": "1,2"},
    "2": {"num_replicas": 2, "devices": "1,2"}
  }
}
```

**Figure 6.** Async chunk + replicas on the speech side.

**What you gain.** **11.7** req/s at concurrency 64 — sweep peak — audio TTFP ~**632 ms**, RTF ~**0.47**. Replica margin widens as concurrency climbs. The table’s replica row is **+4%** req/s vs async output at c=64; the charts also give **6.8** req/s at c=32.

### 6. Hot-path cleanup: Talker Decode and connector payloads

**Why.** Framework-level bottlenecks are gone. Profiling still shows a tail that **scales with utterance length**: redundant connector traffic, per-step `torch.cat` and CPU serialization, Python dispatch in the codec predictor, D2H of Decode state the next step needs back on GPU.

**Why it works** (audio output unchanged):

- **Decode-only connector handoffs.** Chunk 0 still ships full Thinker Prefill; later steps send only the new `embed.decode` row. Connector traffic **O(1) per step**, not growing with prompt length.
- **Single-GPU executor default.** Drop the implicit `"mp"` default for `distributed_executor_backend` → `uni` executor on single-GPU, skip multiprocess startup / IPC / worker-sync.
- **Connector payload construction.** Per-token Decode embeddings; skip repeated `torch.cat`. Skip downstream pooler/multimodal CPU payloads when the request’s final stage is already local (no hidden-state D2H).
- **Talker code predictor rewrite.** Away from Hugging Face `generate()` on very short sequences. Re-prefill with SDPA, native GQA, inline top-k sampling, cached module references, `torch.compile` on the inner transformer. On CUDA this sits **below** Talker’s CUDA Graph (see §2), not a second conflicting graph.
- **GPU-resident Decode state.** `hidden_states.last`, `hidden_states.trailing_text`, `embed.tts_pad_projected`, `codes.audio` stay in `model_intermediate_buffer`. Talker and Code2Wav skip multimodal `get_mrope_input_positions` (cheap linear positions). `_store_value` skips redundant `.to("cpu")` when already on CPU.
- **Numerical guardrail.** RMSNorm variance and RoPE stay fp32 (`epilogue_fusion=False`); per-call embedding buffers avoid cross-request aliasing.

**What you gain.** Long-context single-request test: E2EL **21.28 s → 7.37 s**, audio TTFP **3197 → 1796 ms**, audio RTF **0.71 → 0.28**. Stacks with layers above; shows up in the DFX perf suite, **not** as a separate sweep row. Prompt length, GPU, and concurrency for that E2EL test are **not** in the post.

## Validation results

Same sweep as the overview table, all four concurrency levels (`1` / `16` / `32` / `64`), from the Batch baseline.

**Figure 7.** req/s: c=1 orange, c=16 purple, c=32 green, c=64 red. Replicas: **11.7** req/s at c=64 and **6.8** at c=32, up from **2.2** (Batch at c=64).

**Figure 8.** Mean audio RTF. Batch at or above real-time under load (up to **1.15** at c=64); async output and replicas keep c=32 / c=64 at or below ~**0.47**.

**Figure 9.** Mean audio TTFP, ms, log scale. Async chunk drops c=64 from ~**5884 ms** (Batch) to ~**655 ms**.

### Reading the sweeps together

- **Throughput (Figure 7).** c=64: **2.2 → 11.7** req/s (~**5.4×**). c=32: **1.1 → 6.8**. Largest jump: CUDA Graph (~**4×**). Async output is the last big push at high concurrency; replicas take the peak.
- **RTF (Figure 8).** **1.15 → 0.47** at c=64 — Decode moves from lagging playback to running ahead.
- **First-packet (Figure 9).** ~**5884 → ~632 ms** at c=64; async chunk is the largest single cut (to ~**655 ms**); later layers hold it.

CUDA Graph and async chunk dominate latency; async output and replicas add high-concurrency throughput. The stacked pipeline is their claim of real-time headroom, not a guarantee on another cluster.

## Acknowledgements

Qwen3-Omni contributors in [vLLM-Omni](https://github.com/vllm-project/vllm-omni): Haiyan Wu, Taichang Zhou, Canlin Guo, Ruirui Yang, Ziming Huang, Wengang Zheng, Lianhao Xu, Han Gao, Junhong Liu, Samit Huang, Hao Chen, Alex Brooks, Chenguang Zheng, Peiqi Yin, Wenjing Chen, Nick Cao, Shunyang Li, Yong Yang, Divyansh Singhvi, Yueqian Lin, Dayu Qiu, Roger Wang, Hongsheng Liu.

## References

**Source and configuration:** [`pipeline.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py); [`qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py); [`stage_input_processors/qwen3_omni.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_input_processors/qwen3_omni.py); deploy [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml); DFX [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json), [`test_qwen3_omni_multi_replicas.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_multi_replicas.json); model [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).

**PRs:** CUDA Graph (§2) Thinker [vllm-omni#523](https://github.com/vllm-project/vllm-omni/pull/523), Talker [#669](https://github.com/vllm-project/vllm-omni/pull/669), Code2Wav [#2376](https://github.com/vllm-project/vllm-omni/pull/2376). Async chunk (§3) cross-stage chunked compute/communication [#727](https://github.com/vllm-project/vllm-omni/pull/727), async scheduling to overlap chunk IO and compute [#951](https://github.com/vllm-project/vllm-omni/pull/951), inter-packet latency [#1656](https://github.com/vllm-project/vllm-omni/pull/1656). Async output (§4) async omni output materialization [#4476](https://github.com/vllm-project/vllm-omni/pull/4476). Stage replicas (§5) multi-stage deployment [#2396](https://github.com/vllm-project/vllm-omni/pull/2396), stage runtime and distributed replica control plane [#3855](https://github.com/vllm-project/vllm-omni/pull/3855). Hot-path (§6) [#3007](https://github.com/vllm-project/vllm-omni/pull/3007), [#3164](https://github.com/vllm-project/vllm-omni/pull/3164), [#3878](https://github.com/vllm-project/vllm-omni/pull/3878).

`#sig-omni` on [vLLM Slack](https://slack.vllm.ai); issues on [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni).
