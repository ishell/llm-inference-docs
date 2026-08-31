---
source: https://vllm.ai/blog/2025-11-13-shm-ipc-cache
lang: en
fetched: 2026-09-01
---

# Shared Memory IPC Cache

2025-11-13. Cohere. `mm_processor_cache_type="shm"` in `optimization.md`. Study note.

V1 is multiprocess: frontend, coordinator, per-GPU workers. A 1024×3072 Command-A Vision image is ~**9 MB** int8; multi-image tens of MB. IPC tax compounds across turns.

**Mirrored cache** assumes identical insertion order. Coordinator reordering desyncs it, so vLLM only uses it frontend↔coordinator. Multi-worker coordinator↔worker still sockets. Single worker shares the coordinator process.

**Shared-memory object store:** one writer, many readers, ring buffer, address broadcast. No ordering. Memory does not grow with readers. Evict head when `writer_counter × n_readers == reader_counter`. Frontend writes; each worker reads.

Command-A Vision, 4×A100 TP4, VisionArena-Chat: first request prefill **+11.5%**, TTFT **−10.5%**. Cached KV+image: prefill **+69.9%**, TTFT **−40.5%**. Larger inputs / wider TP help more. API-server scale-out disables this IPC cache (needs 1:1 API↔engine); processor cache stays.
