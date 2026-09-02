---
source: https://vllm.ai/blog/2025-11-13-shm-ipc-cache
lang: zh
voice: literary-study
fetched: 2026-09-01
---

# Shared Memory IPC Cache：大图不要在进程之间复印

英文对照：`en/vllm/blog/serving/shm-ipc.md`  
原文：https://vllm.ai/blog/2025-11-13-shm-ipc-cache  
2025-11-13。Cohere 贡献，先发在他们博客。`optimization.md` 里 `mm_processor_cache_type="shm"` 就是这篇。

V1 是多进程：前端预处理、coordinator 调度、每卡一个 worker。小输入的 IPC 可以忽略；一张 1024×3072 的图在 Command-A Vision 里大约 **9 MB** int8，多图就是几十 MB。多轮再传一遍，税会叠。

旧方案是 **mirrored cache**：发送方和接收方各维护一份顺序相同的副本，命中就假设对面也有、不再传。它要求两边处理顺序完全一致。coordinator 一重排，cache 就对不齐。所以 vLLM 只把它用在前端↔coordinator；多 worker 时 coordinator↔worker 仍走 socket，要序列化、传送、反序列化。单 worker 则和 coordinator 同进程，根本不必 IPC。


本地图（原文版权仍归原站；学习对照用）：

![processes1](../../../../assets/vllm/blog/serving/shm-ipc/01-processes1.png)

![shared memory object store](../../../../assets/vllm/blog/serving/shm-ipc/02-shared_memory_object_store.png)

![processes2](../../../../assets/vllm/blog/serving/shm-ipc/03-processes2.png)

## 共享内存对象库

一份 ring buffer：一个 writer、多个 reader。writer 放入对象、更新地址索引、把地址广播出去；reader 拿地址直接读。不必同序。缓存占用不随 reader 数倍增。空间不够从环头驱逐，用 writer/reader 计数保证还在用的不被赶走：`writer_counter × n_readers == reader_counter` 才驱逐。

前端当 writer，每个 worker 当 reader，大图不必再经 coordinator 复印一份。

## 成绩（Command-A Vision、4×A100 TP4、VisionArena-Chat）

首次请求：prefill 吞吐 **+11.5%**（581→648 tok/s），TTFT **−10.5%**。来自「写一次、工人并发读」，少排队。

KV 和图像都命中时：prefill **+69.9%**（2894→4918），TTFT **−40.5%**（790→471 ms）。IPC 税在这条路径上特别显眼。输入越大、TP 越宽，越值得开。

`mm_processor_cache_type="shm"`。`optimization.md` 提醒：API server 横向扩展会关掉这条 IPC cache（它要求 API 与 engine 一对一），不影响 processor cache。EPD 把编码器拆到另一栋楼；这篇先让同一栋楼里少传同一张图。
