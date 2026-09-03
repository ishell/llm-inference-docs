#!/usr/bin/env python3
"""Redraw principle diagrams as Chinese study figures.

Skip result charts, CLI screenshots, and source-field architecture sketches.
"""
from studyfig import (
    ACCENT,
    CACHED,
    DECODE,
    GPU,
    INK,
    LINE,
    MUTED,
    PANEL,
    PAPER,
    PREFILL,
    SERVICE,
    WAIT,
    WHITE,
    arrow,
    canvas,
    font_cn,
    font_en,
    footnote,
    round_box,
    save,
    text_at,
    title,
)


def legend(d, items, x, y):
    xx = x
    for fill, label in items:
        round_box(d, [xx, y, xx + 28, y + 28], fill, outline=LINE, radius=6)
        text_at(d, (xx + 40, y + 14), label, font_cn(20), MUTED, "lm")
        xx += 28 + 12 + 20 * len(label) + 36


def mastering_kv():
    W, H = 2200, 980
    im, d = canvas(W, H)
    title(d, W, "KV cache：Prefill 建房，Decode 续住")
    legend(
        d,
        [(PREFILL, "这一步新算"), (CACHED, "从 cache 取")],
        620,
        100,
    )

    # Prefill panel
    round_box(d, [80, 150, 1060, 820], WHITE, radius=16, width=3)
    text_at(d, (570, 190), "Prefill · Step 1", font_en(24, True), ACCENT)
    text_at(d, (570, 230), "整段 prompt 一次算完 Q / K / V", font_cn(22), MUTED)
    boxes = [("Q", 4), ("K", 4), ("V", 4)]
    x = 140
    for name, n in boxes:
        for i in range(n):
            y = 300 + i * 70
            round_box(d, [x, y, x + 200, y + 56], PREFILL, outline=(176, 132, 108), radius=8)
        text_at(d, (x + 100, 620), name, font_en(26, True))
        x += 280
    text_at(d, (570, 700), "(Q × Kᵀ) × V", font_en(24))
    text_at(d, (570, 760), "K / V 写入 cache", font_cn(22), ACCENT)

    # Decode panel
    round_box(d, [1140, 150, 2120, 820], WHITE, radius=16, width=3)
    text_at(d, (1630, 190), "Decode · Step N", font_en(24, True), ACCENT)
    text_at(d, (1630, 230), "只为当前 token 新算一截，其余从 cache 取", font_cn(22), MUTED)
    # Q one row
    round_box(d, [1220, 320, 1420, 376], PREFILL, outline=(176, 132, 108), radius=8)
    text_at(d, (1320, 410), "Q 1×", font_en(22, True))
    # K stacked: 3 cached + 1 new
    for i, fill in enumerate([CACHED, CACHED, CACHED, PREFILL]):
        y = 300 + i * 70
        round_box(d, [1500, y, 1700, y + 56], fill, outline=LINE, radius=8)
    text_at(d, (1600, 620), "K", font_en(26, True))
    for i, fill in enumerate([CACHED, CACHED, CACHED, PREFILL]):
        y = 300 + i * 70
        round_box(d, [1780, y, 1980, y + 56], fill, outline=LINE, radius=8)
    text_at(d, (1880, 620), "V", font_en(26, True))
    text_at(d, (1630, 760), "不重读整本自传，只追加这一页", font_cn(22), ACCENT)

    footnote(d, W, H, "Decode 是 memory-bound：贵的是搬 K/V，不是再做一遍矩阵乘。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/01-kv-cache.png")


def mastering_parallel():
    W, H = 2200, 900
    im, d = canvas(W, H)
    title(d, W, "三种切卡：PP / TP / SP")

    panels = [
        (80, "Pipeline parallelism", "按层切开，激活往下传", "通信少，会有 pipeline bubble"),
        (760, "Tensor parallelism", "每一层都切开", "每层 All-Reduce，NVLink 才划算"),
        (1440, "Sequence parallelism", "LayerNorm 等按序列维切", "省激活显存，常和 TP 叠用"),
    ]
    for x0, en, zh, note in panels:
        round_box(d, [x0, 130, x0 + 640, 780], WHITE, radius=16, width=3)
        text_at(d, (x0 + 320, 180), en, font_en(22, True), ACCENT)
        text_at(d, (x0 + 320, 222), zh, font_cn(24))
        if "Pipeline" in en:
            for i, gpu in enumerate(["GPU 0", "GPU 1", "GPU 2", "GPU 3"]):
                y = 280 + i * 100
                round_box(d, [x0 + 80, y, x0 + 560, y + 80], GPU, outline=GPU, radius=10)
                text_at(d, (x0 + 320, y + 40), f"{gpu}  ·  Layer {i + 1}", font_en(20), WHITE)
            for i in range(3):
                y = 360 + i * 100
                arrow(d, (x0 + 320, y), (x0 + 320, y + 18), width=3, head=10)
        elif "Tensor" in en:
            for i, gpu in enumerate(["GPU 0", "GPU 1"]):
                y = 300 + i * 180
                round_box(d, [x0 + 70, y, x0 + 570, y + 140], GPU, outline=GPU, radius=10)
                text_at(d, (x0 + 320, y + 50), gpu, font_en(22, True), WHITE)
                text_at(d, (x0 + 320, y + 96), "Layer 1…N 的一半", font_cn(20), WAIT)
            text_at(d, (x0 + 320, 680), "All-Reduce", font_en(22, True), ACCENT)
        else:
            round_box(d, [x0 + 80, 300, x0 + 300, 700], PREFILL, outline=LINE, radius=12)
            round_box(d, [x0 + 340, 300, x0 + 560, 700], DECODE, outline=LINE, radius=12)
            text_at(d, (x0 + 190, 500), "seq 前半", font_cn(22), INK)
            text_at(d, (x0 + 450, 500), "seq 后半", font_cn(22), WHITE)
        text_at(d, (x0 + 320, 740), note, font_cn(20), MUTED)

    footnote(d, W, H, "一张卡装得下就别切。节点内有 NVLink 先试 TP；跨节点走廊慢则倾向 PP。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/02-parallelism.png")


def mastering_attention():
    W, H = 2200, 720
    im, d = canvas(W, H)
    title(d, W, "Attention 三副骨架：头怎么分 K/V")
    rows = [
        ("MHA", "每个 Q 头有自己的 K/V", "质量最好，KV cache 最肥"),
        ("GQA", "若干 Q 头共用一组 K/V", "Llama 2 70B 走这条路"),
        ("MQA", "所有 Q 头共用一份 K/V", "最瘦，memory-bound 时更划算"),
    ]
    y = 160
    for code, zh, note in rows:
        round_box(d, [120, y, 2080, y + 140], WHITE, radius=14, width=3)
        round_box(d, [160, y + 30, 420, y + 110], SERVICE, outline=SERVICE, radius=10)
        text_at(d, (290, y + 70), code, font_en(28, True), WHITE)
        text_at(d, (760, y + 70), zh, font_cn(28), INK, "lm")
        text_at(d, (1960, y + 70), note, font_cn(24), MUTED, "rm")
        y += 160
    footnote(d, W, H, "数学仍是 scaled dot-product。变的是从显存读多少 K/V。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/03-attention-kv.png")


def mastering_flash():
    W, H = 2200, 780
    im, d = canvas(W, H)
    title(d, W, "FlashAttention：在 SRAM 里把一小块算完")
    round_box(d, [120, 160, 980, 620], SERVICE, outline=SERVICE, radius=16)
    text_at(d, (550, 210), "HBM（大、慢）", font_cn(28), WHITE)
    text_at(d, (550, 270), "Q  K  V  中间表", font_en(24), WAIT)
    text_at(d, (550, 420), "整表分步写回，会把走廊走肿", font_cn(24), WAIT)
    arrow(d, (1000, 390), (1180, 390), width=6, head=16)
    round_box(d, [1220, 160, 2080, 620], PREFILL, outline=(176, 132, 108), radius=16)
    text_at(d, (1650, 210), "SRAM（小、快）", font_cn(28), INK)
    for i, lab in enumerate(["tile Q", "tile K", "tile V"]):
        x = 1320 + i * 230
        round_box(d, [x, 280, x + 200, 400], WHITE, radius=10)
        text_at(d, (x + 100, 340), lab, font_en(22, True), ACCENT)
    text_at(d, (1650, 500), "一块算完，只写回最终 O", font_cn(24), INK)
    text_at(d, (1650, 560), "不改数学，改搬家顺序", font_cn(24), MUTED)
    footnote(d, W, H, "exact attention。已训练的模型也可以换上去。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/04-flash-attention.png")


def mastering_paged():
    W, H = 2200, 860
    im, d = canvas(W, H)
    title(d, W, "连续预留 vs PagedAttention")
    round_box(d, [80, 140, 1060, 720], WHITE, radius=16, width=3)
    text_at(d, (570, 190), "连续分配", font_cn(28, True), ACCENT)
    text_at(d, (570, 236), "人人按 max_seq_len 租一整条走廊", font_cn(22), MUTED)
    for i in range(3):
        y = 290 + i * 120
        round_box(d, [140, y, 980, y + 90], PANEL, outline=LINE, radius=8)
        round_box(d, [160, y + 16, 420, y + 74], DECODE, outline=LINE, radius=6)
        text_at(d, (700, y + 45), "空着也付钱", font_cn(22), MUTED)
    round_box(d, [1140, 140, 2120, 720], WHITE, radius=16, width=3)
    text_at(d, (1630, 190), "按页出租", font_cn(28, True), ACCENT)
    text_at(d, (1630, 236), "逻辑连续，物理可以东一块西一块", font_cn(22), MUTED)
    # scattered blocks
    coords = [(1220, 300), (1480, 300), (1740, 300), (1220, 430), (1600, 430), (1860, 430), (1360, 560), (1740, 560)]
    for i, (x, y) in enumerate(coords):
        fill = DECODE if i < 5 else PREFILL
        round_box(d, [x, y, x + 200, y + 80], fill, outline=LINE, radius=8)
        text_at(d, (x + 100, y + 40), f"block {i + 1}", font_en(18), WHITE if fill == DECODE else INK)
    footnote(d, W, H, "浪费几乎只剩最后一页没填满。页表让并行采样还能共用 prompt 的那几页。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/05-paged-kv.png")


def mastering_spec():
    W, H = 2200, 720
    im, d = canvas(W, H)
    title(d, W, "Speculative decoding：小模型起草，大模型一次验收")
    round_box(d, [120, 160, 700, 520], WAIT, outline=LINE, radius=14)
    text_at(d, (410, 220), "draft", font_en(24, True), ACCENT)
    text_at(d, (410, 280), "便宜地猜 k 个字", font_cn(24))
    for i, t in enumerate(["t1", "t2", "t3", "t4"]):
        x = 180 + i * 120
        round_box(d, [x, 360, x + 100, 460], WHITE, radius=8)
        text_at(d, (x + 50, 410), t, font_en(20, True), ACCENT)
    arrow(d, (720, 340), (900, 340), width=6, head=16)
    round_box(d, [920, 160, 1600, 520], PREFILL, outline=(176, 132, 108), radius=14)
    text_at(d, (1260, 220), "大模型一次验证", font_cn(26, True))
    text_at(d, (1260, 280), "从左到右接受或拒绝", font_cn(22), MUTED)
    round_box(d, [980, 360, 1280, 460], DECODE, outline=LINE, radius=8)
    text_at(d, (1130, 410), "接受 t1 t2", font_cn(22), WHITE)
    round_box(d, [1320, 360, 1540, 460], SERVICE, outline=SERVICE, radius=8)
    text_at(d, (1430, 410), "丢掉 t3…", font_cn(22), WHITE)
    arrow(d, (1620, 340), (1780, 340), width=6, head=16)
    round_box(d, [1800, 220, 2080, 460], WHITE, outline=ACCENT, radius=14, width=3)
    text_at(d, (1940, 300), "分布仍等于", font_cn(22))
    text_at(d, (1940, 360), "只从大模型采样", font_cn(22), ACCENT)
    footnote(d, W, H, "草稿可以是小模型、n-gram、EAGLE、Medusa。统计上诚实，工程上可能更快。")
    save(im, "assets/nvidia/performance-tuning/mastering-llm-techniques/zh/06-speculative.png")


def trtllm_pp():
    W, H = 2200, 700
    im, d = canvas(W, H)
    title(d, W, "Pipeline parallelism：每张卡住一段层")
    round_box(d, [120, 200, 1040, 520], GPU, outline=GPU, radius=16)
    text_at(d, (580, 250), "GPU 0", font_en(26, True), WHITE)
    for i, lab in enumerate(["Layer 1", "Layer 2", "Layer 3"]):
        x = 180 + i * 280
        round_box(d, [x, 320, x + 240, 460], PREFILL, outline=LINE, radius=10)
        text_at(d, (x + 120, 390), lab, font_en(22, True), INK)
    arrow(d, (1060, 360), (1140, 360), width=8, head=18)
    text_at(d, (1100, 300), "Send", font_en(20, True), ACCENT)
    round_box(d, [1160, 200, 2080, 520], GPU, outline=GPU, radius=16)
    text_at(d, (1620, 250), "GPU 1", font_en(26, True), WHITE)
    for i, lab in enumerate(["Layer 4", "Layer 5", "Layer 6"]):
        x = 1220 + i * 280
        round_box(d, [x, 320, x + 240, 460], DECODE, outline=LINE, radius=10)
        text_at(d, (x + 120, 390), lab, font_en(22, True), WHITE)
    footnote(d, W, H, "通信很少：做完自己那一层楼，把输出递给下一张卡。跨节点走廊慢时更合适。")
    save(im, "assets/nvidia/performance-tuning/trtllm-sharding/zh/01-pipeline.png")


def trtllm_tp():
    W, H = 2200, 700
    im, d = canvas(W, H)
    title(d, W, "Tensor parallelism：每一层都切开")
    for i, gpu in enumerate(["GPU 0", "GPU 1"]):
        x = 120 + i * 1040
        round_box(d, [x, 180, x + 960, 540], GPU, outline=GPU, radius=16)
        text_at(d, (x + 480, 230), gpu, font_en(26, True), WHITE)
        for j, lab in enumerate(["L1", "L2", "L3", "L4", "L5", "L6"]):
            xx = x + 40 + j * 150
            round_box(d, [xx, 300, xx + 130, 460], PREFILL if i == 0 else DECODE, outline=LINE, radius=8)
            text_at(d, (xx + 65, 360), lab, font_en(20, True), INK if i == 0 else WHITE)
            text_at(d, (xx + 65, 410), "½", font_en(18), MUTED if i == 0 else WAIT)
    text_at(d, (W / 2, 600), "每层结束要 All-Reduce：下一层需要完整输出", font_cn(24), ACCENT)
    footnote(d, W, H, "矩阵乘更小、算得更快；税是通信。节点内有 NVLink，往往先试这条。")
    save(im, "assets/nvidia/performance-tuning/trtllm-sharding/zh/02-tensor.png")


def trtllm_knobs():
    W, H = 2200, 720
    im, d = canvas(W, H)
    title(d, W, "三个尺寸旋钮")
    knobs = [
        ("max_batch_size", "同时几个人在场", "编译时留够，运行时还能再拧"),
        ("max_num_tokens", "这一拍最多打包多少 token", "去 padding 后的预算，决定 workspace"),
        ("max_seq_len", "一个人最长能住多久", "屋顶，不是地板"),
    ]
    for i, (code, zh, note) in enumerate(knobs):
        x = 80 + i * 700
        round_box(d, [x, 160, x + 660, 560], WHITE, radius=16, width=3)
        round_box(d, [x + 40, 200, x + 620, 300], SERVICE, outline=SERVICE, radius=12)
        text_at(d, (x + 330, 250), code, font_en(26, True), WHITE)
        text_at(d, (x + 330, 380), zh, font_cn(28))
        text_at(d, (x + 330, 460), note, font_cn(22), MUTED)
    footnote(d, W, H, "官方玩具数字：max_batch_size = 4，max_num_tokens = 12。按最长 prompt 去设 token 预算，等于拿显存开玩笑。")
    save(im, "assets/nvidia/performance-tuning/trtllm-paged-attention-ifb/zh/01-three-knobs.png")


def _ifb_row(d, x, y, cells, labels=None):
    for i, fill in enumerate(cells):
        box = [x + i * 70, y, x + i * 70 + 62, y + 54]
        if fill is None:
            round_box(d, box, PAPER, outline=LINE, radius=6)
        else:
            round_box(d, box, fill, outline=LINE, radius=6)
            if labels and labels[i]:
                text_at(d, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), labels[i], font_en(16, True), INK)


def trtllm_ifb_wait():
    W, H = 2200, 820
    im, d = canvas(W, H)
    title(d, W, "调度还没开张：请求在门外")
    legend(d, [(WAIT, "未调度 prompt"), (PAPER, "引擎空位")], 700, 100)
    text_at(d, (400, 180), "未调度", font_cn(24, True), MUTED)
    for i in range(5):
        y = 220 + i * 80
        _ifb_row(d, 120, y, [WAIT] * (3 + (i % 3)))
        text_at(d, (80, y + 27), f"R{i+1}", font_en(20, True), MUTED, "rm")
    round_box(d, [1100, 180, 2080, 700], WHITE, radius=16, width=3)
    text_at(d, (1590, 230), "Engine · Paged KV", font_en(22, True), ACCENT)
    text_at(d, (1590, 280), "max_batch_size = 4   max_num_tokens = 12", font_en(18), MUTED)
    for r in range(4):
        _ifb_row(d, 1200, 340 + r * 80, [None] * 12)
    footnote(d, W, H, "方块只是为了好看，不是真实显存布局。颜色是请求，不是物理地址。")
    save(im, "assets/nvidia/performance-tuning/trtllm-paged-attention-ifb/zh/02-ifb-waiting.png")


def trtllm_ifb_mix():
    W, H = 2200, 900
    im, d = canvas(W, H)
    title(d, W, "IFB：Prefill 和 Decode 挤在同一拍")
    legend(
        d,
        [(PREFILL, "本拍 Prefill / C"), (DECODE, "本拍 Decode / G"), (CACHED, "已在 KV")],
        480,
        100,
    )
    # rows R1-R4 in engine, R5 waiting
    rows = [
        ("R1", [CACHED] * 5 + [DECODE] + [None] * 6, ["C"] * 5 + ["G1"] + [""] * 6),
        ("R2", [CACHED] * 5 + [DECODE] + [None] * 6, ["C"] * 5 + ["G1"] + [""] * 6),
        ("R3", [PREFILL] * 5 + [None] * 7, ["C"] * 5 + [""] * 7),
        ("R4", [PREFILL] * 4 + [None] * 8, ["C"] * 4 + [""] * 8),
    ]
    text_at(d, (200, 170), "本拍 token 预算 ≈ 2G + 9C < 12", font_cn(22), MUTED, "lm")
    for i, (name, cells, labs) in enumerate(rows):
        y = 230 + i * 100
        text_at(d, (100, y + 27), name, font_en(22, True), INK, "rm")
        _ifb_row(d, 140, y, cells, labs)
    text_at(d, (100, 650), "R5", font_en(22, True), MUTED, "rm")
    _ifb_row(d, 140, 650, [WAIT] * 6)
    text_at(d, (700, 677), "token 预算还够，但 batch 已经到 4，进不来", font_cn(22), MUTED, "lm")
    footnote(d, W, H, "调度器优先排 generation。短的 G 只占极少预算，空位给新人做 Prefill。")
    save(im, "assets/nvidia/performance-tuning/trtllm-paged-attention-ifb/zh/03-ifb-inflight.png")


def trtllm_ifb_chunk():
    W, H = 2200, 780
    im, d = canvas(W, H)
    title(d, W, "Chunked context：长 prompt 切块，和 Decode 同拍")
    text_at(d, (W / 2, 120), "不开 chunking：prompt > 剩下的 token 预算 → 整个人在门外罚站", font_cn(24), MUTED)
    round_box(d, [80, 170, 1060, 620], WHITE, radius=16, width=3)
    text_at(d, (570, 220), "一次吃完整段", font_cn(26, True), ACCENT)
    _ifb_row(d, 160, 320, [WAIT] * 12)
    text_at(d, (570, 300), "R长 · 16 token prompt，预算只剩 8", font_cn(22), MUTED)
    text_at(d, (570, 460), "进不来。短 Decode 也只好看着空位发呆。", font_cn(22), MUTED)
    round_box(d, [1140, 170, 2120, 620], WHITE, radius=16, width=3)
    text_at(d, (1630, 220), "切成块", font_cn(26, True), ACCENT)
    text_at(d, (1630, 300), "R长 这一拍只吃 8 个 C", font_cn(22), MUTED)
    _ifb_row(d, 1220, 340, [PREFILL] * 8 + [None] * 4, ["C"] * 8 + [""] * 4)
    _ifb_row(d, 1220, 430, [DECODE] * 2 + [None] * 10, ["G"] * 2 + [""] * 10)
    text_at(d, (1630, 540), "同一拍：别人的 G + 长客的一块 C", font_cn(22), MUTED)
    footnote(d, W, H, "除最后一块外，chunk 大小须是 KV block 的整数倍。短 prompt 可能略慢——它们不再那么幸运。")
    save(im, "assets/nvidia/performance-tuning/trtllm-paged-attention-ifb/zh/04-chunked.png")


def anatomy_chunked():
    W, H = 2200, 700
    im, d = canvas(W, H)
    title(d, W, "Chunked prefill：长客人不要独占一整拍")
    # time axis
    d.line([(160, 560), (2040, 560)], fill=INK, width=3)
    text_at(d, (2040, 600), "时间", font_cn(20), MUTED, "rm")
    # long prefill without chunk: one fat bar
    text_at(d, (200, 160), "一次吃完", font_cn(24, True), MUTED, "lm")
    round_box(d, [200, 190, 1400, 270], SERVICE, outline=SERVICE, radius=10)
    text_at(d, (800, 230), "长 prompt 独占 engine step", font_cn(22), WHITE)
    text_at(d, (1600, 230), "别人的 TTFT 被按在地板上", font_cn(22), ACCENT, "lm")
    text_at(d, (200, 340), "切成块", font_cn(24, True), ACCENT, "lm")
    xs = [(200, 480), (520, 800), (840, 1120)]
    for i, (a, b) in enumerate(xs):
        round_box(d, [a, 370, b, 450], PREFILL, outline=LINE, radius=10)
        text_at(d, ((a + b) / 2, 410), f"chunk {i+1}", font_en(20, True), INK)
    round_box(d, [500, 470, 620, 530], DECODE, outline=LINE, radius=8)
    round_box(d, [820, 470, 940, 530], DECODE, outline=LINE, radius=8)
    text_at(d, (560, 500), "别人的 G", font_cn(18), WHITE)
    text_at(d, (880, 500), "别人的 G", font_cn(18), WHITE)
    footnote(d, W, H, "只在最后一块才采样新 token。V1 里 prompt 超过 token 预算时，即使你没设，也会被截成 chunked prefill。")
    save(im, "assets/vllm/blog/architecture/anatomy/zh/01-chunked-prefill.png")


def anatomy_prefix():
    W, H = 2200, 780
    im, d = canvas(W, H)
    title(d, W, "Prefix cache：同一本手册，不必每人重读")
    text_at(d, (W / 2, 120), "system prompt 超过一个 KV block（默认 16）才能按块缓存", font_cn(22), MUTED)
    # shared blocks
    for i in range(4):
        x = 200 + i * 160
        round_box(d, [x, 200, x + 140, 320], DECODE, outline=LINE, radius=10)
        text_at(d, (x + 70, 260), f"P{i+1}", font_en(20, True), WHITE)
    text_at(d, (520, 360), "共享前缀（引用计数 +1）", font_cn(22), ACCENT)
    # three unique suffixes
    for i, (lab, col) in enumerate([("问 A", PREFILL), ("问 B", WAIT), ("问 C", CACHED)]):
        y = 420 + i * 80
        arrow(d, (840, 260), (980, y + 24), width=3, head=10)
        round_box(d, [1000, y, 1280, y + 56], col, outline=LINE, radius=8)
        text_at(d, (1140, y + 28), lab, font_cn(22), INK)
        round_box(d, [1320, y, 1600, y + 56], WHITE, outline=LINE, radius=8)
        text_at(d, (1460, y + 28), "各自的尾巴", font_cn(20), MUTED)
    footnote(d, W, H, "只加速 prefill，不加速 decode。对不齐 block 边界的尾巴必须重算。")
    save(im, "assets/vllm/blog/architecture/anatomy/zh/02-prefix-cache.png")


def anatomy_spec():
    W, H = 2200, 700
    im, d = canvas(W, H)
    title(d, W, "vLLM 里的投机解码")
    steps = [
        ("1", "drafter 提案", "n-gram / EAGLE / Medusa 猜 k 个"),
        ("2", "大模型一遍", "在 context + draft 上 forward"),
        ("3", "rejection sampler", "从左到右留下或切开"),
    ]
    for i, (n, h, s) in enumerate(steps):
        x = 120 + i * 680
        round_box(d, [x, 180, x + 620, 520], WHITE, radius=16, width=3)
        round_box(d, [x + 40, 220, x + 120, 300], SERVICE, outline=SERVICE, radius=40)
        text_at(d, (x + 80, 260), n, font_en(28, True), WHITE)
        text_at(d, (x + 310, 340), h, font_cn(28, True))
        text_at(d, (x + 310, 420), s, font_cn(22), MUTED)
        if i < 2:
            arrow(d, (x + 640, 350), (x + 670, 350), width=5, head=14)
    footnote(d, W, H, "期望上，收下的序列分布仍等于只从大模型采样。")
    save(im, "assets/vllm/blog/architecture/anatomy/zh/03-spec-decode.png")


def anatomy_pd():
    W, H = 2200, 720
    im, d = canvas(W, H)
    title(d, W, "Prefill / Decode 分离：两只手分别按住 TTFT 和 ITL")
    round_box(d, [120, 180, 900, 560], PREFILL, outline=(176, 132, 108), radius=16)
    text_at(d, (510, 240), "Prefill 实例 × N", font_cn(28, True), INK)
    text_at(d, (510, 320), "吃算力", font_cn(24), MUTED)
    text_at(d, (510, 400), "把 KV 交给 KV 服务", font_cn(24), INK)
    text_at(d, (510, 480), "按实时请求伸缩", font_cn(22), MUTED)
    round_box(d, [980, 280, 1220, 440], SERVICE, outline=SERVICE, radius=14)
    text_at(d, (1100, 330), "KV", font_en(24, True), WHITE)
    text_at(d, (1100, 390), "store / fetch", font_en(18), WAIT)
    arrow(d, (920, 360), (970, 360), width=6, head=14)
    arrow(d, (1230, 360), (1280, 360), width=6, head=14)
    round_box(d, [1300, 180, 2080, 560], DECODE, outline=(86, 104, 84), radius=16)
    text_at(d, (1690, 240), "Decode 实例 × M", font_cn(28, True), WHITE)
    text_at(d, (1690, 320), "吃带宽", font_cn(24), WAIT)
    text_at(d, (1690, 400), "只在请求第一步拉外部 KV", font_cn(24), WHITE)
    text_at(d, (1690, 480), "之后走本地", font_cn(22), WAIT)
    footnote(d, W, H, "长而爆发的 prefill，不再踩着对延迟敏感的 decode 的脚。")
    save(im, "assets/vllm/blog/architecture/anatomy/zh/04-pd-disagg.png")


def anatomy_roofline():
    W, H = 2200, 900
    im, d = canvas(W, H)
    title(d, W, "Roofline：先被带宽按住，再被算力按住")
    ox, oy = 220, 760
    d.line([(ox, 160), (ox, oy)], fill=INK, width=3)
    d.line([(ox, oy), (2000, oy)], fill=INK, width=3)
    text_at(d, (ox - 90, 200), "perf", font_en(20), MUTED, "rm")
    text_at(d, (ox - 90, 240), "FLOPs/s", font_en(18), MUTED, "rm")
    text_at(d, (1900, oy + 40), "arithmetic intensity  FLOPs/byte", font_en(18), MUTED, "rm")
    # roof: diagonal then plateau
    knee = (1100, 280)
    d.line([(ox + 40, oy - 40), knee, (1900, 280)], fill=INK, width=5)
    text_at(d, (1500, 220), "达不到", font_cn(22), MUTED)
    # mem bw
    d.line([(ox + 80, oy - 80), (1000, 340)], fill=ACCENT, width=4)
    text_at(d, (520, 500), "mem bw bound", font_en(20, True), ACCENT)
    text_at(d, (520, 540), "带宽卡住", font_cn(22), ACCENT)
    # compute
    d.line([(1200, 340), (1900, 340)], fill=DECODE, width=4)
    text_at(d, (1550, 380), "compute bound", font_en(20, True), DECODE)
    text_at(d, (1550, 420), "算力卡住", font_cn(22), DECODE)
    text_at(d, (900, 640), "次优区：kernel 形状一变，实际往往掉到屋顶下面", font_cn(22), MUTED)
    footnote(d, W, H, "低于饱和 batch，步时被 HBM 按住，算 1 个和 10 个 token 可能差不多久。")
    save(im, "assets/vllm/blog/architecture/anatomy/zh/05-roofline.png")


def paged_os():
    W, H = 2200, 820
    im, d = canvas(W, H)
    title(d, W, "PagedAttention：操作系统的分页搬进注意力")
    # table
    rows = [
        ("OS", "PagedAttention"),
        ("页", "KV block（固定数量 token 的 K/V）"),
        ("字节", "token"),
        ("进程", "一条序列"),
        ("页表", "block table：逻辑块 → 物理块"),
    ]
    y = 140
    for i, (a, b) in enumerate(rows):
        fill = SERVICE if i == 0 else WHITE
        ink = WHITE if i == 0 else INK
        round_box(d, [160, y, 700, y + 90], fill, outline=LINE, radius=8)
        round_box(d, [720, y, 2040, y + 90], fill, outline=LINE, radius=8)
        text_at(d, (430, y + 45), a, font_cn(26, True) if i == 0 else font_cn(24), ink)
        text_at(d, (1380, y + 45), b, font_en(24, True) if i == 0 else font_cn(24), ink)
        y += 100
    footnote(d, W, H, "浪费几乎只发生在最后一页没填满的地方。并行采样时，好几路输出可以指向同一物理块。")
    save(im, "assets/vllm/blog/architecture/paged-attention/zh/01-os-metaphor.png")


def nim04_tradeoff():
    W, H = 2200, 860
    im, d = canvas(W, H)
    title(d, W, "扫 concurrency：系统更热闹，每一个人更慢")
    ox, oy = 280, 720
    d.line([(ox, 140), (ox, oy)], fill=INK, width=3)
    d.line([(ox, oy), (2000, oy)], fill=INK, width=3)
    text_at(d, (ox - 20, 170), "系统 TPS", font_cn(20), MUTED, "rm")
    text_at(d, (1850, oy + 40), "单用户 TTFT →", font_cn(20), MUTED)
    pts = [(0.08, 0.18), (0.16, 0.32), (0.28, 0.48), (0.42, 0.62), (0.58, 0.74), (0.78, 0.84), (0.95, 0.90)]
    labels = ["1", "2", "8", "32", "64", "128", "饱和"]
    coords = []
    for nx, ny in pts:
        x = ox + nx * 1600
        y = oy - ny * 520
        coords.append((x, y))
    for a, b in zip(coords, coords[1:]):
        d.line([a, b], fill=ACCENT, width=5)
    for (x, y), lab in zip(coords, labels):
        d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=ACCENT)
        text_at(d, (x, y - 28), lab, font_en(18, True), INK)
    footnote(d, W, H, "示意，不是某次实测。数字在你的卡、模型和 ISL/OSL 上才会长出真正的弯。")
    save(im, "assets/nvidia/benchmarking/nim-04-aiperf/zh/01-concurrency-tradeoff.png")


def main():
    mastering_kv()
    mastering_parallel()
    mastering_attention()
    mastering_flash()
    mastering_paged()
    mastering_spec()
    trtllm_pp()
    trtllm_tp()
    trtllm_knobs()
    trtllm_ifb_wait()
    trtllm_ifb_mix()
    trtllm_ifb_chunk()
    anatomy_chunked()
    anatomy_prefix()
    anatomy_spec()
    anatomy_pd()
    anatomy_roofline()
    paged_os()
    nim04_tradeoff()


if __name__ == "__main__":
    main()
