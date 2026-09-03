#!/usr/bin/env python3
"""Redraw blog-01 principle diagrams as Chinese study figures.

Metric names stay English. Explanatory labels are Chinese.
Output: assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh/
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/nvidia/benchmarking/blog-01-fundamental-concepts/zh"

PAPER = (244, 239, 230)
INK = (28, 25, 23)
MUTED = (87, 83, 78)
ACCENT = (154, 52, 18)
PANEL = (235, 228, 216)
LINE = (214, 208, 196)
WHITE = (255, 252, 247)
PREFILL = (219, 186, 164)
DECODE = (122, 140, 118)
SERVICE = (69, 62, 55)
USER = (92, 83, 74)
TOKEN = (154, 52, 18)

HIRAGINO = "/System/Library/Fonts/Hiragino Sans GB.ttc"
HEITI = "/System/Library/Fonts/STHeiti Medium.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"


def font_cn(size, bold=False):
    try:
        return ImageFont.truetype(HIRAGINO, size, index=2 if bold else 0)
    except OSError:
        return ImageFont.truetype(HEITI, size, index=1)


def font_en(size, bold=False):
    return ImageFont.truetype(MENLO, size, index=1 if bold else 0)


def measure(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1], b[0], b[1]


def text_at(draw, xy, text, font, fill=INK, anchor="mm"):
    x, y = xy
    w, h, ox, oy = measure(draw, text, font)
    ax, ay = anchor
    if ax == "m":
        x -= w / 2
    elif ax == "r":
        x -= w
    if ay == "m":
        y -= h / 2
    elif ay == "b":
        y -= h
    draw.text((x - ox, y - oy), text, font=font, fill=fill)


def round_box(draw, box, fill, outline=LINE, radius=10, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, a, b, fill=ACCENT, width=4, head=12):
    x1, y1 = a
    x2, y2 = b
    draw.line([a, b], fill=fill, width=width)
    from math import atan2, cos, sin

    ang = atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * cos(ang - 0.45), y2 - head * sin(ang - 0.45)),
        (x2 - head * cos(ang + 0.45), y2 - head * sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def dim_h(draw, x1, x2, y, label, font, fill=INK, tick=8):
    if x2 < x1:
        x1, x2 = x2, x1
    draw.line([(x1, y - tick), (x1, y + tick)], fill=fill, width=2)
    draw.line([(x2, y - tick), (x2, y + tick)], fill=fill, width=2)
    draw.line([(x1, y), (x2, y)], fill=fill, width=2)
    text_at(draw, ((x1 + x2) / 2, y - 16), label, font, fill=fill, anchor="mm")


def canvas(w, h):
    im = Image.new("RGB", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, w - 1, h - 1], outline=LINE, width=2)
    return im, draw


def save(im, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    im.save(path, "PNG", optimize=True)
    print(path.relative_to(ROOT), im.size)


def fig1_metrics():
    W, H = 2200, 980
    im, d = canvas(W, H)
    title = font_cn(40, True)
    cn = font_cn(28)
    en = font_en(26, True)
    small = font_cn(24)

    text_at(d, (W / 2, 56), "一次请求上的三把尺子", title, anchor="mm")

    sx0, sx1 = 140, 2060
    service_y = (150, 230)
    user_y = (690, 770)
    round_box(d, [sx0, service_y[0], sx1, service_y[1]], SERVICE, outline=SERVICE, radius=14)
    round_box(d, [sx0, user_y[0], sx1, user_y[1]], USER, outline=USER, radius=14)
    text_at(d, ((sx0 + sx1) / 2, (service_y[0] + service_y[1]) / 2), "推理服务", cn, WHITE)
    text_at(d, ((sx0 + sx1) / 2, (user_y[0] + user_y[1]) / 2), "用户", cn, WHITE)

    q_x = 380
    t_xs = [980, 1380, 1780]
    arrow(d, (q_x, user_y[0] - 4), (q_x, service_y[1] + 6), width=8, head=18)
    text_at(d, (q_x - 70, (service_y[1] + user_y[0]) / 2), "请求", cn, ACCENT, "rm")

    for i, x in enumerate(t_xs, 1):
        text_at(d, (x, service_y[1] + 34), f"Token {i}", en, ACCENT, "mm")
        arrow(d, (x, service_y[1] + 58), (x, user_y[0] + 2), width=8, head=18)

    mid = 520
    dim_h(d, q_x, t_xs[0], mid, "TTFT", en)
    dim_h(d, t_xs[0], t_xs[1], 400, "ITL", en)
    dim_h(d, t_xs[1], t_xs[2], 400, "ITL", en)

    gy = 860
    d.line([(t_xs[0], user_y[1] + 18), (t_xs[0], gy)], fill=INK, width=2)
    d.line([(t_xs[2], user_y[1] + 18), (t_xs[2], gy)], fill=INK, width=2)
    d.line([(t_xs[0], gy), (t_xs[2], gy)], fill=INK, width=2)
    text_at(d, ((t_xs[0] + t_xs[2]) / 2, gy + 28), "generation_time", en, INK)

    text_at(
        d,
        (W / 2, H - 36),
        "TTFT：提交 → 第一个非空 token。ITL：相邻输出之间。GenAI-Perf 的 ITL 不含 TTFT。",
        small,
        MUTED,
    )
    save(im, "01-ttft-itl-generation.png")


def _pipeline(d, *, prefill_w, n_decode, decode_w, x0, y0, h, gap=18):
    cn = font_cn(26)
    en = font_en(22)
    small = font_cn(22)
    boxes = []

    x = x0
    tok = (x, y0, x + 200, y0 + h)
    round_box(d, tok, (118, 92, 110), outline=(118, 92, 110), radius=12)
    text_at(d, ((tok[0] + tok[2]) / 2, (tok[1] + tok[3]) / 2), "分词", cn, WHITE)
    boxes.append(("tok", tok))
    x = tok[2] + gap
    arrow(d, (tok[2] + 2, y0 + h / 2), (x - 2, y0 + h / 2), width=5, head=12)

    inner_h = h - 36
    inner_y = y0 + 18
    model_w = 36 + prefill_w + 16 + n_decode * (decode_w + 10) + 20
    model = (x, y0, x + model_w, y0 + h)
    round_box(d, model, WHITE, outline=LINE, radius=16, width=3)
    text_at(d, ((model[0] + model[2]) / 2, y0 - 22), "模型", small, MUTED)

    px = x + 22
    pre = (px, inner_y, px + prefill_w, inner_y + inner_h)
    round_box(d, pre, PREFILL, outline=(176, 132, 108), radius=12)
    text_at(d, ((pre[0] + pre[2]) / 2, (pre[1] + pre[3]) / 2 - 14), "Prefill", en, INK)
    text_at(d, ((pre[0] + pre[2]) / 2, (pre[1] + pre[3]) / 2 + 18), "整段 prompt", small, MUTED)
    boxes.append(("prefill", pre))

    dx = pre[2] + 16
    decode_boxes = []
    for i in range(n_decode):
        b = (dx, inner_y + 8, dx + decode_w, inner_y + inner_h - 8)
        round_box(d, b, DECODE, outline=(86, 104, 84), radius=8)
        decode_boxes.append(b)
        dx = b[2] + 10
    boxes.append(("decode", decode_boxes))
    if decode_boxes:
        cx = (decode_boxes[0][0] + decode_boxes[-1][2]) / 2
        text_at(d, (cx, model[3] + 26), "Decode", en, MUTED)

    x = model[2] + gap
    arrow(d, (model[2] + 2, y0 + h / 2), (x - 2, y0 + h / 2), width=5, head=12)
    det = (x, y0, x + 220, y0 + h)
    round_box(d, det, (118, 92, 110), outline=(118, 92, 110), radius=12)
    text_at(d, ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2), "反分词", cn, WHITE)
    boxes.append(("detok", det))
    return boxes, det


def fig2_first_token():
    W, H = 2200, 720
    im, d = canvas(W, H)
    text_at(d, (W / 2, 56), "走到第一个输出 token", font_cn(40, True))
    boxes, det = _pipeline(d, prefill_w=520, n_decode=1, decode_w=70, x0=160, y0=200, h=260)
    x = det[2] + 24
    arrow(d, (det[2] + 2, 330), (x - 2, 330), width=5, head=12)
    out = (x, 200, x + 280, 460)
    round_box(d, out, WHITE, outline=ACCENT, radius=12, width=3)
    text_at(d, ((out[0] + out[2]) / 2, 300), "第一个", font_cn(26), ACCENT)
    text_at(d, ((out[0] + out[2]) / 2, 348), "输出 token", font_cn(26), ACCENT)
    text_at(
        d,
        (W / 2, 620),
        "TTFT ≈ 排队 + prefill + 网络。Prompt 越长，KV cache 越大，这一口等待越久。",
        font_cn(24),
        MUTED,
    )
    save(im, "02-first-token.png")


def fig3_e2e():
    W, H = 2200, 780
    im, d = canvas(W, H)
    text_at(d, (W / 2, 56), "e2e_latency：从发出到收完", font_cn(40, True))
    boxes, det = _pipeline(d, prefill_w=360, n_decode=6, decode_w=52, x0=90, y0=200, h=250)
    x0 = boxes[0][1][0]
    x1 = det[2]
    y = 560
    d.line([(x0, 470), (x0, y)], fill=INK, width=2)
    d.line([(x1, 470), (x1, y)], fill=INK, width=2)
    d.line([(x0, y), (x1, y)], fill=INK, width=2)
    text_at(d, ((x0 + x1) / 2, y + 32), "e2e_latency = TTFT + generation_time", font_en(26, True))
    text_at(
        d,
        (W / 2, H - 40),
        "generation_time 是第一个 token 到最后一个 token。流式时反分词可能发生很多次。",
        font_cn(24),
        MUTED,
    )
    save(im, "03-e2e.png")


def fig4_itl():
    W, H = 2200, 900
    im, d = canvas(W, H)
    text_at(d, (W / 2, 56), "ITL / TPOT：相邻输出之间", font_cn(40, True))
    cn = font_cn(26)
    en = font_en(24, True)
    small = font_cn(24)

    y = 220
    h = 110
    req = (120, y, 360, y + h)
    round_box(d, req, SERVICE, outline=SERVICE, radius=12)
    text_at(d, ((req[0] + req[2]) / 2, (req[1] + req[3]) / 2), "请求发出", cn, WHITE)

    tokens = []
    x = 520
    for i, label in enumerate(["Token 1", "Token 2", "Token 3"], 1):
        b = (x, y, x + 360, y + h)
        round_box(d, b, WHITE, outline=ACCENT if i == 1 else LINE, radius=12, width=3)
        text_at(d, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), label, en, ACCENT if i == 1 else INK)
        tokens.append(b)
        x = b[2] + 90

    # TTFT from request to first token
    dim_h(d, req[2], tokens[0][0], 180, "TTFT（不计入 ITL）", font_cn(22), MUTED)
    for a, b in zip(tokens, tokens[1:]):
        dim_h(d, a[2], b[0], 390, "ITL", en, ACCENT)

    # brace under ITL span
    x1, x2 = tokens[0][2], tokens[-1][0]
    by = 470
    d.line([(tokens[0][0] + 40, 360), (tokens[0][0] + 40, by)], fill=ACCENT, width=2)
    d.line([(tokens[-1][2] - 40, 360), (tokens[-1][2] - 40, by)], fill=ACCENT, width=2)
    d.line([(tokens[0][0] + 40, by), (tokens[-1][2] - 40, by)], fill=ACCENT, width=2)
    text_at(d, ((tokens[0][0] + tokens[-1][2]) / 2, by + 32), "只平均这一段", cn, ACCENT)

    round_box(d, [160, 560, 2040, 700], PANEL, outline=LINE, radius=14)
    text_at(
        d,
        (W / 2, 630),
        "ITL = (e2e_latency − TTFT) / (output_tokens − 1)",
        font_en(28, True),
    )
    text_at(
        d,
        (W / 2, H - 48),
        "GenAI-Perf / AIPerf 不含 TTFT；LLMPerf 常常把 TTFT 算进去。两把尺子最容易在这里打架。",
        small,
        MUTED,
    )
    save(im, "04-itl.png")


def fig5_timeline():
    W, H = 2200, 920
    im, d = canvas(W, H)
    text_at(d, (W / 2, 56), "一场基准的时间轴", font_cn(40, True))
    cn = font_cn(24)
    en = font_en(22, True)
    small = font_cn(24)

    x_start, x_end = 180, 2020
    axis_y = 720
    d.line([(x_start, axis_y), (x_end, axis_y)], fill=INK, width=3)
    arrow(d, (x_end - 20, axis_y), (x_end, axis_y), width=3, head=14)

    marks = [
        (220, "T_start", "基准开始"),
        (480, "Tx", "第一个请求发出"),
        (1680, "Ty", "最后一次响应"),
        (1960, "T_end", "基准结束"),
    ]
    for x, code, zh in marks:
        d.line([(x, axis_y - 14), (x, axis_y + 14)], fill=INK, width=3)
        text_at(d, (x, axis_y + 36), code, en, INK)
        text_at(d, (x, axis_y + 70), zh, cn, MUTED)

    bars = [
        (480, 1520, 180, "L1"),
        (510, 1180, 280, "L2"),
        (700, 1680, 380, "L3"),
    ]
    colors = [(154, 52, 18), (122, 140, 118), (118, 92, 110)]
    for (x1, x2, y, lab), c in zip(bars, colors):
        round_box(d, [x1, y, x2, y + 54], c, outline=c, radius=8)
        text_at(d, ((x1 + x2) / 2, y + 27), lab, en, WHITE)
    text_at(d, (120, 207), "Li", en, MUTED, "mm")
    text_at(d, (120, 250), "e2e", font_cn(20), MUTED, "mm")

    # highlight GenAI-Perf window
    d.line([(480, 150), (480, 170)], fill=ACCENT, width=2)
    d.line([(1680, 150), (1680, 170)], fill=ACCENT, width=2)
    dim_h(d, 480, 1680, 140, "GenAI-Perf：Ty − Tx", font_cn(22), ACCENT)
    dim_h(d, 220, 1960, 88, "LLMPerf：T_end − T_start", font_cn(22), MUTED)

    round_box(d, [160, 790, 2040, 870], PANEL, outline=LINE, radius=12)
    text_at(
        d,
        (W / 2, 830),
        "TPS：总输出 token / 这段时间。LLMPerf 用 T_end − T_start，单并发时造 prompt 等开销有时能占到 33%。",
        small,
        MUTED,
    )
    save(im, "05-bench-timeline.png")


def main():
    fig1_metrics()
    fig2_first_token()
    fig3_e2e()
    fig4_itl()
    fig5_timeline()


if __name__ == "__main__":
    main()
