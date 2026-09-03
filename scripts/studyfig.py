#!/usr/bin/env python3
"""Shared drawing helpers for Chinese study figures."""
from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

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
CACHED = (176, 160, 148)
WAIT = (214, 196, 168)
GPU = (92, 83, 74)

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
    ang = atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - head * cos(ang - 0.45), y2 - head * sin(ang - 0.45)),
        (x2 - head * cos(ang + 0.45), y2 - head * sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def canvas(w, h):
    im = Image.new("RGB", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, w - 1, h - 1], outline=LINE, width=2)
    return im, draw


def save(im, rel: str):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG", optimize=True)
    print(path.relative_to(ROOT), im.size)


def title(draw, w, s):
    text_at(draw, (w / 2, 52), s, font_cn(36, True))


def footnote(draw, w, h, s):
    text_at(draw, (w / 2, h - 36), s, font_cn(22), MUTED)
