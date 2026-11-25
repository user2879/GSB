from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


GLYPH_EM = 64
TEMPLATE_SIZE = 16
TEMPLATE_VERSION = "dot-matrix-v1"


def _s(x1: int, y1: int, x2: int, y2: int, w: int = 5) -> tuple[int, int, int, int, int]:
    return x1, y1, x2, y2, w


GLYPH_STROKES: dict[str, list[tuple[int, int, int, int, int]]] = {
    "一": [_s(12, 33, 52, 31, 6)],
    "二": [_s(17, 21, 47, 21, 5), _s(10, 44, 54, 44, 6)],
    "三": [_s(19, 15, 45, 15, 4), _s(14, 32, 50, 32, 5), _s(10, 50, 54, 50, 6)],
    "十": [_s(11, 32, 53, 32, 6), _s(32, 9, 32, 55, 6)],
    "工": [_s(16, 13, 48, 13, 6), _s(32, 14, 32, 49, 6), _s(10, 52, 54, 52, 7)],
    "上": [_s(27, 10, 27, 51, 6), _s(28, 28, 47, 28, 5), _s(11, 52, 53, 52, 7)],
    "下": [_s(10, 15, 54, 15, 7), _s(32, 16, 32, 53, 6), _s(34, 35, 47, 48, 6)],
    "山": [_s(32, 11, 32, 51, 7), _s(14, 24, 14, 52, 6), _s(50, 24, 50, 52, 6), _s(11, 52, 53, 52, 7)],
    "川": [_s(17, 12, 12, 52, 6), _s(32, 9, 32, 55, 7), _s(47, 13, 53, 51, 6)],
    "人": [_s(33, 11, 13, 54, 7), _s(32, 25, 53, 54, 7)],
    "大": [_s(11, 24, 53, 24, 6), _s(32, 17, 13, 55, 7), _s(33, 29, 53, 55, 7)],
    "天": [
        _s(15, 11, 49, 11, 6),
        _s(10, 25, 54, 25, 6),
        _s(32, 24, 14, 55, 7),
        _s(33, 30, 52, 55, 7),
    ],
}

GLYPH_ORDER = list(GLYPH_STROKES.keys())


def render_glyph_image(
    char: str,
    *,
    tile_size: int = 80,
    jitter_seed: int | None = None,
    jitter_px: float = 0.65,
    rotate_deg: float | None = None,
) -> Image.Image:
    if char not in GLYPH_STROKES:
        raise KeyError(f"no built-in dot matrix glyph for {char!r}")
    rng = random.Random(jitter_seed)
    base = Image.new("L", (GLYPH_EM + 16, GLYPH_EM + 16), 255)
    draw = ImageDraw.Draw(base)
    offset = 8
    for x1, y1, x2, y2, width in GLYPH_STROKES[char]:
        if jitter_seed is not None:
            x1 += rng.uniform(-jitter_px, jitter_px)
            y1 += rng.uniform(-jitter_px, jitter_px)
            x2 += rng.uniform(-jitter_px, jitter_px)
            y2 += rng.uniform(-jitter_px, jitter_px)
            width = max(3, width + rng.choice([-1, 0, 0, 1]))
        draw.line(
            (x1 + offset, y1 + offset, x2 + offset, y2 + offset),
            fill=0,
            width=width,
            joint="curve",
        )
    if rotate_deg is None and jitter_seed is not None:
        rotate_deg = rng.uniform(-1.35, 1.35)
    if rotate_deg:
        base = base.rotate(rotate_deg, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if tile_size != base.width:
        base = base.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return base


def _to_binary(img: Image.Image, size: int = TEMPLATE_SIZE, threshold: int = 170) -> np.ndarray:
    resized = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.uint8)
    return arr < threshold


def normalize_mask(mask: np.ndarray, size: int = TEMPLATE_SIZE, ink_fit: int = 14) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    canvas = np.zeros((size, size), dtype=bool)
    if len(xs) == 0:
        return canvas
    tight = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = tight.shape
    scale = float(ink_fit) / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    ink_image = Image.fromarray((~tight).astype(np.uint8) * 255, mode="L").resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    resized = np.asarray(ink_image, dtype=np.uint8) < 170
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _translate(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    y1_src, y2_src = max(0, -dy), min(h, h - dy)
    x1_src, x2_src = max(0, -dx), min(w, w - dx)
    y1_dst, y2_dst = max(0, dy), min(h, h + dy)
    x1_dst, x2_dst = max(0, dx), min(w, w + dx)
    if y2_src > y1_src and x2_src > x1_src:
        out[y1_dst:y2_dst, x1_dst:x2_dst] = mask[y1_src:y2_src, x1_src:x2_src]
    return out


def template_variants(version: str = TEMPLATE_VERSION) -> dict[str, list[np.ndarray]]:
    if version != TEMPLATE_VERSION:
        from .codes import TEMPLATE_VERSION_UNKNOWN
        from .errors import BusinessError

        raise BusinessError(TEMPLATE_VERSION_UNKNOWN, f"unknown template version {version!r}")
    templates: dict[str, list[np.ndarray]] = {}
    for char in GLYPH_ORDER:
        raw = _to_binary(render_glyph_image(char, jitter_seed=None, rotate_deg=0.0))
        normal = normalize_mask(raw)
        bold = normal | _translate(normal, 0, 1) | _translate(normal, 1, 0)
        shifted = _translate(normal, 0, -1)
        templates[char] = [normal, bold, shifted]
    return templates


def template_bundle_fingerprint(version: str = TEMPLATE_VERSION) -> str:
    import hashlib

    variants = template_variants(version)
    h = hashlib.sha256()
    h.update(version.encode())
    for char in sorted(variants):
        for variant in variants[char]:
            h.update(char.encode("utf-8"))
            h.update(variant.astype(np.uint8).tobytes())
    return h.hexdigest()
