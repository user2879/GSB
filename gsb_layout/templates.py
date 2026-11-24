"""Built-in 16x16 dot-matrix glyphs and deterministic template variants."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable

import numpy as np
from PIL import Image

TEMPLATE_SIZE = 16
TEMPLATE_ALGORITHM_VERSION = "dot16-procedural-v1"


def _blank() -> np.ndarray:
    return np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.uint8)


def _paint(canvas: np.ndarray, x: int, y: int, radius: int = 0) -> None:
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if 0 <= xx < TEMPLATE_SIZE and 0 <= yy < TEMPLATE_SIZE:
                canvas[yy, xx] = 1


def _line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    radius = max(0, (thickness - 1) // 2)
    while True:
        _paint(canvas, x0, y0, radius)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _box(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, t: int = 1) -> None:
    _line(canvas, x0, y0, x1, y0, t)
    _line(canvas, x0, y1, x1, y1, t)
    _line(canvas, x0, y0, x0, y1, t)
    _line(canvas, x1, y0, x1, y1, t)


def _build_base_glyphs() -> dict[str, np.ndarray]:
    glyphs: dict[str, np.ndarray] = {}

    def glyph(name: str):
        canvas = _blank()
        glyphs[name] = canvas
        return canvas

    c = glyph("一")
    _line(c, 2, 7, 13, 7, 2)

    c = glyph("二")
    _line(c, 2, 4, 13, 4, 2)
    _line(c, 3, 11, 12, 11, 2)

    c = glyph("三")
    _line(c, 2, 3, 13, 3, 1)
    _line(c, 3, 8, 12, 8, 1)
    _line(c, 2, 12, 13, 12, 2)

    c = glyph("十")
    _line(c, 2, 7, 13, 7, 2)
    _line(c, 7, 2, 7, 13, 2)

    c = glyph("人")
    _line(c, 7, 2, 3, 12, 2)
    _line(c, 8, 2, 12, 12, 2)

    c = glyph("大")
    _line(c, 2, 7, 13, 7, 2)
    _line(c, 7, 2, 3, 12, 1)
    _line(c, 8, 2, 12, 12, 1)

    c = glyph("小")
    _line(c, 7, 3, 7, 12, 1)
    _line(c, 4, 5, 2, 8, 1)
    _line(c, 11, 5, 13, 8, 1)
    _line(c, 6, 12, 8, 12, 1)

    c = glyph("山")
    _line(c, 3, 10, 3, 4, 1)
    _line(c, 7, 10, 7, 2, 1)
    _line(c, 11, 10, 11, 4, 1)
    _line(c, 2, 11, 13, 11, 2)

    c = glyph("川")
    _line(c, 4, 3, 4, 12, 1)
    _line(c, 7, 3, 7, 12, 1)
    _line(c, 10, 4, 10, 12, 2)

    c = glyph("口")
    _box(c, 3, 3, 12, 12, 2)

    c = glyph("日")
    _box(c, 3, 2, 12, 13, 2)
    _line(c, 4, 7, 11, 7, 1)

    c = glyph("月")
    _line(c, 4, 2, 4, 13, 2)
    _line(c, 4, 2, 11, 2, 1)
    _line(c, 11, 3, 11, 12, 1)
    _line(c, 5, 7, 11, 7, 1)
    _line(c, 5, 10, 11, 10, 1)

    c = glyph("田")
    _box(c, 3, 3, 12, 12, 2)
    _line(c, 7, 4, 7, 11, 1)
    _line(c, 4, 7, 11, 7, 1)

    c = glyph("王")
    _line(c, 3, 3, 12, 3, 2)
    _line(c, 4, 8, 11, 8, 1)
    _line(c, 3, 12, 12, 12, 2)
    _line(c, 7, 4, 7, 11, 1)

    c = glyph("土")
    _line(c, 3, 7, 12, 7, 2)
    _line(c, 7, 3, 7, 12, 1)
    _line(c, 3, 12, 12, 12, 2)

    c = glyph("工")
    _line(c, 3, 4, 12, 4, 2)
    _line(c, 7, 4, 7, 11, 1)
    _line(c, 3, 11, 12, 11, 2)

    c = glyph("火")
    _line(c, 7, 3, 7, 11, 1)
    _line(c, 4, 5, 2, 9, 1)
    _line(c, 11, 5, 13, 9, 1)
    _line(c, 5, 10, 3, 12, 1)
    _line(c, 10, 10, 12, 12, 1)

    c = glyph("木")
    _line(c, 2, 7, 13, 7, 2)
    _line(c, 7, 2, 7, 12, 1)
    _line(c, 6, 8, 3, 12, 1)
    _line(c, 9, 8, 12, 12, 1)

    c = glyph("水")
    _line(c, 7, 2, 7, 13, 1)
    _line(c, 4, 4, 2, 8, 1)
    _line(c, 10, 4, 12, 7, 1)
    _line(c, 6, 10, 3, 12, 1)
    _line(c, 9, 9, 12, 12, 1)

    c = glyph("石")
    _line(c, 2, 4, 12, 4, 2)
    _line(c, 7, 4, 4, 8, 1)
    _line(c, 3, 8, 12, 8, 2)
    _line(c, 3, 8, 3, 12, 1)
    _line(c, 12, 8, 12, 12, 1)
    _line(c, 3, 12, 12, 12, 2)
    return glyphs


BASE_GLYPHS = _build_base_glyphs()
FONT_CHARACTERS = tuple(sorted(BASE_GLYPHS, key=ord))


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    big = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
        (TEMPLATE_SIZE * 8, TEMPLATE_SIZE * 8), Image.Resampling.NEAREST
    )
    rotated = big.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    small = rotated.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.uint8)
    return (arr >= 80).astype(np.uint8)


@dataclass(frozen=True)
class TemplateSet:
    size: int
    version: str
    characters: tuple[str, ...]
    variants: dict[str, tuple[np.ndarray, ...]]
    weights: tuple[int, ...]
    algorithm: str = TEMPLATE_ALGORITHM_VERSION

    @property
    def variant_count(self) -> int:
        return len(self.weights)

    def ordered_characters(self) -> tuple[str, ...]:
        return tuple(sorted(self.characters, key=ord))


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    height, width = mask.shape
    ys0 = max(0, dy)
    ys1 = min(height, height + dy)
    xs0 = max(0, dx)
    xs1 = min(width, width + dx)
    if ys0 < ys1 and xs0 < xs1:
        shifted[ys0:ys1, xs0:xs1] = mask[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return shifted


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    result = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result = np.maximum(
                result,
                padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]],
            )
    return result


def _variant_transforms(base: np.ndarray) -> tuple[np.ndarray, ...]:
    """Stable template family; callers select a prefix by fusion-weight count."""
    return (
        base.copy(),
        _rotate_mask(base, -2.2),
        _rotate_mask(base, 2.2),
        _shift_mask(base, 1, 0),
        _shift_mask(base, 0, 1),
        _dilate_mask(base),
        _rotate_mask(base, -4.0),
        _rotate_mask(base, 4.0),
    )


def build_template_set(weights: Iterable[int]) -> TemplateSet:
    weight_tuple = tuple(int(v) for v in weights)
    if not 1 <= len(weight_tuple) <= 8:
        raise ValueError("template_weights 必须包含 1 到 8 个正整数")
    variants: dict[str, tuple[np.ndarray, ...]] = {}
    for char in FONT_CHARACTERS:
        base = BASE_GLYPHS[char]
        raw_variants = _variant_transforms(base)[: len(weight_tuple)]
        cropped: list[np.ndarray] = []
        for raw in raw_variants:
            ys, xs = np.nonzero(raw)
            if xs.size:
                tight = raw[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
            else:
                tight = raw
            image = Image.fromarray((tight * 255).astype(np.uint8), mode="L").resize(
                (TEMPLATE_SIZE, TEMPLATE_SIZE), Image.Resampling.BOX
            )
            cropped.append((np.asarray(image, dtype=np.uint8) >= 72).astype(np.uint8))
        variants[char] = tuple(cropped)
    payload = {
        "algorithm": TEMPLATE_ALGORITHM_VERSION,
        "size": TEMPLATE_SIZE,
        "weights": list(weight_tuple),
        "glyphs": {ch: [v.astype(int).tolist() for v in variants[ch]] for ch in FONT_CHARACTERS},
    }
    version = sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TemplateSet(
        size=TEMPLATE_SIZE,
        version=version,
        characters=FONT_CHARACTERS,
        variants=variants,
        weights=weight_tuple,
    )
