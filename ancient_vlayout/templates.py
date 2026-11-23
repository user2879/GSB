"""Built-in 16x16 dot-matrix glyph templates and deterministic variants."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .util import sha256_json


TEMPLATE_VERSION_PREFIX = "dotmatrix-v1"
TEMPLATE_SIZE = 24
_SIZE = TEMPLATE_SIZE
_GLYPHS = ("口", "日", "田", "回", "中", "十", "王", "山", "川", "人", "大", "木")


def _blank() -> np.ndarray:
    return np.zeros((_SIZE, _SIZE), dtype=np.uint8)


def _hline(mask: np.ndarray, y: int, x0: int, x1: int) -> None:
    mask[y, x0 : x1 + 1] = 1


def _vline(mask: np.ndarray, x: int, y0: int, y1: int) -> None:
    mask[y0 : y1 + 1, x] = 1


def _diag(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        mask[y0, x0] = 1
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _outer_box(mask: np.ndarray) -> None:
    _hline(mask, 3, 5, 18)
    _hline(mask, 20, 5, 18)
    _vline(mask, 5, 3, 20)
    _vline(mask, 18, 3, 20)


def _base_glyph(char: str) -> np.ndarray:
    mask = _blank()
    if char == "口":
        _outer_box(mask)
    elif char == "日":
        _outer_box(mask)
        _hline(mask, 12, 6, 17)
    elif char == "田":
        _outer_box(mask)
        _hline(mask, 12, 6, 17)
        _vline(mask, 12, 4, 19)
    elif char == "回":
        _outer_box(mask)
        _hline(mask, 7, 9, 14)
        _hline(mask, 16, 9, 14)
        _vline(mask, 9, 7, 16)
        _vline(mask, 14, 7, 16)
    elif char == "中":
        _outer_box(mask)
        _vline(mask, 12, 0, 23)
    elif char == "十":
        _hline(mask, 12, 4, 19)
        _vline(mask, 12, 1, 22)
    elif char == "王":
        _hline(mask, 4, 5, 18)
        _hline(mask, 12, 5, 18)
        _hline(mask, 20, 5, 18)
        _vline(mask, 12, 4, 20)
    elif char == "山":
        _hline(mask, 20, 4, 19)
        _vline(mask, 7, 4, 20)
        _vline(mask, 12, 4, 20)
        _vline(mask, 17, 4, 20)
    elif char == "川":
        _vline(mask, 7, 2, 20)
        _vline(mask, 12, 4, 22)
        _vline(mask, 17, 2, 20)
    elif char == "人":
        _diag(mask, 12, 3, 5, 21)
        _diag(mask, 12, 3, 19, 21)
    elif char == "大":
        _hline(mask, 11, 4, 19)
        _diag(mask, 9, 4, 12, 11)
        _diag(mask, 15, 4, 12, 11)
        _diag(mask, 12, 11, 5, 21)
        _diag(mask, 12, 11, 19, 21)
    elif char == "木":
        _vline(mask, 12, 1, 22)
        _hline(mask, 10, 4, 19)
        _diag(mask, 12, 10, 5, 21)
        _diag(mask, 12, 10, 19, 21)
    else:  # pragma: no cover - roster is internal
        raise KeyError(char)
    return mask


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _row_jitter(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        shift = (-1, 0, 1)[(y * 7 + 3) % 3]
        out[y] = np.roll(mask[y], shift)
    return out


def _variants(base: np.ndarray) -> tuple[np.ndarray, ...]:
    return (base.copy(), _dilate(base), _row_jitter(base))


def build_templates(roster: Iterable[str] | None = None) -> dict[str, list[np.ndarray]]:
    chars = tuple(roster) if roster is not None else _GLYPHS
    return {char: [variant.copy() for variant in _variants(_base_glyph(char))] for char in chars}


def template_roster() -> tuple[str, ...]:
    return _GLYPHS


def template_variant_count() -> int:
    return 3


def template_version(templates: dict[str, list[np.ndarray]] | None = None) -> str:
    templates = templates if templates is not None else build_templates()
    payload = {
        "prefix": TEMPLATE_VERSION_PREFIX,
        "size": _SIZE,
        "glyphs": {
            char: [variant.astype(np.uint8).tolist() for variant in variants]
            for char, variants in sorted(templates.items())
        },
    }
    return sha256_json(payload)


def clone_templates(templates: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {char: [variant.copy() for variant in variants] for char, variants in templates.items()}
