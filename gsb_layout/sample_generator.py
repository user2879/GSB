"""Generate deterministic local vertical-page PNG samples and truth JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image

from .config import Config
from .templates import BASE_GLYPHS, TEMPLATE_SIZE

SAMPLE_WIDTH = 480
SAMPLE_HEIGHT = 360


@dataclass
class RenderedGlyph:
    text: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    x: int
    y: int


def _render_glyph_mask(char: str, size: int, angle: float, rng: np.random.Generator) -> np.ndarray:
    base = BASE_GLYPHS[char]
    big = Image.fromarray((base * 255).astype(np.uint8), mode="L").resize(
        (TEMPLATE_SIZE * 5, TEMPLATE_SIZE * 5), Image.Resampling.NEAREST
    )
    rotated = big.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)
    side = min(rotated.size)
    left = (rotated.size[0] - side) // 2
    top = (rotated.size[1] - side) // 2
    rotated = rotated.crop((left, top, left + side, top + side))
    small = rotated.resize((size, size), Image.Resampling.BOX)
    mask = np.asarray(small, dtype=np.uint8).astype(np.float64) / 255.0
    # Deterministic sub-pixel translation plus small edge chips/speckles. These
    # keep stroke topology intact while still exercising template fusion.
    dy = int(rng.integers(-1, 2))
    dx = int(rng.integers(-1, 2))
    mask = np.roll(mask, (dy, dx), axis=(0, 1))
    if dy < 0:
        mask[dy:, :] = 0
    elif dy > 0:
        mask[:dy, :] = 0
    if dx < 0:
        mask[:, dx:] = 0
    elif dx > 0:
        mask[:, :dx] = 0
    ink_y, ink_x = np.nonzero(mask >= 0.55)
    if ink_x.size:
        for _ in range(2):
            pick = int(rng.integers(0, ink_x.size))
            mask[ink_y[pick], ink_x[pick]] = 0.0
    for _ in range(3):
        yy = int(rng.integers(2, size - 2))
        xx = int(rng.integers(2, size - 2))
        if mask[yy, xx] < 0.08:
            mask[yy, xx] = float(rng.integers(12, 28)) / 100.0
    return mask


def _paste_mask(page: np.ndarray, mask: np.ndarray, x: int, y: int, ink: int = 18) -> None:
    h, w = mask.shape
    y1 = min(page.shape[0], y + h)
    x1 = min(page.shape[1], x + w)
    region = page[y:y1, x:x1].astype(np.float64)
    alpha = mask[: y1 - y, : x1 - x]
    region = region * (1.0 - alpha) + float(ink) * alpha
    page[y:y1, x:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


def _ink_bbox(mask: np.ndarray, x: int, y: int, cutoff: float = 0.10) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask >= cutoff)
    if xs.size == 0:
        return x, y, x + max(1, mask.shape[1]), y + max(1, mask.shape[0])
    return (
        int(x + xs.min()),
        int(y + ys.min()),
        int(x + xs.max() + 1),
        int(y + ys.max() + 1),
    )


def _draw_bridge(
    page: np.ndarray,
    previous: RenderedGlyph,
    current: RenderedGlyph,
    rng: np.random.Generator,
) -> None:
    px0, py0, px1, py1 = previous.bbox
    cx0, cy0, cx1, cy1 = current.bbox
    x = (px0 + px1 + cx0 + cx1) // 4
    gap_top = py1 - 2
    gap_bottom = cy0 + 2
    if gap_bottom <= gap_top:
        gap_top, gap_bottom = py1, cy0 + 1
    for yy in range(max(0, gap_top), min(page.shape[0], gap_bottom + 1)):
        width = 1 + int(rng.integers(0, 3))
        centre = x + int(rng.integers(-2, 3))
        for xx in range(max(0, centre - width), min(page.shape[1], centre + width + 1)):
            page[yy, xx] = min(int(page[yy, xx]), int(rng.integers(10, 70)))


def _render_column(
    page: np.ndarray,
    texts: str,
    center_x: int,
    start_y: int,
    seed: int,
    joined_after: set[int] | None = None,
) -> tuple[list[RenderedGlyph], tuple[int, int, int, int]]:
    joined_after = joined_after or set()
    rng = np.random.default_rng(seed)
    glyphs: list[RenderedGlyph] = []
    next_y = start_y
    for index, char in enumerate(texts):
        angle = float(rng.uniform(-2.6, 2.6))
        mask = _render_glyph_mask(char, 30, angle, rng)
        x = center_x - mask.shape[1] // 2 + int(rng.integers(-1, 2))
        ink_rows = np.flatnonzero((mask >= 0.10).sum(axis=1) > 0)
        first_ink_row = int(ink_rows[0]) if ink_rows.size else 0
        if glyphs and index - 1 in joined_after:
            # Keep the truth boxes disjoint (one blank row), then reconnect them with
            # an ink speck so projection segmentation sees one oversize block.
            paste_y = glyphs[-1].bbox[3] + 1 - first_ink_row
        else:
            paste_y = next_y + int(rng.integers(-1, 2))
        _paste_mask(page, mask, x, paste_y)
        bbox = _ink_bbox(mask, x, paste_y)
        glyph = RenderedGlyph(char, bbox, mask, x, paste_y)
        if glyphs and index - 1 in joined_after:
            _draw_bridge(page, glyphs[-1], glyph, rng)
        glyphs.append(glyph)
        next_y = max(bbox[3] + 13, paste_y + 13)
    x0 = min(g.bbox[0] for g in glyphs)
    y0 = min(g.bbox[1] for g in glyphs)
    x1 = max(g.bbox[2] for g in glyphs)
    y1 = max(g.bbox[3] for g in glyphs)
    return glyphs, (x0, y0, x1, y1)


def _add_background(page: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-3, 4, size=page.shape, dtype=np.int16)
    page[:] = np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Very light paper fiber marks remain above the default threshold.
    for _ in range(45):
        yy = int(rng.integers(0, page.shape[0]))
        xx = int(rng.integers(0, page.shape[1]))
        page[yy, xx] = min(255, max(205, int(page[yy, xx]) - int(rng.integers(0, 18))))


def _page_specs() -> list[dict]:
    return [
        {
            "name": "page_001",
            "columns": [
                {"texts": "日月", "center": 418, "start": 28, "joined": {0}},
                {"texts": "田王土", "center": 337, "start": 30, "joined": set()},
                {"texts": "口工山", "center": 256, "start": 26, "joined": {1}},
            ],
        },
        {
            "name": "page_002",
            "columns": [
                {"texts": "十大人", "center": 425, "start": 24, "joined": set()},
                {"texts": "月木水", "center": 347, "start": 30, "joined": {0}},
                {"texts": "火土工", "center": 269, "start": 27, "joined": set()},
                {"texts": "口日田", "center": 191, "start": 33, "joined": set()},
            ],
        },
        {
            "name": "page_003",
            "columns": [
                {"texts": "木水火", "center": 414, "start": 27, "joined": {1}},
                {"texts": "土工口", "center": 333, "start": 31, "joined": set()},
                {"texts": "日月田", "center": 252, "start": 25, "joined": {0}},
            ],
        },
    ]


def _base_config() -> dict:
    return {
        "binarization_threshold": 160,
        "row_projection_window": 5,
        "char_min_height": 14,
        "char_max_height": 30,
        "split_penalty": 12,
        "match_metric": "iou",
        "template_weights": [4, 3, 2],
        "fixed_bits": 10,
        "quant_levels": 16,
        "confidence_tolerance": 0.62,
        "thread_count": 4,
        "match_point_budget": 2000000,
        "snapshot_interval": 1,
        "expected_width": SAMPLE_WIDTH,
        "expected_height": SAMPLE_HEIGHT,
        "border_residual_tolerance": 5,
        "confidence_low_q": 0,
        "confidence_high_q": None,
    }


def generate_samples(
    samples_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = True,
) -> dict:
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path) if config_path else samples_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = list(samples_dir.glob("page_*.png"))
        if existing:
            raise FileExistsError("样本已存在；使用 overwrite=True 覆盖")
    created: list[str] = []
    truth_index: dict[str, str] = {}
    for page_index, spec in enumerate(_page_specs(), start=1):
        page = np.full((SAMPLE_HEIGHT, SAMPLE_WIDTH), 244, dtype=np.uint8)
        column_records: list[dict] = []
        all_columns: list[tuple[list[RenderedGlyph], tuple[int, int, int, int]]] = []
        for column_index, column in enumerate(spec["columns"], start=1):
            glyphs, column_bbox = _render_column(
                page,
                column["texts"],
                column["center"],
                column["start"],
                seed=1000 * page_index + 50 * column_index,
                joined_after=column["joined"],
            )
            all_columns.append((glyphs, column_bbox))
        _add_background(page, seed=9000 + page_index)
        # Recompute tight bboxes after background noise. Text masks dominate, but the
        # crop remains deterministic because noise is at least 205 gray (> threshold).
        for column_index, (glyphs, _) in enumerate(all_columns, start=1):
            char_records = []
            for char_index, glyph in enumerate(sorted(glyphs, key=lambda g: g.bbox[1]), start=1):
                char_records.append(
                    {
                        "reading_index": char_index,
                        "bbox": list(glyph.bbox),
                        "text": glyph.text,
                    }
                )
            x0 = min(c["bbox"][0] for c in char_records)
            y0 = min(c["bbox"][1] for c in char_records)
            x1 = max(c["bbox"][2] for c in char_records)
            y1 = max(c["bbox"][3] for c in char_records)
            column_records.append(
                {
                    "reading_index": column_index,
                    "bbox": [x0, y0, x1, y1],
                    "chars": char_records,
                }
            )
        png_path = samples_dir / f"{spec['name']}.png"
        Image.fromarray(page, mode="L").save(png_path, format="PNG", optimize=False)
        truth = {
            "image": png_path.name,
            "width": SAMPLE_WIDTH,
            "height": SAMPLE_HEIGHT,
            "columns": column_records,
        }
        truth_path = samples_dir / f"{spec['name']}.truth.json"
        truth_path.write_text(
            json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.extend([png_path.name, truth_path.name])
        truth_index[png_path.name] = truth_path.name
    config_path.write_text(
        json.dumps(_base_config(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    created.append(config_path.name)
    return {
        "samples_dir": str(samples_dir),
        "config": str(config_path),
        "created": created,
        "truth": truth_index,
    }


def load_sample_config(samples_dir: str | Path) -> Config:
    from .config import load_config

    return load_config(Path(samples_dir) / "config.json")
