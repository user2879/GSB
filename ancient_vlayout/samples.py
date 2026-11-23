"""Deterministic local sample-page generator (no network access)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG, config_hash, normalize_config
from .imaging import detect_column_x_ranges, regress_border
from .templates import build_templates, template_roster
from .util import atomic_write_json, sha256_bytes


SAMPLE_WIDTH = 600
SAMPLE_HEIGHT = 720
_COLUMN_CENTERS = (500, 370, 240, 110)
_CHAR_PITCH = 78
_CHAR_CENTERS_Y = (145, 223, 301, 379, 457)
_GLYPH_SIZE = 72
_WORK_SIZE = 84


def _rng(seed: int, page: int, column: int, char: int) -> np.random.Generator:
    material = f"{seed}:{page}:{column}:{char}".encode("ascii")
    derived = int(sha256_bytes(material)[:16], 16)
    return np.random.default_rng(derived)


def _render_glyph_mask(template: np.ndarray, seed: int, page: int, col: int, char: int) -> np.ndarray:
    rng = _rng(seed, page, col, char)
    source = Image.fromarray((template.astype(np.uint8) * 255), mode="L")
    scaled = source.resize((_GLYPH_SIZE, _GLYPH_SIZE), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("L", (_WORK_SIZE, _WORK_SIZE), 0)
    offset = (_WORK_SIZE - _GLYPH_SIZE) // 2
    canvas.paste(scaled, (offset, offset))
    angle = float(rng.uniform(-2.2, 2.2))
    rotated = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=(_WORK_SIZE / 2.0, _WORK_SIZE / 2.0),
        fillcolor=0,
    )
    mask = np.asarray(rotated, dtype=np.uint8) >= 128
    # Deliberate stroke jitter: each row can slip by one pixel.
    jittered = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        shift = int(rng.integers(-1, 2))
        if shift:
            jittered[y] = np.roll(mask[y], shift)
        else:
            jittered[y] = mask[y]
    return jittered


def _place_mask(
    page_mask: np.ndarray, glyph: np.ndarray, cx: int, cy: int
) -> tuple[tuple[int, int, int, int], int, int]:
    height, width = glyph.shape
    x0 = int(cx - width // 2)
    y0 = int(cy - height // 2)
    local = regress_border(glyph, 0, width, 0, height)
    bbox = (
        int(x0 + local.bbox[0]),
        int(y0 + local.bbox[1]),
        int(x0 + local.bbox[2]),
        int(y0 + local.bbox[3]),
    )
    page_mask[y0 : y0 + height, x0 : x0 + width] |= glyph
    return bbox, x0, y0


def _draw_one_pixel_bridge(mask: np.ndarray, upper: tuple[int, int, int, int], lower: tuple[int, int, int, int]) -> None:
    x = (upper[0] + upper[2] - 1) // 2
    y0 = upper[3] - 2
    y1 = lower[1] + 2
    if y0 <= y1:
        mask[y0 : y1 + 1, x] = True


def _line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= y0 < mask.shape[0] and 0 <= x0 < mask.shape[1]:
            mask[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _render_page(
    page_index: int,
    seed: int,
    templates: dict[str, list[np.ndarray]],
    contents: list[list[str]],
    touched_pair: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed + 1000 * page_index)
    glyph_layer = np.zeros((SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=bool)
    columns_data: list[dict[str, Any]] = []
    glyph_records: list[list[tuple[np.ndarray, int, int]]] = []
    coarse_bboxes: list[list[tuple[int, int, int, int]]] = []

    for col_index, cx in enumerate(_COLUMN_CENTERS):
        records: list[tuple[np.ndarray, int, int]] = []
        coarse_boxes: list[tuple[int, int, int, int]] = []
        for char_index, cy in enumerate(_CHAR_CENTERS_Y):
            char = contents[col_index][char_index]
            chosen = templates[char][int(rng.integers(0, len(templates[char])))]
            glyph = _render_glyph_mask(chosen, seed, page_index, col_index, char_index)
            bbox, offset_x, offset_y = _place_mask(glyph_layer, glyph, cx, cy)
            records.append((glyph, offset_x, offset_y))
            coarse_boxes.append(bbox)
        glyph_records.append(records)
        coarse_bboxes.append(coarse_boxes)

    clean_columns = detect_column_x_ranges(glyph_layer, 3)
    if len(clean_columns) != len(_COLUMN_CENTERS):
        raise RuntimeError(f"generator saw {len(clean_columns)} clean columns, expected 4")

    bboxes: list[list[tuple[int, int, int, int]]] = []
    column_boxes: list[tuple[int, int, int, int]] = []
    for col_index, clean in enumerate(clean_columns):
        column_x0, column_x1 = int(clean["x0"]), int(clean["x1"])
        char_boxes: list[tuple[int, int, int, int]] = []
        for glyph, offset_x, offset_y in glyph_records[col_index]:
            local_x0 = max(0, column_x0 - offset_x)
            local_x1 = min(glyph.shape[1], column_x1 - offset_x)
            local = glyph[:, local_x0:local_x1]
            if not local.any():
                raise RuntimeError("generator glyph lies outside detected column")
            regression = regress_border(local, 0, local.shape[1], 0, local.shape[0])
            bbox = (
                int(offset_x + local_x0 + regression.bbox[0]),
                int(offset_y + regression.bbox[1]),
                int(offset_x + local_x0 + regression.bbox[2]),
                int(offset_y + regression.bbox[3]),
            )
            char_boxes.append(bbox)
        bboxes.append(char_boxes)
        column_boxes.append(
            (
                column_x0,
                min(box[1] for box in char_boxes),
                column_x1,
                max(box[3] for box in char_boxes),
            )
        )

    touched_column, touched_first = touched_pair
    bridge_mask = glyph_layer.copy()
    upper = bboxes[touched_column][touched_first]
    lower = bboxes[touched_column][touched_first + 1]
    _draw_one_pixel_bridge(bridge_mask, upper, lower)
    bridge_only = bridge_mask & ~glyph_layer

    ink_mask = glyph_layer | bridge_only
    # Noise is placed only in existing inter-character blank gaps.  Small
    # isolated components exercise projection-discard diagnostics without
    # changing the ground-truth character boxes.
    for col_index, (box_x0, _, box_x1, _) in enumerate(column_boxes):
        for char_index in range(len(_CHAR_CENTERS_Y) - 1):
            gap_y0 = bboxes[col_index][char_index][3] + 3
            gap_y1 = bboxes[col_index][char_index + 1][1] - 3
            if gap_y1 - gap_y0 < 5:
                continue
            if (col_index + char_index + page_index) % 3 != 0:
                continue
            bx = int(rng.integers(box_x0 + 3, max(box_x0 + 4, box_x1 - 3)))
            by = int(rng.integers(gap_y0, max(gap_y0 + 1, gap_y1 - 1)))
            ink_mask[by, bx] = True
            ink_mask[min(SAMPLE_HEIGHT - 1, by + 2), max(0, bx - 1)] = True
        # A handful of one-pixel dust specks.
        for _ in range(5):
            px = int(rng.integers(box_x0, box_x1))
            py = int(rng.integers(0, SAMPLE_HEIGHT))
            ink_mask[py, px] = True

    paper = rng.integers(212, 246, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    texture = ((np.arange(SAMPLE_WIDTH)[None, :] * 3 + np.arange(SAMPLE_HEIGHT)[:, None] * 5) % 7).astype(np.uint8)
    paper = np.clip(paper.astype(np.int16) + texture - 3, 0, 255).astype(np.uint8)
    ink_values = rng.integers(26, 66, size=(SAMPLE_HEIGHT, SAMPLE_WIDTH), dtype=np.uint8)
    gray = np.where(ink_mask, ink_values, paper).astype(np.uint8)

    for col_index, char_boxes in enumerate(bboxes):
        chars = []
        for char_index, bbox in enumerate(char_boxes):
            chars.append(
                {
                    "reading_index": char_index,
                    "bbox": [int(value) for value in bbox],
                    "char": contents[col_index][char_index],
                }
            )
        columns_data.append(
            {
                "index": col_index,
                "bbox": [int(value) for value in column_boxes[col_index]],
                "chars": chars,
            }
        )
    return gray, columns_data


def _contents_for_page(page: int, roster: tuple[str, ...]) -> list[list[str]]:
    total = 4 * 5
    sequence = [roster[(page * 3 + index * 5) % len(roster)] for index in range(total)]
    if page == 1:
        sequence[0] = "回"
    return [sequence[column * 5 : column * 5 + 5] for column in range(4)]


def _injection_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "items": [
            {
                "id": "INJ-001",
                "kind": "truncate_png",
                "target": "page_04.png",
                "truncate_at_ratio": 0.63,
                "expected_error": "E_IMAGE_INVALID",
            },
            {
                "id": "INJ-002",
                "kind": "truth_bbox_out_of_bounds",
                "target": "page_03.png",
                "column_index": 1,
                "char_index": 2,
                "bbox": [950, 100, 1010, 160],
                "expected_error": "E_TRUTH_INVALID",
            },
            {
                "id": "INJ-003",
                "kind": "missing_template",
                "target": "page_02.png",
                "template_id": "回",
                "expected_error": "E_TEMPLATE_MISSING",
            },
            {
                "id": "INJ-004",
                "kind": "nonfinite_fusion_weight",
                "target": "page_01.png",
                "weight_index": 0,
                "literal": "nan",
                "expected_error": "E_CONFIG_INVALID",
            },
        ],
    }


def generate_samples(
    target_dir: str | Path,
    *,
    seed: int = 20261124,
    clean: bool = True,
) -> dict[str, Any]:
    """Generate PNG pages, truth JSON, config JSON and injection manifest."""

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    if clean:
        for pattern in ("page_*.png", "page_*.truth.json", "config.json", "injections.json", "manifest.json"):
            for path in target.glob(pattern):
                path.unlink()
        for cache in target.glob(".tmp-*"):
            if cache.is_dir():
                shutil.rmtree(cache, ignore_errors=True)

    templates = build_templates()
    roster = template_roster()
    raw_config = dict(DEFAULT_CONFIG)
    raw_config.update(
        {
            "binarization_threshold": 160,
            "projection_window": 3,
            "min_char_height": 22,
            "max_char_height": 90,
            "split_penalty": 12,
            "split_valley_weight": 2,
            "match_metric": "overlap",
            "fusion_weights": [1, 1, 1],
            "fixed_point_bits": 12,
            "confidence_quantization_steps": 100,
            "ma_point_budget": 2_000_000,
            "thread_pool_workers": 4,
            "snapshot_interval": 2,
            "injection_manifest": "injections.json",
            "expected_width": SAMPLE_WIDTH,
            "expected_height": SAMPLE_HEIGHT,
        }
    )
    config = normalize_config(raw_config, variant_count=len(templates[roster[0]]))
    atomic_write_json(target / "config.json", config)
    injection_manifest = _injection_manifest()
    atomic_write_json(target / "injections.json", injection_manifest)

    pages: list[dict[str, Any]] = []
    for page in range(4):
        contents = _contents_for_page(page, roster)
        touched_pair = (page % 4, (page + 1) % 4)
        gray, columns = _render_page(page, seed, templates, contents, touched_pair)
        image_name = f"page_{page + 1:02d}.png"
        image_path = target / image_name
        Image.fromarray(gray, mode="L").save(image_path, format="PNG", optimize=False, compress_level=6)
        truth = {
            "schema_version": 1,
            "image": image_name,
            "width": SAMPLE_WIDTH,
            "height": SAMPLE_HEIGHT,
            "columns": columns,
        }
        atomic_write_json(target / f"page_{page + 1:02d}.truth.json", truth)
        pages.append(
            {
                "image": image_name,
                "truth": f"page_{page + 1:02d}.truth.json",
                "sha256": sha256_bytes(image_path.read_bytes()),
                "columns": len(columns),
                "characters": sum(len(column["chars"]) for column in columns),
                "touched_pair": list(touched_pair),
            }
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "config": "config.json",
        "config_hash": config_hash(config),
        "injections": "injections.json",
        "pages": pages,
    }
    atomic_write_json(target / "manifest.json", manifest)
    return manifest
