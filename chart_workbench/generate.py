"""Deterministic chart-page generator and ground-truth writer."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from .font import draw_text, text_height, text_width


CANVAS_WIDTH = 960
CANVAS_HEIGHT = 640
PLOT_X0 = 118
PLOT_Y0 = 72
PLOT_X1 = 858
PLOT_Y1 = 548
TICK_LENGTH = 8
AXIS_WIDTH = 3

BACKGROUND = 246
AXIS_INK = 0
LINE_INK = 20
MARKER_INK = 100
BAR_INK = 110
LEGEND_INK = 168


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count == 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [start + step * index for index in range(count)]


def _jitter(ideal: Sequence[int], rng: random.Random, amount: int = 1) -> List[int]:
    result = [int(value) for value in ideal]
    for index in range(1, len(result) - 1):
        result[index] += rng.choice((-amount, 0, amount))
    return result


def _linear_y_map(value: float, y_min: float, y_max: float) -> int:
    usable_top = PLOT_Y0 + 32
    usable_bottom = PLOT_Y1 - 32
    ratio = (float(value) - y_min) / (y_max - y_min)
    return int(round(usable_bottom - ratio * (usable_bottom - usable_top)))


def _log_y_map(value: float, values: Sequence[float]) -> int:
    usable_top = PLOT_Y0 + 32
    usable_bottom = PLOT_Y1 - 32
    low = math.log10(min(values))
    high = math.log10(max(values))
    ratio = (math.log10(value) - low) / (high - low)
    return int(round(usable_bottom - ratio * (usable_bottom - usable_top)))


def _x_map(value: float, values: Sequence[float]) -> int:
    left = PLOT_X0 + 52
    right = PLOT_X1 - 52
    low = min(values)
    high = max(values)
    ratio = (float(value) - low) / (high - low) if high != low else 0.0
    return int(round(left + ratio * (right - left)))


def _format_tick_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    x_ticks: Sequence[Tuple[int, float]],
    y_ticks: Sequence[Tuple[int, float]],
    x_scale: int,
) -> Dict[str, List[List[int]]]:
    draw.line((PLOT_X0, PLOT_Y1, PLOT_X1, PLOT_Y1), fill=AXIS_INK, width=AXIS_WIDTH)
    draw.line((PLOT_X0, PLOT_Y0, PLOT_X0, PLOT_Y1), fill=AXIS_INK, width=AXIS_WIDTH)

    label_boxes: Dict[str, List[List[int]]] = {"x": [], "y": []}
    for pixel_y, value in y_ticks:
        draw.line(
            (PLOT_X0 - 2, pixel_y, PLOT_X0 - TICK_LENGTH - 1, pixel_y),
            fill=AXIS_INK,
            width=3,
        )
        text = _format_tick_value(value)
        width = text_width(text, 2)
        x = PLOT_X0 - TICK_LENGTH - 12 - width
        y = pixel_y - text_height(2) // 2
        box = draw_text(draw, x, y, text, scale=2, fill=AXIS_INK)
        label_boxes["y"].append([box[0], box[1], box[2], box[3]])

    for pixel_x, value in x_ticks:
        draw.line(
            (pixel_x, PLOT_Y1 + 2, pixel_x, PLOT_Y1 + TICK_LENGTH + 1),
            fill=AXIS_INK,
            width=3,
        )
        text = _format_tick_value(value)
        width = text_width(text, x_scale)
        x = pixel_x - width // 2
        y = PLOT_Y1 + TICK_LENGTH + 10
        box = draw_text(draw, x, y, text, scale=x_scale, fill=AXIS_INK)
        label_boxes["x"].append([box[0], box[1], box[2], box[3]])
    return label_boxes


def _draw_legend(draw: ImageDraw.ImageDraw) -> None:
    left = PLOT_X1 - 190
    top = PLOT_Y0 - 58
    right = PLOT_X1 - 20
    bottom = PLOT_Y0 - 18
    draw.rectangle((left, top, right, bottom), outline=LEGEND_INK, width=1)
    draw.rectangle((left + 12, top + 10, left + 42, top + 30), fill=BAR_INK)
    draw.rectangle((left + 56, top + 10, left + 86, top + 30), fill=LINE_INK)
    draw.line((left + 100, top + 20, left + 154, top + 20), fill=LINE_INK, width=3)


def _occlude_labels(
    draw: ImageDraw.ImageDraw,
    label_boxes: Dict[str, List[List[int]]],
    axis: str,
    indices: Sequence[int],
) -> None:
    for index in indices:
        x0, y0, x1, y1 = label_boxes[axis][index]
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=BACKGROUND)


def make_page(
    output_dir: Path,
    stem: str,
    *,
    chart_type: str,
    y_mode: str,
    y_values: Sequence[float],
    data_values: Sequence[float],
    data_count: int,
    seed: int,
    rgb: bool = False,
    dense_x: bool = False,
    occluded_y: Sequence[int] = (),
    occluded_x: Sequence[int] = (),
) -> Dict:
    """Create one PNG page and its truth JSON.

    The truth file records rendered (possibly jittered) tick positions. Pixel
    analysis never reads this file; it is only consumed by the verifier.
    """

    rng = random.Random(seed)
    image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    if y_mode == "linear":
        y_ideal = [_linear_y_map(value, min(y_values), max(y_values)) for value in y_values]
        expected_y_mapping = "linear"
    elif y_mode == "log10":
        y_ideal = [_log_y_map(value, y_values) for value in y_values]
        expected_y_mapping = "log10"
    elif y_mode == "broken":
        y_ideal = [516, 476, 436, 300, 260, 220]
        expected_y_mapping = "broken_linear"
    else:
        raise ValueError(f"unknown y_mode: {y_mode}")

    x_values = [float(index) for index in range(data_count)]
    x_ideal = [_x_map(value, x_values) for value in x_values]
    x_scale = 1 if dense_x else 2

    y_positions = _jitter(y_ideal, rng)
    x_positions = _jitter(x_ideal, rng)
    y_ticks = list(zip(y_positions, y_values))
    x_ticks = list(zip(x_positions, x_values))

    label_boxes = _draw_axes(draw, x_ticks, y_ticks, x_scale=x_scale)

    data_pixels: List[Tuple[int, int]] = []
    if y_mode == "linear":
        data_y_ideal = [_linear_y_map(value, min(y_values), max(y_values)) for value in data_values]
    elif y_mode == "log10":
        data_y_ideal = [_log_y_map(value, y_values) for value in data_values]
    else:
        data_y_ideal = []
        first_segment = y_values[:3]
        for value in data_values:
            if min(first_segment) <= value <= max(first_segment):
                data_y_ideal.append(round(516 - (value - 0.0) * 2.0))
            else:
                data_y_ideal.append(round(300 - (value - 100.0) * 2.0))

    for pixel_x, ideal_y in zip(x_positions, data_y_ideal):
        data_pixels.append((int(pixel_x), int(ideal_y)))

    if chart_type == "bar":
        bar_width = 44 if dense_x else 64
        bottom = PLOT_Y1 - 3
        for pixel_x, pixel_y in data_pixels:
            draw.rectangle(
                (pixel_x - bar_width // 2, pixel_y, pixel_x + bar_width // 2, bottom),
                fill=BAR_INK,
            )
    elif chart_type == "line":
        draw.line(data_pixels, fill=LINE_INK, width=2, joint="curve")
        for pixel_x, pixel_y in data_pixels:
            radius = 5
            draw.ellipse(
                (pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius),
                fill=MARKER_INK,
            )
    else:
        raise ValueError(f"unknown chart_type: {chart_type}")

    _draw_legend(draw)
    _occlude_labels(draw, label_boxes, "y", occluded_y)
    _occlude_labels(draw, label_boxes, "x", occluded_x)

    if rgb:
        saved_image = image.convert("RGB")
        mode = "RGB"
    else:
        saved_image = image
        mode = "L"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}.png"
    truth_path = output_dir / f"{stem}.json"
    saved_image.save(image_path, format="PNG", compress_level=9)

    truth = {
        "schema_version": 1,
        "page_id": stem,
        "chart_type": chart_type,
        "image_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "image_mode": mode,
        "plot_rect": {
            "x0": PLOT_X0,
            "y0": PLOT_Y0,
            "x1": PLOT_X1,
            "y1": PLOT_Y1,
        },
        "axes": {
            "x": {
                "orientation": "horizontal",
                "pixel_positions": [int(value) for value in x_positions],
                "values": [float(value) for value in x_values],
            },
            "y": {
                "orientation": "vertical",
                "pixel_positions": [int(value) for value in y_positions],
                "values": [float(value) for value in y_values],
            },
        },
        "data_points": [
            {
                "index": index,
                "pixel": [int(pixel[0]), int(pixel[1])],
                "value": float(value),
            }
            for index, (pixel, value) in enumerate(zip(data_pixels, data_values))
        ],
        "expected_mapping": {"x": "linear", "y": expected_y_mapping},
        "occluded_ticks": {"x": list(occluded_x), "y": list(occluded_y)},
        "generator": {"seed": seed, "tick_length": TICK_LENGTH, "axis_width": AXIS_WIDTH},
    }
    truth_path.write_text(
        json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return truth


def generate_samples(output_dir: Path) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: List[Dict] = []
    pages.append(
        make_page(
            output_dir,
            "page_01_linear_bars",
            chart_type="bar",
            y_mode="linear",
            y_values=[0, 20, 40, 60, 80, 100],
            data_values=[12, 37, 68, 91, 55, 23],
            data_count=6,
            seed=1101,
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_02_log_line",
            chart_type="line",
            y_mode="log10",
            y_values=[1, 10, 100, 1000],
            data_values=[2.5, 9.0, 31.0, 170.0, 800.0, 45.0],
            data_count=6,
            seed=1102,
            rgb=True,
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_03_broken_line",
            chart_type="line",
            y_mode="broken",
            y_values=[0, 20, 40, 100, 120, 140],
            data_values=[10, 30, 110, 130],
            data_count=4,
            seed=1103,
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_04_linear_line_occluded",
            chart_type="line",
            y_mode="linear",
            y_values=[0, 20, 40, 60, 80, 100],
            data_values=[5, 25, 45, 75, 95, 60],
            data_count=6,
            seed=1104,
            occluded_y=(2,),
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_05_dense_bars",
            chart_type="bar",
            y_mode="linear",
            y_values=[0, 10, 20, 30, 40, 50],
            data_values=[5, 15, 35, 25, 45, 10, 30, 20, 40, 12, 28],
            data_count=11,
            seed=1105,
            dense_x=True,
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_06_linear_bar_occluded",
            chart_type="bar",
            y_mode="linear",
            y_values=[0, 20, 40, 60, 80],
            data_values=[18, 57, 73, 34],
            data_count=4,
            seed=1106,
            occluded_x=(1,),
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_07_log_bars",
            chart_type="bar",
            y_mode="log10",
            y_values=[1, 10, 100, 1000],
            data_values=[3.0, 8.0, 25.0, 150.0, 700.0],
            data_count=5,
            seed=1107,
        )
    )
    return pages


def generate_variants(output_dir: Path) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: List[Dict] = []
    pages.append(
        make_page(
            output_dir,
            "page_01_linear_bars",
            chart_type="bar",
            y_mode="linear",
            y_values=[0, 20, 40, 60, 80, 100],
            data_values=[90, 10, 50, 30, 70, 95],
            data_count=6,
            seed=2101,
        )
    )
    pages.append(
        make_page(
            output_dir,
            "page_08_extra_line",
            chart_type="line",
            y_mode="linear",
            y_values=[0, 25, 50, 75, 100],
            data_values=[10, 44, 82, 61, 20],
            data_count=5,
            seed=2108,
        )
    )
    return pages


def write_default_config(path: Path) -> Dict:
    config = {
        "schema_version": 1,
        "sample_dir": "../samples",
        "output_dir": "../out",
        "tick_min_length": 6,
        "line_width_tolerance": 2,
        "axis_detection_threshold": 0.45,
        "broken_axis_detection_threshold": 0.35,
        "log_axis_detection_threshold": 0.01,
        "label_occlusion_tolerance": 0.34,
        "inverse_mode": "ols",
        "fixed_decimal_places": 3,
        "quantization_step": 0.1,
        "thread_pool_concurrency": 4,
        "tick_comparison_budget": 24,
        "page_data_point_quota": 12,
        "snapshot_commit_interval_pages": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def write_edit_commands(path: Path) -> Dict:
    commands = {
        "schema_version": 1,
        "commands": [
            {
                "op": "replace_page",
                "page": "page_01_linear_bars.png",
                "source_png": "../variants/page_01_linear_bars.png",
                "source_truth": "../variants/page_01_linear_bars.json",
            },
            {
                "op": "add_page",
                "page": "page_08_extra_line.png",
                "source_png": "../variants/page_08_extra_line.png",
                "source_truth": "../variants/page_08_extra_line.json",
            },
            {"op": "remove_page", "page": "page_06_linear_bar_occluded.png"},
            {"op": "set_config", "path": "log_axis_detection_threshold", "value": 0.000001},
            {"op": "set_config", "path": "quantization_step", "value": 0.5},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return commands
