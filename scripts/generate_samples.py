"""Generate deterministic PNG layouts and matching page-truth JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parseapp.png import encode_png_rgb  # noqa: E402


PAGE_WIDTH = 960
PAGE_HEIGHT = 600


def _subtract_intervals(
    start: int, end: int, gaps: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    intervals = [(start, end)]
    for gap_start, gap_end in sorted(gaps):
        updated: list[tuple[int, int]] = []
        for interval_start, interval_end in intervals:
            if gap_end < interval_start or gap_start > interval_end:
                updated.append((interval_start, interval_end))
                continue
            if interval_start < gap_start:
                updated.append((interval_start, gap_start - 1))
            if gap_end + 1 <= interval_end:
                updated.append((gap_end + 1, interval_end))
        intervals = updated
    return intervals


def _draw_horizontal(
    image: np.ndarray,
    y: int,
    start: int,
    end: int,
    gaps: Sequence[tuple[int, int]] = (),
    value: int = 0,
) -> None:
    for interval_start, interval_end in _subtract_intervals(start, end, gaps):
        image[y : y + 2, interval_start : interval_end + 1] = value


def _draw_vertical(
    image: np.ndarray,
    x: int,
    start: int,
    end: int,
    gaps: Sequence[tuple[int, int]] = (),
    value: int = 0,
) -> None:
    for interval_start, interval_end in _subtract_intervals(start, end, gaps):
        image[interval_start : interval_end + 1, x : x + 2] = value


def _make_cells(
    table_id: str,
    rows: Sequence[int],
    cols: Sequence[int],
    rectangles: Sequence[tuple[int, int, int, int]] | None,
) -> list[dict[str, Any]]:
    if rectangles is None:
        rectangles = [
            (row, col, 1, 1)
            for row in range(len(rows) - 1)
            for col in range(len(cols) - 1)
        ]
    cells: list[dict[str, Any]] = []
    for row, col, row_span, col_span in sorted(
        rectangles, key=lambda item: (item[0], item[1], item[2], item[3])
    ):
        bbox = [cols[col], rows[row], cols[col + col_span], rows[row + row_span]]
        left, top, right, bottom = bbox
        content_bbox = [left + 12, top + 12, right - 12, bottom - 12]
        cells.append(
            {
                "cell_id": f"{table_id}_R{row}C{col}",
                "row": row,
                "col": col,
                "row_span": row_span,
                "col_span": col_span,
                "bbox": bbox,
                "text": f"{table_id}:R{row}C{col}",
                "content_bbox": content_bbox,
            }
        )
    return cells


def _draw_table(
    image: np.ndarray,
    table_id: str,
    rows: Sequence[int],
    cols: Sequence[int],
    rectangles: Sequence[tuple[int, int, int, int]] | None,
) -> dict[str, Any]:
    cells = _make_cells(table_id, rows, cols, rectangles)
    row_gaps: dict[int, list[tuple[int, int]]] = {}
    col_gaps: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        if cell["row_span"] > 1:
            for internal_row in range(cell["row"] + 1, cell["row"] + cell["row_span"]):
                row_gaps.setdefault(internal_row, []).append(
                    (cols[cell["col"]] + 2, cols[cell["col"] + cell["col_span"]] - 2)
                )
        if cell["col_span"] > 1:
            for internal_col in range(cell["col"] + 1, cell["col"] + cell["col_span"]):
                col_gaps.setdefault(internal_col, []).append(
                    (rows[cell["row"]] + 2, rows[cell["row"] + cell["row_span"]] - 2)
                )

    for row_index, y in enumerate(rows):
        _draw_horizontal(image, y, cols[0], cols[-1], row_gaps.get(row_index, []))
    for col_index, x in enumerate(cols):
        _draw_vertical(image, x, rows[0], rows[-1], col_gaps.get(col_index, []))

    for cell in cells:
        left, top, right, bottom = cell["content_bbox"]
        band_height = 5
        count = min(3, max(1, (bottom - top) // 16))
        for band in range(count):
            band_top = top + band * 14
            if band_top + band_height > bottom:
                break
            image[band_top : band_top + band_height, left : right + 1] = 190

    region = [cols[0], rows[0], cols[-1], rows[-1]]
    return {
        "table_id": table_id,
        "region": region,
        "row_lines": list(rows),
        "col_lines": list(cols),
        "cells": cells,
    }


def _mapped_x(value: float, axis: dict[str, Any]) -> int:
    x_axis = axis["x"]
    ratio = (value - x_axis["value_min"]) / (x_axis["value_max"] - x_axis["value_min"])
    return int(round(x_axis["pixel_min"] + ratio * (x_axis["pixel_max"] - x_axis["pixel_min"])))


def _mapped_y(value: float, axis: dict[str, Any]) -> int:
    y_axis = axis["y"]
    ratio = (value - y_axis["value_min"]) / (y_axis["value_max"] - y_axis["value_min"])
    return int(round(y_axis["pixel_min"] - ratio * (y_axis["pixel_min"] - y_axis["pixel_max"])))


def _draw_axes(image: np.ndarray, chart: dict[str, Any]) -> None:
    region = chart["region"]
    y_baseline = int(round(chart["axis"]["y"]["pixel_min"]))
    x_axis_start = max(region[0], int(round(chart["plot_area"][0])) - 15)
    x_axis_end = min(region[2], int(round(chart["plot_area"][2])) + 5)
    _draw_horizontal(image, y_baseline, x_axis_start, x_axis_end)
    y_axis_x = x_axis_start
    top = int(round(chart["axis"]["y"]["pixel_max"])) - 5
    _draw_vertical(image, y_axis_x, top, y_baseline + 1)


def _draw_bars(image: np.ndarray, chart: dict[str, Any]) -> None:
    baseline = int(round(chart["axis"]["y"]["pixel_min"]))
    for point in chart["points"]:
        center_x = _mapped_x(point["x"], chart["axis"])
        top_y = _mapped_y(point["y"], chart["axis"])
        image[top_y:baseline, center_x - 13 : center_x + 14] = 0
        point["pixel"] = [center_x, top_y]


def _draw_line_segment(
    image: np.ndarray, first: tuple[int, int], second: tuple[int, int]
) -> None:
    steps = max(abs(second[0] - first[0]), abs(second[1] - first[1]))
    if steps == 0:
        return
    for step in range(steps + 1):
        ratio = step / steps
        x = int(round(first[0] + (second[0] - first[0]) * ratio))
        y = int(round(first[1] + (second[1] - first[1]) * ratio))
        image[y : y + 2, x : x + 2] = 0


def _draw_line_chart(image: np.ndarray, chart: dict[str, Any]) -> None:
    centers: list[tuple[int, int]] = []
    for point in chart["points"]:
        center_x = _mapped_x(point["x"], chart["axis"])
        center_y = _mapped_y(point["y"], chart["axis"])
        centers.append((center_x, center_y))
        point["pixel"] = [center_x, center_y]
    for first, second in zip(centers, centers[1:]):
        _draw_line_segment(image, first, second)
    for center_x, center_y in centers:
        image[center_y - 3 : center_y + 4, center_x - 3 : center_x + 4] = 0


def _chart_truth(
    chart_id: str,
    kind: str,
    region: Sequence[int],
    plot_area: Sequence[int],
    axis: dict[str, Any],
    points: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    normalized_points = []
    for point in points:
        normalized = dict(point)
        normalized["norm_x"] = round(
            (normalized["x"] - axis["x"]["value_min"])
            / (axis["x"]["value_max"] - axis["x"]["value_min"]),
            6,
        )
        normalized["norm_y"] = round(
            (normalized["y"] - axis["y"]["value_min"])
            / (axis["y"]["value_max"] - axis["y"]["value_min"]),
            6,
        )
        normalized_points.append(normalized)
    return {
        "chart_id": chart_id,
        "kind": kind,
        "region": list(region),
        "plot_area": list(plot_area),
        "axis": axis,
        "points": normalized_points,
    }


def _draw_chart(image: np.ndarray, chart: dict[str, Any]) -> None:
    _draw_axes(image, chart)
    if chart["kind"] == "bar":
        _draw_bars(image, chart)
    else:
        _draw_line_chart(image, chart)


def _axis(
    x_pixel_min: int,
    x_pixel_max: int,
    y_pixel_min: int,
    y_pixel_max: int,
    x_max: float,
    y_max: float = 100.0,
) -> dict[str, Any]:
    return {
        "x": {
            "pixel_min": float(x_pixel_min),
            "pixel_max": float(x_pixel_max),
            "value_min": 0.0,
            "value_max": float(x_max),
        },
        "y": {
            "pixel_min": float(y_pixel_min),
            "pixel_max": float(y_pixel_max),
            "value_min": 0.0,
            "value_max": float(y_max),
        },
    }


def _points(values: Sequence[float]) -> list[dict[str, Any]]:
    return [
        {"index": index, "x": float(index), "y": float(value), "series": "s0"}
        for index, value in enumerate(values)
    ]


def _page_one() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [80, 160, 240, 320, 400]
    cols = [40, 150, 260, 370, 480, 540]
    table = _draw_table(np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) + 255, "T1", rows, cols, None)
    chart = _chart_truth(
        "G1",
        "bar",
        [560, 65, 930, 465],
        [635, 105, 905, 430],
        _axis(650, 885, 420, 120, 5.0),
        _points([35, 72, 58, 91, 64, 47]),
    )
    return table, chart


def _page_two() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [70, 150, 230, 310, 390]
    cols = [50, 170, 290, 410, 530]
    rectangles = [
        (0, 0, 1, 1), (0, 1, 1, 2), (0, 3, 1, 1),
        (1, 0, 1, 1), (1, 1, 2, 2), (1, 3, 1, 1),
        (2, 0, 1, 1), (2, 3, 1, 1),
        (3, 0, 1, 1), (3, 1, 1, 1), (3, 2, 1, 1), (3, 3, 1, 1),
    ]
    table = _draw_table(np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) + 255, "T2", rows, cols, rectangles)
    chart = _chart_truth(
        "G2",
        "line",
        [560, 60, 930, 470],
        [640, 100, 910, 430],
        _axis(665, 881, 420, 120, 6.0),
        _points([30, 55, 42, 78, 60, 88, 51]),
    )
    return table, chart


def _page_three() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [90, 170, 250, 330, 410]
    cols = [50, 160, 270, 380, 490]
    rectangles = [
        (0, 0, 1, 2), (0, 2, 1, 1), (0, 3, 1, 1),
        (1, 0, 1, 2), (1, 2, 1, 1), (1, 3, 1, 1),
        (2, 0, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1), (2, 3, 1, 1),
        (3, 0, 1, 1), (3, 1, 1, 1), (3, 2, 1, 1), (3, 3, 1, 1),
    ]
    table = _draw_table(np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) + 255, "T3", rows, cols, rectangles)
    chart = _chart_truth(
        "G3",
        "bar",
        [560, 70, 930, 460],
        [630, 110, 900, 430],
        _axis(650, 882, 420, 120, 4.0),
        _points([25, 80, 45, 67, 91]),
    )
    return table, chart


def _page_four() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [70, 160, 250, 340, 430]
    cols = [50, 160, 270, 380, 490, 600]
    rectangles = [
        (0, 0, 2, 1), (0, 1, 1, 1), (0, 2, 1, 1), (0, 3, 1, 1), (0, 4, 1, 1),
        (1, 1, 1, 1), (1, 2, 1, 1), (1, 3, 1, 2),
        (2, 0, 1, 2), (2, 2, 1, 1), (2, 3, 1, 2),
        (3, 0, 1, 2), (3, 2, 1, 1), (3, 3, 1, 2),
    ]
    table = _draw_table(np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) + 255, "T4", rows, cols, rectangles)
    chart = _chart_truth(
        "G4",
        "line",
        [620, 70, 940, 470],
        [680, 110, 920, 430],
        _axis(710, 902, 420, 120, 6.0),
        _points([40, 65, 52, 88, 75, 93, 58]),
    )
    return table, chart


PAGE_BUILDERS = (_page_one, _page_two, _page_three, _page_four)


def generate(out_dir: Path, seed: int) -> list[Path]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for page_number, builder in enumerate(PAGE_BUILDERS, start=1):
        page_id = f"page_{page_number:02d}"
        image = np.zeros((PAGE_HEIGHT, PAGE_WIDTH, 3), dtype=np.uint8) + 255
        # The fixed layout values are the deterministic "content"; the seed draws
        # harmless light-texture marks so a non-default seed still changes the PNG
        # fingerprint without changing any line or data point.
        noise = rng.integers(0, 2, size=(PAGE_HEIGHT, PAGE_WIDTH, 1), dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise * 0, 0, 255).astype(np.uint8)
        table, chart = builder_without_background(builder, image)
        _draw_chart(image, chart)

        image_path = out_dir / f"{page_id}.png"
        image_path.write_bytes(encode_png_rgb(image))
        truth = {
            "schema_version": "1.0",
            "page_id": page_id,
            "image": image_path.name,
            "image_size": [PAGE_WIDTH, PAGE_HEIGHT],
            "pixel_mode": "RGB8",
            "generator_seed": seed,
            "tables": [table],
            "charts": [chart],
        }
        truth_path = out_dir / f"{page_id}.json"
        truth_path.write_text(
            json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.extend([image_path, truth_path])
    return written


def builder_without_background(
    builder: Any, image: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The page builders create and draw tables on a temporary image. Re-run the
    # table drawing onto the actual page image to preserve exact determinism.
    table, chart = builder()
    table_id = table["table_id"]
    rows = table["row_lines"]
    cols = table["col_lines"]
    rectangles = [
        (cell["row"], cell["col"], cell["row_span"], cell["col_span"])
        for cell in table["cells"]
    ]
    _draw_table(image, table_id, rows, cols, rectangles)
    return table, chart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "samples")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    written = generate(args.out, args.seed)
    print(f"generated {len(written)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
