"""Page truth schema validation and business-code error handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class PageInvalid(Exception):
    detail_code: str
    message: str
    business_code: str = "E_PAGE_INVALID"

    def __str__(self) -> str:
        return f"{self.detail_code}: {self.message}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect_int(value: Any, path: str, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise PageInvalid("E_TRUTH_SCHEMA", f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise PageInvalid("E_TRUTH_SCHEMA", f"{path} must be >= {minimum}")
    return value


def _expect_box(
    value: Any, path: str, width: int, height: int
) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise PageInvalid("E_TRUTH_SCHEMA", f"{path} must be [left, top, right, bottom]")
    left, top, right, bottom = (
        _expect_int(value[0], f"{path}[0]"),
        _expect_int(value[1], f"{path}[1]"),
        _expect_int(value[2], f"{path}[2]"),
        _expect_int(value[3], f"{path}[3]"),
    )
    if not (0 <= left < right < width and 0 <= top < bottom < height):
        raise PageInvalid(
            "E_TRUTH_OUT_OF_BOUNDS",
            f"{path} must satisfy 0<=left<right<{width} and 0<=top<bottom<{height}",
        )
    return left, top, right, bottom


def _strict_increasing(values: list[int], path: str, low: int, high: int) -> None:
    if any(value < low or value > high for value in values):
        raise PageInvalid(
            "E_TRUTH_OUT_OF_BOUNDS", f"{path} contains a coordinate outside [{low},{high}]"
        )
    if any(values[index] >= values[index + 1] for index in range(len(values) - 1)):
        raise PageInvalid(
            "E_TRUTH_LINE_ORDER", f"{path} must be strictly increasing and unique"
        )


def normalize_truth(truth: Any, width: int, height: int) -> dict[str, Any]:
    if not isinstance(truth, Mapping):
        raise PageInvalid("E_TRUTH_SCHEMA", "truth root must be an object")
    normalized = dict(truth)

    tables_raw = normalized.get("tables", [])
    charts_raw = normalized.get("charts", [])
    if "table" in normalized and not tables_raw:
        tables_raw = [normalized["table"]]
    if "chart" in normalized and not charts_raw:
        charts_raw = [normalized["chart"]]
    if not isinstance(tables_raw, list) or not isinstance(charts_raw, list):
        raise PageInvalid("E_TRUTH_SCHEMA", "tables/charts must be arrays")
    if not tables_raw and not charts_raw:
        raise PageInvalid("E_TRUTH_SCHEMA", "a page must contain at least one table or chart")

    tables = [
        _normalize_table(table, index, width, height)
        for index, table in enumerate(tables_raw)
    ]
    charts = [
        _normalize_chart(chart, index, width, height)
        for index, chart in enumerate(charts_raw)
    ]

    table_ids = [table["table_id"] for table in tables]
    chart_ids = [chart["chart_id"] for chart in charts]
    if len(set(table_ids)) != len(table_ids):
        raise PageInvalid("E_TRUTH_SCHEMA", "duplicate table_id")
    if len(set(chart_ids)) != len(chart_ids):
        raise PageInvalid("E_TRUTH_SCHEMA", "duplicate chart_id")

    normalized["tables"] = tables
    normalized["charts"] = charts
    normalized.pop("table", None)
    normalized.pop("chart", None)
    return normalized


def _normalize_table(
    table: Any, index: int, width: int, height: int
) -> dict[str, Any]:
    if not isinstance(table, Mapping):
        raise PageInvalid("E_TRUTH_SCHEMA", f"tables[{index}] must be an object")
    result = dict(table)
    table_id = result.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"tables[{index}].table_id must be a non-empty string"
        )
    region = _expect_box(
        result.get("region"), f"tables[{index}].region", width, height
    )
    row_lines = result.get("row_lines")
    col_lines = result.get("col_lines")
    if not isinstance(row_lines, list) or not isinstance(col_lines, list):
        raise PageInvalid(
            "E_TRUTH_SCHEMA",
            f"tables[{index}].row_lines and col_lines must be arrays",
        )
    row_lines = [
        _expect_int(value, f"tables[{index}].row_lines[{line_index}]")
        for line_index, value in enumerate(row_lines)
    ]
    col_lines = [
        _expect_int(value, f"tables[{index}].col_lines[{line_index}]")
        for line_index, value in enumerate(col_lines)
    ]
    if len(row_lines) < 2 or len(col_lines) < 2:
        raise PageInvalid(
            "E_TRUTH_SCHEMA",
            f"tables[{index}] needs at least two row and two column lines",
        )
    _strict_increasing(row_lines, f"tables[{index}].row_lines", region[1], region[3])
    _strict_increasing(col_lines, f"tables[{index}].col_lines", region[0], region[2])

    cells_raw = result.get("cells")
    if not isinstance(cells_raw, list) or not cells_raw:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"tables[{index}].cells must be a non-empty array"
        )
    cells = [
        _normalize_cell(
            cell,
            table_index=index,
            cell_index=cell_index,
            row_lines=row_lines,
            col_lines=col_lines,
            width=width,
            height=height,
        )
        for cell_index, cell in enumerate(cells_raw)
    ]

    coverage: dict[tuple[int, int], str] = {}
    for cell in cells:
        for row in range(cell["row"], cell["row"] + cell["row_span"]):
            for col in range(cell["col"], cell["col"] + cell["col_span"]):
                key = (row, col)
                if key in coverage:
                    raise PageInvalid(
                        "E_TRUTH_CELL_OVERLAP",
                        f"tables[{index}] cells {coverage[key]} and {cell['cell_id']} overlap",
                    )
                coverage[key] = cell["cell_id"]
    required = {
        (row, col)
        for row in range(len(row_lines) - 1)
        for col in range(len(col_lines) - 1)
    }
    if set(coverage) != required:
        missing = sorted(required.difference(coverage))
        raise PageInvalid(
            "E_TRUTH_CELL_INCOMPLETE",
            f"tables[{index}] lacks cell coverage at {missing[:4]}",
        )

    result["region"] = list(region)
    result["row_lines"] = row_lines
    result["col_lines"] = col_lines
    result["cells"] = cells
    return result


def _normalize_cell(
    cell: Any,
    *,
    table_index: int,
    cell_index: int,
    row_lines: list[int],
    col_lines: list[int],
    width: int,
    height: int,
) -> dict[str, Any]:
    path = f"tables[{table_index}].cells[{cell_index}]"
    if not isinstance(cell, Mapping):
        raise PageInvalid("E_TRUTH_SCHEMA", f"{path} must be an object")
    result = dict(cell)
    cell_id = result.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"{path}.cell_id must be a non-empty string"
        )
    row = _expect_int(result.get("row"), f"{path}.row", 0)
    col = _expect_int(result.get("col"), f"{path}.col", 0)
    row_span = _expect_int(result.get("row_span"), f"{path}.row_span", 1)
    col_span = _expect_int(result.get("col_span"), f"{path}.col_span", 1)
    if row + row_span > len(row_lines) - 1:
        raise PageInvalid("E_TRUTH_CELL_RANGE", f"{path}.row_span exceeds table rows")
    if col + col_span > len(col_lines) - 1:
        raise PageInvalid("E_TRUTH_CELL_RANGE", f"{path}.col_span exceeds table columns")
    expected_bbox = [
        col_lines[col],
        row_lines[row],
        col_lines[col + col_span],
        row_lines[row + row_span],
    ]
    bbox = _expect_box(result.get("bbox"), f"{path}.bbox", width, height)
    if list(bbox) != expected_bbox:
        raise PageInvalid(
            "E_TRUTH_CELL_BBOX",
            f"{path}.bbox must equal the span-derived rectangle {expected_bbox}",
        )
    text = result.get("text")
    if not isinstance(text, str) or not text:
        raise PageInvalid("E_TRUTH_CELL_TEXT", f"{path}.text must be a non-empty string")
    if "content_bbox" in result:
        content_bbox = _expect_box(
            result["content_bbox"], f"{path}.content_bbox", width, height
        )
        left, top, right, bottom = bbox
        if not (
            left < content_bbox[0] < content_bbox[2] < right
            and top < content_bbox[1] < content_bbox[3] < bottom
        ):
            raise PageInvalid(
                "E_TRUTH_CELL_CONTENT", f"{path}.content_bbox must lie inside bbox"
            )
        result["content_bbox"] = list(content_bbox)
    result.update(
        {
            "row": row,
            "col": col,
            "row_span": row_span,
            "col_span": col_span,
            "bbox": list(bbox),
        }
    )
    return result


def _normalize_chart(
    chart: Any, index: int, width: int, height: int
) -> dict[str, Any]:
    if not isinstance(chart, Mapping):
        raise PageInvalid("E_TRUTH_SCHEMA", f"charts[{index}] must be an object")
    result = dict(chart)
    chart_id = result.get("chart_id")
    if not isinstance(chart_id, str) or not chart_id:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"charts[{index}].chart_id must be a non-empty string"
        )
    region = _expect_box(result.get("region"), f"charts[{index}].region", width, height)
    kind = result.get("kind")
    kind_aliases = {"column": "bar", "polyline": "line"}
    if isinstance(kind, str):
        kind = kind_aliases.get(kind, kind)
        result["kind"] = kind
    if kind not in {"bar", "line"}:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"charts[{index}].kind must be 'bar' or 'line'"
        )

    plot_area = result.get("plot_area", list(region))
    plot_area = _expect_box(plot_area, f"charts[{index}].plot_area", width, height)
    if not (
        region[0] <= plot_area[0] < plot_area[2] <= region[2]
        and region[1] <= plot_area[1] < plot_area[3] <= region[3]
    ):
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"charts[{index}].plot_area must lie inside region"
        )

    axis = result.get("axis")
    if not isinstance(axis, Mapping):
        raise PageInvalid("E_TRUTH_SCHEMA", f"charts[{index}].axis must be an object")
    normalized_axis: dict[str, dict[str, Any]] = {}
    for axis_name in ("x", "y"):
        axis_part = axis.get(axis_name)
        if not isinstance(axis_part, Mapping):
            raise PageInvalid(
                "E_TRUTH_SCHEMA", f"charts[{index}].axis.{axis_name} must be an object"
            )
        pixel_min = axis_part.get("pixel_min")
        pixel_max = axis_part.get("pixel_max")
        value_min = axis_part.get("value_min")
        value_max = axis_part.get("value_max")
        if not _is_number(pixel_min) or not _is_number(pixel_max):
            raise PageInvalid(
                "E_TRUTH_SCHEMA",
                f"charts[{index}].axis.{axis_name} pixel bounds must be numbers",
            )
        if not _is_number(value_min) or not _is_number(value_max):
            raise PageInvalid(
                "E_TRUTH_SCHEMA",
                f"charts[{index}].axis.{axis_name} value bounds must be numbers",
            )
        if float(value_min) >= float(value_max):
            raise PageInvalid(
                "E_TRUTH_SCHEMA",
                f"charts[{index}].axis.{axis_name} value_min < value_max required",
            )
        normalized_axis[axis_name] = {
            "pixel_min": float(pixel_min),
            "pixel_max": float(pixel_max),
            "value_min": float(value_min),
            "value_max": float(value_max),
        }

    x_axis = normalized_axis["x"]
    y_axis = normalized_axis["y"]
    if not (
        0 <= x_axis["pixel_min"] <= width - 1
        and 0 <= x_axis["pixel_max"] <= width - 1
        and 0 <= y_axis["pixel_min"] <= height - 1
        and 0 <= y_axis["pixel_max"] <= height - 1
    ):
        raise PageInvalid(
            "E_TRUTH_OUT_OF_BOUNDS", f"charts[{index}] axis pixels are outside image"
        )
    if not (
        plot_area[0] <= x_axis["pixel_min"] < x_axis["pixel_max"] <= plot_area[2]
        and plot_area[1] <= y_axis["pixel_max"] < y_axis["pixel_min"] <= plot_area[3]
    ):
        raise PageInvalid(
            "E_TRUTH_AXIS",
            f"charts[{index}] axis pixels must lie inside plot_area with y pixel_max above pixel_min",
        )

    points_raw = result.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise PageInvalid(
            "E_TRUTH_SCHEMA", f"charts[{index}].points must be a non-empty array"
        )
    points: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for point_index, point in enumerate(points_raw):
        path = f"charts[{index}].points[{point_index}]"
        if not isinstance(point, Mapping):
            raise PageInvalid("E_TRUTH_SCHEMA", f"{path} must be an object")
        point_result = dict(point)
        sequence = _expect_int(point_result.get("index"), f"{path}.index", 0)
        if sequence in seen_indices:
            raise PageInvalid("E_TRUTH_POINT_INDEX", f"{path}.index is duplicated")
        seen_indices.add(sequence)
        x_value = point_result.get("x")
        y_value = point_result.get("y")
        if not _is_number(x_value) or not _is_number(y_value):
            raise PageInvalid("E_TRUTH_SCHEMA", f"{path}.x/y must be numbers")
        series = point_result.get("series", "s0")
        if not isinstance(series, str) or not series:
            raise PageInvalid("E_TRUTH_SCHEMA", f"{path}.series must be a string")
        if "pixel" in point_result:
            pixel = point_result["pixel"]
            if not isinstance(pixel, list) or len(pixel) != 2 or not all(
                _is_number(value) for value in pixel
            ):
                raise PageInvalid("E_TRUTH_SCHEMA", f"{path}.pixel must be [x,y]")
            if not (
                region[0] <= float(pixel[0]) <= region[2]
                and region[1] <= float(pixel[1]) <= region[3]
            ):
                raise PageInvalid(
                    "E_TRUTH_OUT_OF_BOUNDS", f"{path}.pixel lies outside chart region"
                )
            point_result["pixel"] = [float(pixel[0]), float(pixel[1])]
        point_result.update(
            {"index": sequence, "x": float(x_value), "y": float(y_value), "series": series}
        )
        points.append(point_result)

    result.update(
        {
            "region": list(region),
            "plot_area": list(plot_area),
            "kind": kind,
            "axis": normalized_axis,
            "points": points,
        }
    )
    return result


def region_signature(truth: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Small stable representation used by stage cache keys."""

    if stage == "grid":
        return {
            "tables": [
                {
                    "table_id": table["table_id"],
                    "region": table["region"],
                    "row_lines": table["row_lines"],
                    "col_lines": table["col_lines"],
                    "cells": [
                        {
                            "cell_id": cell["cell_id"],
                            "row": cell["row"],
                            "col": cell["col"],
                            "row_span": cell["row_span"],
                            "col_span": cell["col_span"],
                            "text": cell["text"],
                        }
                        for cell in table["cells"]
                    ],
                }
                for table in truth["tables"]
            ]
        }
    if stage == "chart":
        return {
            "charts": [
                {
                    "chart_id": chart["chart_id"],
                    "region": chart["region"],
                    "plot_area": chart["plot_area"],
                    "kind": chart["kind"],
                    "axis": chart["axis"],
                    "point_count": len(chart["points"]),
                    "point_metadata": [
                        {"index": point["index"], "series": point["series"]}
                        for point in sorted(chart["points"], key=lambda item: item["index"])
                    ],
                }
                for chart in truth["charts"]
            ],
            "tables": [table["region"] for table in truth["tables"]],
        }
    raise ValueError(f"unknown signature stage: {stage}")


def all_base_cells(table: Mapping[str, Any]) -> Iterable[tuple[int, int, dict[str, Any]]]:
    for cell in table["cells"]:
        for row in range(cell["row"], cell["row"] + cell["row_span"]):
            for col in range(cell["col"], cell["col"] + cell["col_span"]):
                yield row, col, cell
