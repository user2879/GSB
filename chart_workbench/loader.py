"""PNG/truth loading and strict input validation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, UnidentifiedImageError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SUPPORTED_MODES = {"L", "LA", "RGB", "RGBA"}


class ChartValidationError(ValueError):
    """A validation failure that maps to the E_CHART_INVALID business code."""


@dataclass(frozen=True)
class PageRecord:
    page: str
    image_path: Path
    truth_path: Path
    image_sha256: str
    truth_sha256: str
    width: int
    height: int
    source_mode: str
    gray: np.ndarray
    truth: Dict[str, Any]


@dataclass(frozen=True)
class LoadFailure:
    page: str
    code: str
    message: str


@dataclass
class LoadResult:
    pages: List[PageRecord]
    failures: List[LoadFailure]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChartValidationError(message)


def _validate_tick_axis(axis: Any, axis_name: str, width: int, height: int) -> Dict[str, Any]:
    _require(isinstance(axis, dict), f"axes.{axis_name} must be an object")
    positions = axis.get("pixel_positions")
    values = axis.get("values")
    _require(isinstance(positions, list) and positions, f"axes.{axis_name}.pixel_positions must be non-empty")
    _require(isinstance(values, list) and values, f"axes.{axis_name}.values must be non-empty")
    _require(len(positions) == len(values), f"axes.{axis_name} tick arrays have different lengths")

    normalized_positions: List[int] = []
    normalized_values: List[float] = []
    for index, position in enumerate(positions):
        _require(_is_int(position), f"axes.{axis_name}.pixel_positions[{index}] is not an integer")
        _require(0 <= position < width, f"axes.{axis_name}.pixel_positions[{index}] is outside image width")
        normalized_positions.append(int(position))
    for index, value in enumerate(values):
        _require(_is_number(value), f"axes.{axis_name}.values[{index}] is not finite numeric")
        normalized_values.append(float(value))

    if len(normalized_values) > 1:
        deltas = [normalized_values[i + 1] - normalized_values[i] for i in range(len(normalized_values) - 1)]
        increasing = all(delta > 0 for delta in deltas)
        decreasing = all(delta < 0 for delta in deltas)
        _require(increasing or decreasing, f"axes.{axis_name}.values are not strictly monotonic")

    if axis_name == "x":
        expected_orientation = "horizontal"
    else:
        expected_orientation = "vertical"
    orientation = axis.get("orientation", expected_orientation)
    _require(orientation == expected_orientation, f"axes.{axis_name}.orientation must be {expected_orientation}")

    return {
        "orientation": orientation,
        "pixel_positions": normalized_positions,
        "values": normalized_values,
    }


def _validate_truth(truth: Any, width: int, height: int) -> Dict[str, Any]:
    _require(isinstance(truth, dict), "truth root must be an object")
    image_size = truth.get("image_size")
    _require(
        isinstance(image_size, list) and len(image_size) == 2 and all(_is_int(v) for v in image_size),
        "truth.image_size must be [width, height] integers",
    )
    _require(tuple(image_size) == (width, height), "truth.image_size does not match PNG dimensions")

    rect = truth.get("plot_rect")
    _require(isinstance(rect, dict), "truth.plot_rect must be an object")
    for key in ("x0", "y0", "x1", "y1"):
        _require(_is_int(rect.get(key)), f"truth.plot_rect.{key} must be an integer")
    x0, y0, x1, y1 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
    _require(0 <= x0 < x1 < width and 0 <= y0 < y1 < height, "truth.plot_rect is outside the image or empty")

    axes = truth.get("axes")
    _require(isinstance(axes, dict), "truth.axes must be an object")
    normalized_axes = {
        "x": _validate_tick_axis(axes.get("x"), "x", width, height),
        "y": _validate_tick_axis(axes.get("y"), "y", width, height),
    }

    data_points = truth.get("data_points")
    _require(isinstance(data_points, list), "truth.data_points must be a list")
    normalized_points: List[Dict[str, Any]] = []
    for expected_index, point in enumerate(data_points):
        _require(isinstance(point, dict), f"truth.data_points[{expected_index}] must be an object")
        _require(point.get("index") == expected_index, "data point sequence positions are not contiguous")
        pixel = point.get("pixel")
        _require(
            isinstance(pixel, list) and len(pixel) == 2 and all(_is_int(v) for v in pixel),
            f"truth.data_points[{expected_index}].pixel must contain two integers",
        )
        px, py = int(pixel[0]), int(pixel[1])
        _require(0 <= px < width and 0 <= py < height, f"truth.data_points[{expected_index}].pixel is outside image")
        _require(_is_number(point.get("value")), f"truth.data_points[{expected_index}].value is not numeric")
        normalized_points.append({"index": expected_index, "pixel": [px, py], "value": float(point["value"])})

    result = dict(truth)
    result["image_size"] = [width, height]
    result["plot_rect"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    result["axes"] = normalized_axes
    result["data_points"] = normalized_points
    return result


def load_page(image_path: Path, base_dir: Path, truth_path: Path | None = None) -> PageRecord:
    page = image_path.relative_to(base_dir).as_posix()
    truth_path = truth_path or image_path.with_suffix(".json")
    if not truth_path.is_file():
        raise ChartValidationError(f"truth JSON is missing for {page}")

    raw = image_path.read_bytes()
    if len(raw) < len(PNG_SIGNATURE) or raw[:8] != PNG_SIGNATURE:
        raise ChartValidationError("PNG signature is invalid")

    truth_raw = truth_path.read_bytes()
    try:
        truth = json.loads(truth_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChartValidationError(f"truth JSON is not valid UTF-8 JSON: {exc}") from exc

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ChartValidationError(f"PNG container verification failed: {exc}") from exc

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ChartValidationError("container format is not PNG")
            if image.mode not in SUPPORTED_MODES:
                raise ChartValidationError(f"unsupported PNG pixel mode: {image.mode}")
            width, height = image.size
            if width <= 0 or height <= 0 or width > 10000 or height > 10000:
                raise ChartValidationError("image dimensions are outside supported bounds")
            source_mode = image.mode
            gray_image = image.convert("L")
            gray = np.asarray(gray_image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ChartValidationError(f"PNG decoding failed: {exc}") from exc

    normalized_truth = _validate_truth(truth, width, height)
    return PageRecord(
        page=page,
        image_path=image_path,
        truth_path=truth_path,
        image_sha256=hashlib.sha256(raw).hexdigest(),
        truth_sha256=hashlib.sha256(truth_raw).hexdigest(),
        width=width,
        height=height,
        source_mode=source_mode,
        gray=gray,
        truth=normalized_truth,
    )


def load_sample_directory(sample_dir: Path) -> LoadResult:
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")

    image_paths = sorted(
        (path for path in sample_dir.rglob("*.png") if path.is_file()),
        key=lambda path: path.relative_to(sample_dir).as_posix(),
    )
    pages: List[PageRecord] = []
    failures: List[LoadFailure] = []
    for image_path in image_paths:
        page = image_path.relative_to(sample_dir).as_posix()
        try:
            pages.append(load_page(image_path, sample_dir))
        except ChartValidationError as exc:
            failures.append(LoadFailure(page=page, code="E_CHART_INVALID", message=str(exc)))
        except Exception as exc:  # Defensive page isolation: one bad file cannot stop the batch.
            failures.append(LoadFailure(page=page, code="E_CHART_INVALID", message=f"unexpected validation error: {exc}"))

    failures.sort(key=lambda item: item.page)
    return LoadResult(pages=pages, failures=failures)
