"""PNG decoding and deterministic, quantized image feature extraction."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import ImageDecodeError
from .util import sha256_bytes


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FEATURE_ALGORITHM_VERSION = 1


def decode_png(data: bytes, expected_size: tuple[int, int]) -> dict[str, Any]:
    if len(data) < len(PNG_MAGIC) or data[:8] != PNG_MAGIC:
        raise ImageDecodeError("invalid PNG container header")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format != "PNG":
                raise ImageDecodeError("image format is not PNG")
            if probe.size != expected_size:
                raise ImageDecodeError(
                    f"image size {probe.size[0]}x{probe.size[1]} does not match configured "
                    f"{expected_size[0]}x{expected_size[1]}"
                )
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            if image.size != expected_size:
                raise ImageDecodeError("image size changed during decode")
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except ImageDecodeError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as exc:
        raise ImageDecodeError(f"PNG decode failed: {type(exc).__name__}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageDecodeError("decoded image is not an RGB uint8 array")
    if array.dtype != np.uint8:
        raise ImageDecodeError("decoded image array is not uint8")
    return {
        "array": array,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "pixel_sha256": sha256_bytes(array.tobytes(order="C")),
    }


def _quantize_vector(values: np.ndarray, levels: int, decimals: int) -> np.ndarray:
    quantized = np.rint(values.astype(np.float64) * int(levels)) / float(levels)
    quantized = np.round(quantized, int(decimals))
    quantized[np.abs(quantized) < 10 ** (-decimals) / 2.0] = 0.0
    return quantized.astype(np.float64)


def color_histogram(array: np.ndarray, bins: int) -> np.ndarray:
    pixels = array.shape[0] * array.shape[1]
    quantized = (array.astype(np.uint32) * int(bins)) // 256
    values: list[np.ndarray] = []
    for channel in range(3):
        histogram = np.bincount(
            quantized[:, :, channel].reshape(-1), minlength=int(bins)
        ).astype(np.float64)
        values.append(histogram / max(1, pixels))
    return np.concatenate(values, axis=0)


def hog_descriptor(
    array: np.ndarray,
    orientation_bins: int,
    cells: tuple[int, int],
) -> np.ndarray:
    # Integer Rec.601 luma keeps the grayscale input platform-stable.
    luma = (
        299 * array[:, :, 0].astype(np.int32)
        + 587 * array[:, :, 1].astype(np.int32)
        + 114 * array[:, :, 2].astype(np.int32)
        + 500
    ) // 1000
    luma_float = luma.astype(np.float64)
    gradient_x = np.zeros_like(luma_float)
    gradient_y = np.zeros_like(luma_float)
    gradient_x[:, 1:-1] = luma_float[:, 2:] - luma_float[:, :-2]
    gradient_y[1:-1, :] = luma_float[2:, :] - luma_float[:-2, :]
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180.0
    orientation = np.floor(angle / (180.0 / int(orientation_bins))).astype(np.int32)
    orientation = np.clip(orientation, 0, int(orientation_bins) - 1)

    cells_y, cells_x = int(cells[0]), int(cells[1])
    height, width = luma.shape
    features: list[np.ndarray] = []
    for cell_y in range(cells_y):
        y0 = (cell_y * height) // cells_y
        y1 = ((cell_y + 1) * height) // cells_y
        for cell_x in range(cells_x):
            x0 = (cell_x * width) // cells_x
            x1 = ((cell_x + 1) * width) // cells_x
            cell_magnitude = magnitude[y0:y1, x0:x1].reshape(-1)
            cell_orientation = orientation[y0:y1, x0:x1].reshape(-1)
            histogram = np.bincount(
                cell_orientation,
                weights=cell_magnitude,
                minlength=int(orientation_bins),
            ).astype(np.float64)
            total = float(histogram.sum())
            if total > 0.0:
                histogram = histogram / total
            features.append(histogram)
    return np.concatenate(features, axis=0)


def extract_features(array: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("feature extraction requires an RGB uint8 array")
    method = str(params["feature_method"])
    color_bins = int(params["color_bins"])
    hog_bins = int(params["hog_bins"])
    hog_cells = (int(params["hog_cells"][0]), int(params["hog_cells"][1]))
    levels = int(params["quantization_levels"])
    decimals = int(params["rounding_decimals"])
    if method == "color":
        raw = color_histogram(array, color_bins)
    elif method == "hog":
        raw = hog_descriptor(array, hog_bins, hog_cells)
    elif method == "concat":
        raw = np.concatenate(
            [color_histogram(array, color_bins), hog_descriptor(array, hog_bins, hog_cells)],
            axis=0,
        )
    else:
        raise ValueError(f"unsupported feature method: {method}")
    vector = _quantize_vector(raw, levels, decimals)
    expected_dimension = int(params["feature_dimension"])
    if vector.shape != (expected_dimension,):
        raise ValueError(
            f"feature dimension mismatch: got {vector.shape[0]}, expected {expected_dimension}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("feature vector contains non-finite values")
    metadata = {
        "feature_method": method,
        "color_bins": color_bins,
        "hog_bins": hog_bins,
        "hog_cells": list(hog_cells),
        "feature_dimension": int(vector.shape[0]),
        "quantization_levels": levels,
        "precision_level": params.get("precision_level"),
        "rounding_decimals": decimals,
        "rounding_mode": params.get("rounding_mode"),
        "quantization_mode": params.get("quantization_mode"),
    }
    return vector, metadata
