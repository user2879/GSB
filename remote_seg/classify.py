"""Deterministic windowed feature extraction and pixel classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_CONFIG_INVALID, SegmentationError
from .inputs import LoadedTile, SegmentationConfig
from .util import canonical_hash, rounded_float


FEATURE_NAMES = ("vegetation", "water", "edge", "slope")


@dataclass(slots=True)
class TileResult:
    tile_id: str
    row: int
    col: int
    cache_key: str
    content_fingerprint: str
    band_order_version: str
    result_version: str
    classes: np.ndarray
    scores: np.ndarray
    feature_summary: dict[str, Any]
    score_summary: dict[str, Any]
    class_histogram: dict[str, int]
    windows: list[dict[str, Any]]

    @property
    def order_key(self) -> tuple[int, int, str]:
        return (self.row, self.col, self.tile_id)

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "cache_key": self.cache_key,
            "content_fingerprint": self.content_fingerprint,
            "band_order_version": self.band_order_version,
            "result_version": self.result_version,
            "shape": list(self.classes.shape),
        }


def _window_origins(length: int, window: int, stride: int) -> list[int]:
    if window > length:
        raise SegmentationError(
            E_CONFIG_INVALID,
            f"window_size {window} exceeds tile dimension {length}",
            stage="classify",
        )
    values = list(range(0, length - window + 1, stride))
    last = length - window
    if not values or values[-1] != last:
        values.append(last)
    return values


def _raw_features(window_bands: np.ndarray) -> dict[str, np.ndarray]:
    red = window_bands[:, :, 0].astype(np.float64) / 255.0
    green = window_bands[:, :, 1].astype(np.float64) / 255.0
    nir = window_bands[:, :, 2].astype(np.float64) / 255.0
    elevation = window_bands[:, :, 3].astype(np.float64) / 255.0
    eps = 1e-9
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    luminance = (red + green) / 2.0
    grad_y, grad_x = np.gradient(luminance)
    edge = (np.abs(grad_x) + np.abs(grad_y)) * 0.5
    slope_y, slope_x = np.gradient(elevation)
    slope = np.sqrt(slope_x * slope_x + slope_y * slope_y)
    return {
        "vegetation": ndvi,
        "water": ndwi,
        "edge": edge,
        "slope": slope,
    }


def _feature_strengths(raw: dict[str, np.ndarray], thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    strengths: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        threshold = thresholds[name]
        values = raw[name]
        scale = 1.0 - threshold
        if scale <= 1e-9:
            strength = (values >= threshold).astype(np.float64)
        else:
            strength = np.clip((values - threshold) / scale, 0.0, 1.0)
        strengths[name] = strength
    return strengths


def _score_pixels(
    strengths: dict[str, np.ndarray], config: SegmentationConfig
) -> np.ndarray:
    pixels = next(iter(strengths.values()))
    scores = np.zeros((pixels.shape[0], pixels.shape[1], len(config.class_codes)), dtype=np.float64)
    for code in config.class_codes:
        weights = config.classification_weights[code]
        score = np.full(pixels.shape, weights["bias"], dtype=np.float64)
        for name in FEATURE_NAMES:
            score += weights[name] * strengths[name]
        scores[:, :, code] = score
    return scores


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": rounded_float(array.mean()),
        "min": rounded_float(array.min()),
        "max": rounded_float(array.max()),
    }


def classify_tile(loaded: LoadedTile, config: SegmentationConfig) -> TileResult:
    bands = loaded.array
    height, width, _ = bands.shape
    window = config.window_size
    origins_y = _window_origins(height, window, config.stride)
    origins_x = _window_origins(width, window, config.stride)

    score_sum = np.zeros((height, width, len(config.class_codes)), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.int32)
    raw_accum = {name: np.zeros((height, width), dtype=np.float64) for name in FEATURE_NAMES}
    strength_accum = {name: np.zeros((height, width), dtype=np.float64) for name in FEATURE_NAMES}
    raw_values = {name: [] for name in FEATURE_NAMES}
    windows: list[dict[str, Any]] = []

    window_index = 0
    for y0 in origins_y:
        for x0 in origins_x:
            y1 = y0 + window
            x1 = x0 + window
            ey0 = max(0, y0 - config.context_overlap)
            ex0 = max(0, x0 - config.context_overlap)
            ey1 = min(height, y1 + config.context_overlap)
            ex1 = min(width, x1 + config.context_overlap)
            expanded = bands[ey0:ey1, ex0:ex1]
            raw = _raw_features(expanded)
            strengths = _feature_strengths(raw, config.feature_thresholds)
            scores = _score_pixels(strengths, config)

            sy0 = y0 - ey0
            sx0 = x0 - ex0
            sy1 = sy0 + window
            sx1 = sx0 + window
            central_raw = {name: raw[name][sy0:sy1, sx0:sx1] for name in FEATURE_NAMES}
            central_strength = {
                name: strengths[name][sy0:sy1, sx0:sx1] for name in FEATURE_NAMES
            }
            central_scores = scores[sy0:sy1, sx0:sx1, :]

            score_sum[y0:y1, x0:x1, :] += central_scores
            coverage[y0:y1, x0:x1] += 1
            for name in FEATURE_NAMES:
                raw_accum[name][y0:y1, x0:x1] += central_raw[name]
                strength_accum[name][y0:y1, x0:x1] += central_strength[name]
                raw_values[name].extend(central_raw[name].ravel().tolist())

            window_classes = np.argmax(central_scores, axis=2).astype(np.uint8)
            counts = np.bincount(window_classes.ravel(), minlength=len(config.class_codes))
            stable_key = (
                f"{loaded.spec.row:08d}:{loaded.spec.col:08d}:classify:"
                f"{window_index:08d}:{y0:08d}:{x0:08d}"
            )
            windows.append(
                {
                    "window_index": window_index,
                    "bbox": [x0, y0, x1, y1],
                    "expanded_bbox": [ex0, ey0, ex1, ey1],
                    "stable_sort_key": stable_key,
                    "task_key_sha256": canonical_hash(stable_key, prefix="task-v1|"),
                    "feature_means": {
                        name: rounded_float(float(central_raw[name].mean())) for name in FEATURE_NAMES
                    },
                    "class_histogram": {
                        str(code): int(counts[code]) for code in config.class_codes
                    },
                }
            )
            window_index += 1

    if np.any(coverage == 0):
        raise SegmentationError(
            E_CONFIG_INVALID,
            "sliding windows did not cover every tile pixel",
            stage="classify",
        )
    final_scores = score_sum / coverage[:, :, None]
    classes = np.argmax(final_scores, axis=2).astype(np.uint8)
    histogram_array = np.bincount(classes.ravel(), minlength=len(config.class_codes))

    feature_summary = {
        "raw_means": {
            name: rounded_float(float((raw_accum[name] / coverage).mean())) for name in FEATURE_NAMES
        },
        "strength_means": {
            name: rounded_float(float((strength_accum[name] / coverage).mean()))
            for name in FEATURE_NAMES
        },
        "raw_ranges": {name: _summary(raw_values[name]) for name in FEATURE_NAMES},
    }
    score_summary = {
        str(code): {
            "mean": rounded_float(float(final_scores[:, :, code].mean())),
            "min": rounded_float(float(final_scores[:, :, code].min())),
            "max": rounded_float(float(final_scores[:, :, code].max())),
        }
        for code in config.class_codes
    }
    class_histogram = {
        str(code): int(histogram_array[code]) for code in config.class_codes
    }
    result_version = canonical_hash(
        {
            "cache_key": loaded.cache_key,
            "window_size": config.window_size,
            "stride": config.stride,
            "context_overlap": config.context_overlap,
            "class_histogram": class_histogram,
            "score_summary": score_summary,
        },
        prefix="tile-result-v1|",
    )
    return TileResult(
        tile_id=loaded.spec.tile_id,
        row=loaded.spec.row,
        col=loaded.spec.col,
        cache_key=loaded.cache_key,
        content_fingerprint=loaded.content_fingerprint,
        band_order_version=loaded.spec.band_order_version,
        result_version=result_version,
        classes=classes,
        scores=final_scores,
        feature_summary=feature_summary,
        score_summary=score_summary,
        class_histogram=class_histogram,
        windows=windows,
    )
