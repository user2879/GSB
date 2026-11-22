"""Stage 2: deterministic image decoding and feature extraction.

Three feature modes are supported:

* ``color_hist``: per-channel color histograms;
* ``hog``: unsigned orientation gradient histogram over a fixed cell grid;
* ``color_hog``: deterministic concatenation of both vectors.

Every float vector is converted to fixed-point integers with a preserved
residual. All operations are vectorized NumPy and do not use global RNG.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .errors import StageError
from .fixedpoint import FixedPointQuantizer

RGB_WEIGHTS = np.asarray([0.2989, 0.5870, 0.1140], dtype=np.float64)


def decode_png(png_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
    except Exception as exc:  # defensive: validation normally catches this
        raise StageError(f"PNG decode failed: {exc}") from exc
    if image.mode != "RGB":
        raise StageError(f"expected RGB image, got {image.mode!r}")
    return np.asarray(image, dtype=np.uint8)


def color_histogram(pixels: np.ndarray, bins: int) -> np.ndarray:
    """L1-normalized per-channel histogram over deterministic bin edges."""
    channels = []
    for channel in range(3):
        values = pixels[:, :, channel].reshape(-1)
        histogram, _ = np.histogram(
            values, bins=bins, range=(0, 255), density=False
        )
        total = histogram.sum(dtype=np.float64)
        normalized = histogram.astype(np.float64) / total if total > 0 else np.zeros(bins)
        channels.append(normalized)
    return np.concatenate(channels).astype(np.float64)


def _gradient_angles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # np.gradient uses fixed finite-difference edge handling, which is stable.
    grad_y, grad_x = np.gradient(gray.astype(np.float64))
    magnitude = np.hypot(grad_x, grad_y)
    # Unsigned orientations in [0, pi) mapped to [0, bins).
    angle = np.arctan2(grad_y, grad_x)
    angle = np.mod(angle, np.pi)
    return magnitude, angle


def hog_features(pixels: np.ndarray, bins: int, cell: int) -> np.ndarray:
    height, width = pixels.shape[:2]
    gray = pixels.astype(np.float64).dot(RGB_WEIGHTS)
    magnitude, angle = _gradient_angles(gray)
    cells_x = max(1, width // cell)
    cells_y = max(1, height // cell)
    usable_w = cells_x * cell
    usable_h = cells_y * cell
    magnitude = magnitude[:usable_h, :usable_w]
    angle = angle[:usable_h, :usable_w]
    bin_idx = np.floor(angle / np.pi * bins).astype(np.int64)
    np.clip(bin_idx, 0, bins - 1, out=bin_idx)

    histograms = np.zeros((cells_y, cells_x, bins), dtype=np.float64)
    for cy in range(cells_y):
        for cx in range(cells_x):
            cell_mag = magnitude[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell]
            cell_bin = bin_idx[cy * cell : (cy + 1) * cell, cx * cell : (cx + 1) * cell]
            for b in range(bins):
                histograms[cy, cx, b] = float(cell_mag[cell_bin == b].sum())
            norm = np.sqrt(np.sum(histograms[cy, cx] ** 2))
            if norm > 0:
                histograms[cy, cx] /= norm
    return histograms.reshape(-1).astype(np.float64)


def extract_float_vector(pixels: np.ndarray, config) -> np.ndarray:
    if config.feature_mode == "color_hist":
        return color_histogram(pixels, config.bins)
    if config.feature_mode == "hog":
        return hog_features(pixels, config.bins, config.hog_cell)
    color = color_histogram(pixels, config.bins)
    hog = hog_features(pixels, config.bins, config.hog_cell)
    return np.concatenate([color, hog]).astype(np.float64)


def feature_params_fingerprint_input(config) -> dict:
    return {
        "feature_mode": config.feature_mode,
        "bins": config.bins,
        "hog_cell": config.hog_cell,
        "quant_level": config.quant_level,
        "frac_bits": config.frac_bits,
        "rounding": config.rounding,
    }


def feature_vector_length(config) -> int:
    width, height = config.image_size
    color_len = config.bins * 3
    cells = max(1, width // config.hog_cell) * max(1, height // config.hog_cell)
    hog_len = cells * config.bins
    if config.feature_mode == "color_hist":
        return color_len
    if config.feature_mode == "hog":
        return hog_len
    return color_len + hog_len


@dataclass(frozen=True)
class FeatureArtifact:
    sample: str
    feature_mode: str
    bins: int
    quant_level: str
    frac_bits: int
    rounding: str
    integers: np.ndarray
    residual: np.ndarray
    image_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "feature_mode": self.feature_mode,
            "bins": self.bins,
            "quant_level": self.quant_level,
            "frac_bits": self.frac_bits,
            "rounding": self.rounding,
            "integers": [int(value) for value in self.integers],
            "residual": [float(value) for value in self.residual],
            "image_fingerprint": self.image_fingerprint,
            "length": int(len(self.integers)),
        }


def build_feature_artifact(valid_sample, config, quantizer: FixedPointQuantizer) -> FeatureArtifact:
    pixels = decode_png(valid_sample.png_bytes)
    if pixels.shape[0] != valid_sample.height or pixels.shape[1] != valid_sample.width:
        raise StageError(
            "decoded dimensions disagree with validation artifact",
            stage="features",
            sample=valid_sample.name,
        )
    vector = extract_float_vector(pixels, config)
    integers, residual = quantizer.quantize(vector)
    return FeatureArtifact(
        sample=valid_sample.name,
        feature_mode=config.feature_mode,
        bins=config.bins,
        quant_level=config.quant_level,
        frac_bits=config.frac_bits,
        rounding=config.rounding,
        integers=integers,
        residual=residual,
        image_fingerprint=valid_sample.image_fingerprint,
    )


def feature_point_cost(config) -> int:
    """Deterministic per-sample cost consumed from the feature-point budget."""
    return feature_vector_length(config)


def dequantize_matrix(artifacts: list[FeatureArtifact], quantizer: FixedPointQuantizer) -> np.ndarray:
    ordered = sorted(artifacts, key=lambda artifact: artifact.sample)
    rows = [quantizer.dequantize(artifact.integers) for artifact in ordered]
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)
