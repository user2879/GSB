"""Deterministic fixed-point visual block extraction and quantization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .cache import BoundedBlockCache
from .config import FusionConfig, QuantizationProfile
from .inputs import SampleInput
from .util import derive_matrix, hash_json, round_div, saturate_signed


METRIC_COUNT = 5
PROJECTION_SPAN = 128
EDGE_THRESHOLD_PER_CHANNEL_SUM = 24


@dataclass(frozen=True)
class VisualStage:
    vector: list[int]
    blocks: list[dict[str, Any]]
    quantization: dict[str, Any]
    cache_keys: list[str]
    visual_cache_key: str
    projection_fingerprint: str
    saturation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "blocks": self.blocks,
            "quantization": self.quantization,
            "cache_keys": self.cache_keys,
            "visual_cache_key": self.visual_cache_key,
            "projection_fingerprint": self.projection_fingerprint,
            "saturation_count": self.saturation_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VisualStage":
        return cls(
            vector=[int(item) for item in value["vector"]],
            blocks=value["blocks"],
            quantization=value["quantization"],
            cache_keys=value["cache_keys"],
            visual_cache_key=value["visual_cache_key"],
            projection_fingerprint=value["projection_fingerprint"],
            saturation_count=int(value["saturation_count"]),
        )


def visual_stage_key(sample: SampleInput, config: FusionConfig, profile: QuantizationProfile) -> str:
    return hash_json(
        {
            "schema": "visual-stage-v1",
            "visual_config_hash": config.visual_config_hash,
            "image_hash": sample.image_hash,
            "profile": profile.as_dict(),
        },
        namespace="visual-stage-key-v1",
    )


@lru_cache(maxsize=32)
def _projection_matrix(namespace: str, dimension: int) -> tuple[tuple[int, ...], ...]:
    matrix = derive_matrix(namespace, METRIC_COUNT, dimension, span=PROJECTION_SPAN)
    return tuple(tuple(row) for row in matrix)


def projection_namespace(config: FusionConfig) -> str:
    return (
        "visual-projection-v1"
        f"|{config.projection_seed}"
        f"|{config.visual_config_hash}"
        f"|{config.visual_dim}"
    )


def _to_profile_scale(value_q8: int, fractional_bits: int) -> int:
    if fractional_bits >= 8:
        return value_q8 << (fractional_bits - 8)
    return round_div(value_q8, 1 << (8 - fractional_bits))


def _block_metrics(block: np.ndarray) -> list[int]:
    """Return [mean_r, mean_g, mean_b, gradient, edge] in Q8 format."""

    rgb = block[:, :, :3].astype(np.int64)
    height, width, _ = rgb.shape
    pixel_count = height * width
    channel_sums = rgb.sum(axis=(0, 1), dtype=np.int64)
    means = [round_div(int(channel_sums[channel]) * 256, pixel_count) for channel in range(3)]

    dx = np.abs(np.diff(rgb, axis=1)).sum(axis=2) if width > 1 else np.zeros((height, 0), dtype=np.int64)
    dy = np.abs(np.diff(rgb, axis=0)).sum(axis=2) if height > 1 else np.zeros((0, width), dtype=np.int64)
    gradient_numerator = int(dx.sum()) + int(dy.sum())
    gradient_denominator = dx.size + dy.size
    gradient_q8 = round_div(gradient_numerator * 256, gradient_denominator) if gradient_denominator else 0

    edge_count = int((dx >= EDGE_THRESHOLD_PER_CHANNEL_SUM).sum()) + int(
        (dy >= EDGE_THRESHOLD_PER_CHANNEL_SUM).sum()
    )
    edge_q8 = round_div(edge_count * 256, gradient_denominator) if gradient_denominator else 0
    return [*means, gradient_q8, edge_q8]


def _compute_block(
    block: np.ndarray,
    matrix: tuple[tuple[int, ...], ...],
    profile: QuantizationProfile,
) -> dict[str, Any]:
    metrics_q8 = _block_metrics(block)
    raw_values: list[int] = []
    quantized_values: list[int] = []
    residuals: list[int] = []
    saturations = 0
    for column in range(len(matrix[0])):
        projected_q8 = sum(metrics_q8[row] * matrix[row][column] for row in range(METRIC_COUNT))
        projected_q8 = round_div(projected_q8, 256 * 64)
        raw_q = _to_profile_scale(projected_q8, profile.fractional_bits)
        raw_q, saturated = saturate_signed(raw_q, profile.word_bits)
        saturations += int(saturated)
        quantized = round_div(raw_q, profile.quant_step) * profile.quant_step
        quantized, _ = saturate_signed(quantized, profile.word_bits)
        raw_values.append(raw_q)
        quantized_values.append(quantized)
        residuals.append(raw_q - quantized)
    return {
        "metrics_q8": metrics_q8,
        "raw": raw_values,
        "values": quantized_values,
        "residuals": residuals,
        "saturations": saturations,
    }


def _quantize_vector(values: list[int], profile: QuantizationProfile) -> tuple[list[int], list[int], int]:
    quantized: list[int] = []
    residuals: list[int] = []
    saturations = 0
    for value in values:
        value, saturated = saturate_signed(value, profile.word_bits)
        saturations += int(saturated)
        rounded = round_div(value, profile.quant_step) * profile.quant_step
        rounded, _ = saturate_signed(rounded, profile.word_bits)
        quantized.append(rounded)
        residuals.append(value - rounded)
    return quantized, residuals, saturations


def extract_visual(
    sample: SampleInput,
    config: FusionConfig,
    profile: QuantizationProfile,
    block_cache: BoundedBlockCache[dict[str, Any]],
) -> VisualStage:
    namespace = projection_namespace(config)
    matrix = _projection_matrix(namespace, config.visual_dim)
    block_size = config.block_size
    rows = sample.height // block_size
    columns = sample.width // block_size
    blocks: list[dict[str, Any]] = []
    cache_keys: list[str] = []
    saturation_count = 0

    for block_row in range(rows):
        for block_column in range(columns):
            block_index = block_row * columns + block_column
            cache_key = hash_json(
                {
                    "schema": "visual-block-v1",
                    "image_hash": sample.image_hash,
                    "projection": namespace,
                    "profile": profile.as_dict(),
                    "block_index": block_index,
                    "block_size": block_size,
                },
                namespace="visual-block-key-v1",
            )
            cache_keys.append(cache_key)
            y0 = block_row * block_size
            x0 = block_column * block_size
            block = sample.pixels[y0 : y0 + block_size, x0 : x0 + block_size, :].copy()
            computed = block_cache.get_or_compute(cache_key, lambda block=block: _compute_block(block, matrix, profile))
            saturation_count += int(computed["saturations"])
            blocks.append(
                {
                    "block_index": block_index,
                    "row": block_row,
                    "column": block_column,
                    "quant_step": profile.quant_step,
                    "quant_step_ratio": f"{profile.quant_step}/{profile.one}",
                    "metrics_q8": computed["metrics_q8"],
                    "raw": computed["raw"],
                    "values": computed["values"],
                    "residuals": computed["residuals"],
                    "cache_key": cache_key,
                }
            )

    block_count = len(blocks)
    aggregate_raw = [
        round_div(sum(block["values"][dimension] for block in blocks), block_count) if block_count else 0
        for dimension in range(config.visual_dim)
    ]
    vector, aggregate_residuals, aggregate_saturations = _quantize_vector(aggregate_raw, profile)
    saturation_count += aggregate_saturations
    visual_cache_key = visual_stage_key(sample, config, profile)
    quantization = {
        "profile": profile.as_dict(),
        "block_count": block_count,
        "aggregate_raw": aggregate_raw,
        "aggregate_residuals": aggregate_residuals,
    }
    return VisualStage(
        vector=vector,
        blocks=blocks,
        quantization=quantization,
        cache_keys=cache_keys,
        visual_cache_key=visual_cache_key,
        projection_fingerprint=namespace,
        saturation_count=saturation_count,
    )
