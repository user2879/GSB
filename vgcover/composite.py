"""Premultiplied BGRA source-over compositing with deterministic rounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import CoverageError, ErrorCode
from .hashing import hash_uint8, sha256_json
from .scene import validate_alpha


COMPOSITE_POLICY = {
    "version": "vgcover.premultiplied_bgra8.source_over.v1",
    "channel_order": "BGRA",
    "storage": "premultiplied BGR channels, straight alpha channel, uint8",
    "background": [0, 0, 0, 0],
    "order": "region_id ascending",
    "rounding": "nearest_half_up",
    "rounding_formula": "floor(numerator / 255 + 0.5)",
    "effective_alpha_formula": "a8 = floor(gray * alpha + 0.5)",
    "premultiplied_color_formula": "c_p = floor(color * a8 / 255 + 0.5)",
    "source_over_formula": "out = src_premultiplied + floor(dst * (255 - src_alpha) / 255 + 0.5)",
    "per_step_channel_rounding_error_bound_u8": 2.0,
    "per_step_coverage_error_bound": 2.0 / 255.0,
}


def _round_div_255(numerator: np.ndarray | int) -> np.ndarray:
    """Round a non-negative integer numerator divided by 255, ties impossible."""

    values = np.asarray(numerator, dtype=np.uint32)
    return ((values + np.uint32(127)) // np.uint32(255)).astype(np.uint32)


def _validate_color(fill_color: Sequence[int], region_id: int | None) -> tuple[int, int, int]:
    if (
        not isinstance(fill_color, (list, tuple))
        or len(fill_color) != 3
        or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in fill_color)
    ):
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "fill_color must be three integer RGB channels in [0, 255]",
            region_id=region_id,
            details={"received": list(fill_color) if hasattr(fill_color, "__iter__") else fill_color},
        )
    return int(fill_color[0]), int(fill_color[1]), int(fill_color[2])


@dataclass
class SourceLayer:
    region_id: int
    gray: np.ndarray
    fill_color: tuple[int, int, int]
    alpha: float
    bgra: np.ndarray
    nonzero_source_pixels: int
    nonzero_gray_pixels: int
    layer_content_hash: str
    rounding_error_bound_u8: float
    rounding_error_bound_coverage: float


@dataclass
class CompositeStep:
    position: int
    region_id: int
    layer_content_hash: str
    nonzero_source_pixels: int
    step_rounding_error_bound_u8: float
    cumulative_rounding_error_bound_u8: float
    step_rounding_error_bound_coverage: float


@dataclass
class CompositeResult:
    buffer: np.ndarray
    buffer_hash: str
    layer_sequence: list[int]
    layer_hashes: dict[int, str]
    steps: list[CompositeStep]
    policy: dict[str, Any]


def build_source_layer(
    *,
    region_id: int,
    gray: np.ndarray,
    fill_color: Sequence[int],
    alpha: float,
) -> SourceLayer:
    if not isinstance(region_id, int) or isinstance(region_id, bool) or region_id < 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "region_id must be a non-negative integer",
            details={"received": region_id},
        )
    validate_alpha(alpha, region_id=region_id)
    rgb = _validate_color(fill_color, region_id)
    coverage = np.asarray(gray)
    if coverage.ndim != 2 or coverage.dtype != np.uint8:
        raise CoverageError(
            ErrorCode.COVERAGE_RANGE,
            "gray layer must be a two-dimensional uint8 array",
            region_id=region_id,
            details={"dtype": str(coverage.dtype), "shape": list(coverage.shape)},
        )

    alpha_codes = np.floor(coverage.astype(np.float64) * float(alpha) + 0.5).astype(np.uint32)
    alpha_codes = np.clip(alpha_codes, 0, 255).astype(np.uint32)
    color = np.asarray(rgb, dtype=np.uint32)
    premultiplied = _round_div_255(color[None, None, :] * alpha_codes[:, :, None])
    bgra = np.zeros((*coverage.shape, 4), dtype=np.uint8)
    bgra[:, :, 0] = premultiplied[:, :, 2].astype(np.uint8)
    bgra[:, :, 1] = premultiplied[:, :, 1].astype(np.uint8)
    bgra[:, :, 2] = premultiplied[:, :, 0].astype(np.uint8)
    bgra[:, :, 3] = alpha_codes.astype(np.uint8)

    content_hash = sha256_json(
        {
            "domain": "vgcover.layer_content.v1",
            "region_id": region_id,
            "shape": list(bgra.shape),
            "dtype": "<u1",
            "bgra_hash": hash_uint8(bgra),
        }
    )
    bound_u8 = float(COMPOSITE_POLICY["per_step_channel_rounding_error_bound_u8"])
    return SourceLayer(
        region_id=region_id,
        gray=coverage,
        fill_color=rgb,
        alpha=float(alpha),
        bgra=bgra,
        nonzero_source_pixels=int(np.count_nonzero(alpha_codes != 0)),
        nonzero_gray_pixels=int(np.count_nonzero(coverage != 0)),
        layer_content_hash=content_hash,
        rounding_error_bound_u8=bound_u8,
        rounding_error_bound_coverage=bound_u8 / 255.0,
    )


def _apply_source_over(destination: np.ndarray, source: np.ndarray) -> np.ndarray:
    source_alpha = source[:, :, 3].astype(np.uint32)
    inverse_alpha = (np.uint32(255) - source_alpha).astype(np.uint32)
    destination_u32 = destination.astype(np.uint32)
    scaled_destination = _round_div_255(destination_u32 * inverse_alpha[:, :, None])
    output = (source.astype(np.uint32) + scaled_destination).astype(np.uint32)
    return np.clip(output, 0, 255).astype(np.uint8)


def compose_layers(layers: Iterable[SourceLayer]) -> CompositeResult:
    ordered = sorted(list(layers), key=lambda layer: layer.region_id)
    if not ordered:
        raise CoverageError(ErrorCode.SCENE_INVALID, "at least one source layer is required")
    region_ids = [layer.region_id for layer in ordered]
    if len(set(region_ids)) != len(region_ids):
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "source layer region_id values must be unique",
            details={"region_ids": region_ids},
        )
    shape = ordered[0].bgra.shape
    for layer in ordered[1:]:
        if layer.bgra.shape != shape:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "source layers must share the same canvas shape",
                region_id=layer.region_id,
                details={"expected": list(shape), "received": list(layer.bgra.shape)},
            )

    destination = np.zeros(shape, dtype=np.uint8)
    steps: list[CompositeStep] = []
    layer_hashes: dict[int, str] = {}
    cumulative_u8 = 0.0
    for position, layer in enumerate(ordered):
        destination = _apply_source_over(destination, layer.bgra)
        cumulative_u8 += layer.rounding_error_bound_u8
        layer_hashes[layer.region_id] = layer.layer_content_hash
        steps.append(
            CompositeStep(
                position=position,
                region_id=layer.region_id,
                layer_content_hash=layer.layer_content_hash,
                nonzero_source_pixels=layer.nonzero_source_pixels,
                step_rounding_error_bound_u8=layer.rounding_error_bound_u8,
                cumulative_rounding_error_bound_u8=cumulative_u8,
                step_rounding_error_bound_coverage=layer.rounding_error_bound_coverage,
            )
        )
    return CompositeResult(
        buffer=destination,
        buffer_hash=hash_uint8(destination),
        layer_sequence=region_ids,
        layer_hashes=layer_hashes,
        steps=steps,
        policy=dict(COMPOSITE_POLICY),
    )
