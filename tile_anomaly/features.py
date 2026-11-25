"""Spectrum, texture, elevation features and fixed-point observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import quantization_key, window_parameter_key
from .tiles import Mosaic, TileData
from .util import decimal_string, round_half_away_from_zero, sha256_json
from .windows import WindowSpec


DIMENSION_NAMES = [
    "vegetation_index",
    "normalized_water_index",
    "texture_gradient",
    "elevation_slope",
]
FEATURE_ALGORITHM_VERSION = "features-v1"


def _gradient_magnitude_mean(array: np.ndarray) -> float:
    row_gradient, col_gradient = np.gradient(array)
    magnitude = np.hypot(row_gradient, col_gradient)
    return float(np.mean(magnitude, dtype=np.float64))


def raw_features(
    window: WindowSpec,
    mosaic: Mosaic,
    uploader_tiles: dict[str, TileData],
) -> dict[str, float]:
    del uploader_tiles
    top, left, bottom, right = window.context_rect
    height = bottom - top
    width = right - left
    red = mosaic.extraction("red", top, left, height, width)
    green = mosaic.extraction("green", top, left, height, width)
    blue = mosaic.extraction("blue", top, left, height, width)
    elevation = mosaic.extraction("elevation", top, left, height, width) / 65535.0
    red_sum = np.sum(red, dtype=np.float64)
    green_sum = np.sum(green, dtype=np.float64)
    blue_sum = np.sum(blue, dtype=np.float64)
    vegetation_numerator = green_sum - red_sum
    vegetation_denominator = green_sum + red_sum + 1.0e-9
    water_numerator = blue_sum - red_sum
    water_denominator = blue_sum + red_sum + 1.0e-9
    vegetation_index = float(vegetation_numerator / vegetation_denominator)
    water_index = float(water_numerator / water_denominator)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    texture_gradient = _gradient_magnitude_mean(luminance)
    elevation_slope = _gradient_magnitude_mean(elevation)
    return {
        "vegetation_index": vegetation_index,
        "normalized_water_index": water_index,
        "texture_gradient": texture_gradient,
        "elevation_slope": elevation_slope,
        "spectral_ratio_inputs": {
            "vegetation_numerator": vegetation_numerator,
            "vegetation_denominator": vegetation_denominator,
            "water_numerator": water_numerator,
            "water_denominator": water_denominator,
        },
    }


def quantize_dimension(
    real_value: float,
    dimension: str,
    fraction_bits: int,
    gear: int,
    dimension_scales_milli: dict[str, int],
) -> dict[str, Any]:
    scale_milli = int(dimension_scales_milli[dimension])
    denominator = (1 << fraction_bits) * gear * scale_milli
    scaled = real_value * denominator / 1000.0
    quantized = round_half_away_from_zero(scaled)
    residual_scaled = scaled - quantized
    step_numerator = 1000
    step_denominator = denominator
    common = math.gcd(step_numerator, step_denominator)
    step_numerator //= common
    step_denominator //= common
    return {
        "name": dimension,
        "real": float(real_value),
        "real_decimal": decimal_string(real_value),
        "quant_step_numerator": step_numerator,
        "quant_step_denominator": step_denominator,
        "quant_step_decimal": decimal_string(step_numerator / step_denominator),
        "q": int(quantized),
        "dequantized": float(quantized * step_numerator / step_denominator),
        "dequantized_decimal": decimal_string(
            quantized * step_numerator / step_denominator
        ),
        "residual": float(residual_scaled * step_numerator / step_denominator),
        "residual_in_steps": float(residual_scaled),
        "residual_in_steps_decimal": decimal_string(residual_scaled),
    }


def _quant_params(config: dict[str, Any], fraction_bits: int | None, gear: int | None) -> tuple[int, int]:
    bits = int(config["quantization"]["fraction_bits"] if fraction_bits is None else fraction_bits)
    selected_gear = int(config["quantization"]["gear"] if gear is None else gear)
    return bits, selected_gear


def feature_key_for_window(
    window: WindowSpec,
    tiles: dict[str, TileData],
    config: dict[str, Any],
    *,
    fraction_bits: int | None = None,
    gear: int | None = None,
) -> tuple[str, str, str, int, int]:
    bits, selected_gear = _quant_params(config, fraction_bits, gear)
    upstream_tile_keys = {
        tile_id: tiles[tile_id].content_key for tile_id in window.source_tile_ids
    }
    window_key = window_parameter_key(config)
    quant_key = quantization_key(
        {
            **config,
            "quantization": {
                **config["quantization"],
                "fraction_bits": bits,
                "gear": selected_gear,
            },
        }
    )
    feature_key = sha256_json(
        {
            "feature_algorithm": FEATURE_ALGORITHM_VERSION,
            "window_parameter_key": window_key,
            "quantization_key": quant_key,
            "upstream_tile_keys": upstream_tile_keys,
            "window": window.to_dict(),
        }
    )
    return feature_key, window_key, quant_key, bits, selected_gear


def compute_window_feature(
    window: WindowSpec,
    mosaic: Mosaic,
    tiles: dict[str, TileData],
    config: dict[str, Any],
    *,
    fraction_bits: int | None = None,
    gear: int | None = None,
) -> dict[str, Any]:
    bits, selected_gear = _quant_params(config, fraction_bits, gear)
    raw = raw_features(window, mosaic, tiles)
    feature_key, window_key, quant_key, bits, selected_gear = feature_key_for_window(
        window,
        tiles,
        config,
        fraction_bits=bits,
        gear=selected_gear,
    )
    upstream_tile_keys = {
        tile_id: tiles[tile_id].content_key for tile_id in window.source_tile_ids
    }
    dimensions = [
        quantize_dimension(
            raw[dimension],
            dimension,
            bits,
            selected_gear,
            config["quantization"]["dimension_scales_milli"],
        )
        for dimension in DIMENSION_NAMES
    ]
    return {
        "window_id": window.window_id,
        "owner_tile_id": window.owner_tile_id,
        "core_rect": list(window.core_rect),
        "context_rect": list(window.context_rect),
        "source_tile_ids": list(window.source_tile_ids),
        "upstream_tile_keys": upstream_tile_keys,
        "window_parameter_key": window_key,
        "quantization_key": quant_key,
        "fraction_bits": bits,
        "gear": selected_gear,
        "raw": {
            key: value
            for key, value in raw.items()
            if key != "spectral_ratio_inputs"
        },
        "spectral_ratio_inputs": raw["spectral_ratio_inputs"],
        "dimensions": dimensions,
        "observation": [dimension["q"] for dimension in dimensions],
        "feature_key": feature_key,
    }
