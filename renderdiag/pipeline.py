from __future__ import annotations

import base64
import io
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .config import Config
from .errors import E_BLOCK_ROW_OUT_OF_RANGE, E_NONFINITE_LATENT, fail
from .util import (
    deterministic_ints,
    finite_number,
    fingerprint,
    quantize,
    round_div,
    sha256_bytes,
)


def latent_shape(config: Config) -> tuple[int, int, int]:
    return (config.latent_height, config.latent_width, config.latent_channels)


def latent_element_count(config: Config) -> int:
    return config.latent_height * config.latent_width * config.latent_channels


def latent_summary(values: Sequence[int]) -> str:
    return fingerprint({"values": [int(value) for value in values]}, "latent-summary-v1")


def validate_latent_values(
    values: Any,
    expected_count: int,
    entry: str,
    *,
    injection_id: str | None = None,
) -> list[int]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        fail(
            E_NONFINITE_LATENT,
            entry,
            "latent must be a numeric sequence",
            injection_id=injection_id,
        )
    if len(values) != expected_count:
        fail(
            E_NONFINITE_LATENT,
            entry,
            "latent element count is invalid",
            detail={"expected": expected_count, "actual": len(values)},
            injection_id=injection_id,
        )
    result: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not finite_number(item):
            fail(
                E_NONFINITE_LATENT,
                f"{entry}[{index}]",
                "latent contains a non-finite value",
                detail={"value": repr(item)},
                injection_id=injection_id,
            )
        if isinstance(item, (int, np.integer)):
            result.append(int(item))
            continue
        numeric = float(item)
        if not numeric.is_integer():
            fail(
                E_NONFINITE_LATENT,
                f"{entry}[{index}]",
                "latent fixed-point values must be integers",
                detail={"value": repr(item)},
                injection_id=injection_id,
            )
        result.append(int(numeric))
    return result


def initialize_latent(
    fusion: dict[str, Any],
    frame_index: int,
    config: Config,
    *,
    step_fingerprint: str,
    init_config_fingerprint: str | None = None,
    seed: int | None = None,
    quantization_level: int | None = None,
) -> dict[str, Any]:
    count = latent_element_count(config)
    shape = latent_shape(config)
    q = config.scale
    config_fp = init_config_fingerprint or fingerprint(
        {
            "latent_shape": list(shape),
            "frac_bits": config.frac_bits,
        },
        "latent-init-config-v1",
    )
    active_seed = config.seed if seed is None else int(seed)
    step = config.quantization_level if quantization_level is None else int(quantization_level)
    key = {
        "domain": "latent-init-random-v1",
        "fusion_fingerprint": fusion["fingerprint"],
        "seed": active_seed,
        "frame": frame_index,
        "shape": list(shape),
        "config_fingerprint": config_fp,
    }
    raw = deterministic_ints(key, count, -q, q)
    quantized: list[int] = []
    directions = {"up": 0, "down": 0, "none": 0}
    residual_abs_max = 0
    for value in raw:
        q_value, residual, direction = quantize(value, step)
        quantized.append(q_value)
        directions[direction] += 1
        residual_abs_max = max(residual_abs_max, abs(residual))
    nonzero = [name for name in ("up", "down", "none") if directions[name]]
    overall_direction = nonzero[0] if len(nonzero) == 1 else "mixed"
    body = {
        "kind": "latent-init",
        "frame": frame_index,
        "shape": list(shape),
        "fusion_fingerprint": fusion["fingerprint"],
        "seed": active_seed,
        "quantization_step": step,
        "config_fingerprint": step_fingerprint,
        "init_config_fingerprint": config_fp,
        "values": quantized,
        "summary": latent_summary(quantized),
        "rounding_direction": overall_direction,
        "rounding_directions": directions,
        "residual_abs_max": residual_abs_max,
    }
    body["fingerprint"] = fingerprint(body, "latent-init-v1")
    return body


def noise_table_entry(
    frame_index: int,
    step_index: int,
    config: Config,
    *,
    seed: int | None = None,
    total_steps: int | None = None,
) -> list[int]:
    q = config.scale
    active_seed = config.seed if seed is None else int(seed)
    active_total_steps = config.denoise_steps if total_steps is None else int(total_steps)
    amplitude = max(1, round_div(q, active_total_steps + 1))
    key = {
        "domain": "denoise-noise-table-v1",
        "noise_seed": config.noise_seed,
        "seed": active_seed,
        "frame": frame_index,
        "step": step_index,
        "total_steps": active_total_steps,
        "shape": list(latent_shape(config)),
    }
    return deterministic_ints(key, latent_element_count(config), -amplitude, amplitude)


def denoise_step(
    previous_values: Sequence[int],
    fusion_vector: Sequence[int],
    frame_index: int,
    step_index: int,
    guidance_fixed: int,
    config: Config,
    *,
    step_config_fingerprint: str,
    seed: int | None = None,
    total_steps: int | None = None,
    quantization_level: int | None = None,
    noise_seed: int | None = None,
) -> dict[str, Any]:
    count = latent_element_count(config)
    previous = validate_latent_values(previous_values, count, "latent.previous")
    if not fusion_vector:
        raise ValueError("fusion vector must not be empty")
    active_seed = config.seed if seed is None else int(seed)
    active_total_steps = config.denoise_steps if total_steps is None else int(total_steps)
    active_noise_seed = config.noise_seed if noise_seed is None else int(noise_seed)
    active_quantization = (
        config.quantization_level if quantization_level is None else int(quantization_level)
    )
    noise = noise_table_entry(
        frame_index,
        step_index,
        config,
        seed=active_seed,
        total_steps=active_total_steps,
    )
    q = config.scale
    step = active_quantization
    result: list[int] = []
    directions = {"up": 0, "down": 0, "none": 0}
    residual_abs_max = 0
    condition_directions = {"up": 0, "down": 0, "none": 0}
    for index in range(count):
        condition_value = int(fusion_vector[index % len(fusion_vector)])
        condition_scaled_exact = round_div(condition_value * guidance_fixed, q)
        condition_scaled, condition_residual, condition_direction = quantize(
            condition_scaled_exact, step
        )
        condition_directions[condition_direction] += 1
        combined = int(previous[index]) + int(noise[index]) + condition_scaled
        quantized, residual, direction = quantize(combined, step)
        result.append(quantized)
        directions[direction] += 1
        residual_abs_max = max(residual_abs_max, abs(residual), abs(condition_residual))
    nonzero = [name for name in ("up", "down", "none") if directions[name]]
    overall_direction = nonzero[0] if len(nonzero) == 1 else "mixed"
    body = {
        "kind": "denoise-step",
        "frame": frame_index,
        "step": step_index,
        "total_steps": active_total_steps,
        "seed": active_seed,
        "noise_seed": active_noise_seed,
        "guidance_fixed": guidance_fixed,
        "quantization_step": step,
        "step_config_fingerprint": step_config_fingerprint,
        "summary": latent_summary(result),
        "rounding_direction": overall_direction,
        "rounding_directions": directions,
        "condition_rounding_directions": condition_directions,
        "residual_abs_max": residual_abs_max,
        "values": result,
    }
    body["fingerprint"] = fingerprint(body, "denoise-step-v1")
    return body


def projection_parameters(config: Config, *, seed: int | None = None) -> dict[str, Any]:
    q = config.scale
    active_seed = config.seed if seed is None else int(seed)
    projection_seed = int(
        fingerprint(
            {
                "domain": "projection-seed-v1",
                "seed": active_seed,
                "latent_shape": list(latent_shape(config)),
                "frame_shape": [config.height, config.width, config.channels],
            }
        )[:16],
        16,
    )
    count = config.height * config.width * config.channels * config.latent_channels
    coefficient_values = deterministic_ints(
        {"domain": "projection-coefficients-v1", "seed": projection_seed, "count": count},
        count,
        -6 * q,
        6 * q,
    )
    bias_values = deterministic_ints(
        {"domain": "projection-bias-v1", "seed": projection_seed, "count": config.height * config.width * config.channels},
        config.height * config.width * config.channels,
        24 * q,
        231 * q,
    )
    coefficients = np.asarray(coefficient_values, dtype=np.int64).reshape(
        config.height, config.width, config.channels, config.latent_channels
    )
    bias = np.asarray(bias_values, dtype=np.int64).reshape(
        config.height, config.width, config.channels
    )
    body = {
        "kind": "projection-parameters",
        "projection_seed": projection_seed,
        "frame_shape": [config.height, config.width, config.channels],
        "latent_shape": list(latent_shape(config)),
        "frac_bits": config.frac_bits,
        "coefficient_range": [-6 * q, 6 * q],
        "bias_range": [24 * q, 231 * q],
        "coefficient_sum": int(coefficients.sum()),
        "bias_sum": int(bias.sum()),
    }
    body["fingerprint"] = fingerprint(body, "projection-params-v1")
    return {"metadata": body, "coefficients": coefficients, "bias": bias}


def block_row_bounds(
    row_index: int,
    config: Config,
    block_rows: int | None = None,
) -> tuple[int, int]:
    effective_rows = config.block_rows if block_rows is None else block_rows
    if row_index < 0 or row_index >= effective_rows:
        fail(
            E_BLOCK_ROW_OUT_OF_RANGE,
            "decode.row",
            "block row index is outside the configured range",
            detail={"row": row_index, "block_rows": effective_rows},
        )
    return (
        row_index * config.height // effective_rows,
        (row_index + 1) * config.height // effective_rows,
    )


def latent_band_for_row(
    row_index: int,
    config: Config,
    block_rows: int | None = None,
) -> tuple[int, int]:
    effective_rows = config.block_rows if block_rows is None else block_rows
    base, remainder = divmod(config.latent_height, effective_rows)
    start = row_index * base + min(row_index, remainder)
    size = base + (1 if row_index < remainder else 0)
    return start, start + size


def decode_block_row(
    latent_values: Sequence[int],
    latent_fingerprint: str,
    row_index: int,
    projection: dict[str, Any],
    config: Config,
    *,
    block_rows: int | None = None,
    quantization_level: int | None = None,
    requested_row_override: int | None = None,
) -> dict[str, Any]:
    effective_rows = config.block_rows if block_rows is None else block_rows
    active_quantization = (
        config.quantization_level if quantization_level is None else int(quantization_level)
    )
    actual_row = row_index if requested_row_override is None else requested_row_override
    y0, y1 = block_row_bounds(actual_row, config, effective_rows)
    latent = np.asarray(latent_values, dtype=np.int64).reshape(latent_shape(config))
    coefficients = projection["coefficients"]
    bias = projection["bias"]
    q = config.scale
    row_height = y1 - y0
    latent_y0, latent_y1 = latent_band_for_row(actual_row, config, effective_rows)
    band_size = latent_y1 - latent_y0
    x_indices = np.arange(config.width, dtype=np.int64) * config.latent_width // config.width
    if band_size > 0:
        local_band_rows = (np.arange(row_height, dtype=np.int64) * band_size) // max(1, row_height)
        latent_y_indices = latent_y0 + np.minimum(local_band_rows, band_size - 1)
        selected_latent = latent[latent_y_indices[:, None], x_indices[None, :], :]
    else:
        selected_latent = np.zeros(
            (row_height, config.width, config.latent_channels), dtype=np.int64
        )
    products = (
        selected_latent[:, :, None, :]
        * coefficients[y0:y1, :, :, :]
    ).sum(axis=3, dtype=np.int64)
    projected = _round_div_array(products, q)
    fixed_values = bias[y0:y1, :, :] + projected
    pixel_values = np.clip(_round_div_array(fixed_values, q), 0, 255).astype(np.uint8)
    pixels = pixel_values
    pixel_bytes = pixels.tobytes()
    row_sha = sha256_bytes(pixel_bytes)
    png_buffer = io.BytesIO()
    Image.fromarray(pixels).save(
        png_buffer,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    png_bytes = png_buffer.getvalue()
    body = {
        "kind": "decode-row",
        "row": actual_row,
        "frame_height": config.height,
        "block_rows": effective_rows,
        "pixel_y_range": [y0, y1],
        "latent_region": {
            "rows": [latent_y0, latent_y1],
            "cols": [0, config.latent_width],
            "channels": [0, config.latent_channels],
        },
        "latent_fingerprint": latent_fingerprint,
        "projection_fingerprint": projection["metadata"]["fingerprint"],
        "quantization_step": active_quantization,
        "pixel_sha256": row_sha,
        "png_sha256": sha256_bytes(png_bytes),
        "pixels": pixels.reshape(-1).tolist(),
        "png_b64": base64.b64encode(png_bytes).decode("ascii"),
    }
    body["fingerprint"] = fingerprint(
        {
            "latent": latent_fingerprint,
            "projection": projection["metadata"]["fingerprint"],
            "quantization": active_quantization,
            "row": actual_row,
            "pixel_sha256": row_sha,
        },
        "decode-row-v1",
    )
    return body


def _round_div_array(numerator: np.ndarray, denominator: int) -> np.ndarray:
    """Deterministic half-away-from-zero integer division for int64 arrays."""
    values = np.asarray(numerator, dtype=np.int64)
    sign = np.where(values < 0, -1, 1).astype(np.int64)
    absolute = np.abs(values)
    quotient, remainder = np.divmod(absolute, int(denominator))
    quotient = quotient + (2 * remainder >= int(denominator))
    return sign * quotient


def compose_frame(
    frame_index: int,
    latent_fingerprint: str,
    row_products: Sequence[dict[str, Any]],
    projection: dict[str, Any],
    config: Config,
    *,
    block_rows: int | None = None,
    quantization_level: int | None = None,
    denoise_step_fingerprints: Sequence[str] = (),
) -> dict[str, Any]:
    effective_rows = config.block_rows if block_rows is None else block_rows
    active_quantization = (
        config.quantization_level if quantization_level is None else int(quantization_level)
    )
    if len(row_products) != effective_rows:
        fail(
            E_BLOCK_ROW_OUT_OF_RANGE,
            "decode.frame.rows",
            "frame composition requires every configured block row",
            detail={"expected": effective_rows, "actual": len(row_products)},
        )
    ordered = sorted(row_products, key=lambda item: int(item["row"]))
    if [int(item["row"]) for item in ordered] != list(range(effective_rows)):
        fail(
            E_BLOCK_ROW_OUT_OF_RANGE,
            "decode.frame.rows",
            "block row set is incomplete or duplicated",
        )
    frame_pixels = np.empty((config.height, config.width, config.channels), dtype=np.uint8)
    for item in ordered:
        y0, y1 = item["pixel_y_range"]
        pixels = np.asarray(item["pixels"], dtype=np.uint8).reshape(
            y1 - y0, config.width, config.channels
        )
        frame_pixels[y0:y1, :, :] = pixels
    pixel_bytes = frame_pixels.tobytes()
    pixels_sha = sha256_bytes(pixel_bytes)
    png_buffer = io.BytesIO()
    Image.fromarray(frame_pixels).save(
        png_buffer,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    png_bytes = png_buffer.getvalue()
    row_fingerprints = [item["fingerprint"] for item in ordered]
    body = {
        "kind": "decode-frame",
        "frame": frame_index,
        "latent_fingerprint": latent_fingerprint,
        "projection_fingerprint": projection["metadata"]["fingerprint"],
        "quantization_step": active_quantization,
        "denoise_step_fingerprints": list(denoise_step_fingerprints),
        "row_fingerprints": row_fingerprints,
        "pixel_sha256": pixels_sha,
        "png_sha256": sha256_bytes(png_bytes),
        "pixels": frame_pixels.reshape(-1).tolist(),
        "png_b64": base64.b64encode(png_bytes).decode("ascii"),
    }
    body["content_fingerprint"] = fingerprint(
        {
            "latent": latent_fingerprint,
            "projection": projection["metadata"]["fingerprint"],
            "quantization": active_quantization,
            "denoise_steps": list(denoise_step_fingerprints),
            "rows": row_fingerprints,
            "pixels": pixels_sha,
        },
        "frame-content-v1",
    )
    body["fingerprint"] = body["content_fingerprint"]
    return body
