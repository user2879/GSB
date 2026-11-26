"""Pure fixed-point condition, fusion, denoise, and decode kernels."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .canonical import HashPRNG, content_digest, round_div, rounding_direction, stable_u64
from .config import RenderConfig
from .errors import E_DECODE_OOB, E_DIMENSION, fail
from .exchange import StageRecord
from .fixed import QuantProfile, quantize_milli_vector, weighted_accumulate

RECIPE_VERSION = "mmfm-render-v1"


def verify_vector(vector: list[int], expected: int, *, label: str) -> None:
    if len(vector) != expected:
        fail(
            E_DIMENSION,
            f"{label} dimension differs from configuration",
            expected=expected,
            actual=len(vector),
        )


def fuse_conditions(
    condition_artifacts: dict[str, dict[str, Any]],
    *,
    config: RenderConfig,
) -> dict[str, Any]:
    if set(condition_artifacts) != {"text", "image", "motion"}:
        fail(
            E_DIMENSION,
            "fusion requires text, image, and motion condition vectors",
            present=sorted(condition_artifacts),
        )
    quantized: dict[str, list[int]] = {}
    for condition_type in sorted(condition_artifacts):
        vector = [int(value) for value in condition_artifacts[condition_type]["vector"]]
        verify_vector(vector, config.condition_dim, label=f"{condition_type} condition")
        quantized[condition_type] = vector
    fused = weighted_accumulate(
        quantized,
        config.fusion_weights_permille,
        ["image", "motion", "text"],
        config.quant_profile,
    )
    verify_vector(fused["vector"], config.condition_dim, label="fused condition")
    fused["source_digests"] = {
        condition_type: condition_artifacts[condition_type]["digest"]
        for condition_type in sorted(condition_artifacts)
    }
    fused["digest"] = content_digest(
        {
            "type": "fusion",
            "vector": fused["vector"],
            "profile": config.quant_profile.name,
            "weights": {key: config.fusion_weights_permille[key] for key in sorted(config.fusion_weights_permille)},
            "sources": fused["source_digests"],
        }
    )
    return fused


def initial_latent(
    *,
    frame_number: int,
    seed_offset: int,
    fusion_digest: str,
    config: RenderConfig,
) -> dict[str, Any]:
    profile = config.quant_profile
    key = "|".join(
        [
            "initial-latent",
            RECIPE_VERSION,
            fusion_digest,
            str(config.seed),
            str(frame_number),
            str(seed_offset),
            config.latent_config_fingerprint,
            config.exchange_version,
        ]
    )
    prng = HashPRNG(key)
    low = -2 * profile.scale
    high = 2 * profile.scale
    vector = prng.integers(config.latent_dim, low, high)
    return {
        "frame": frame_number,
        "vector": vector,
        "profile": profile.name,
        "scale": profile.scale,
        "source_fusion_digest": fusion_digest,
        "seed_material_digest": content_digest(
            {
                "key": key,
                "prng": "sha256-counter-v1",
                "range": [low, high],
            }
        ),
        "digest": content_digest(
            {
                "frame": frame_number,
                "vector": vector,
                "profile": profile.name,
                "fusion": fusion_digest,
            }
        ),
    }


def _noise_entry(
    *,
    config: RenderConfig,
    frame_number: int,
    seed_offset: int,
    step: int,
) -> tuple[list[int], str, dict[str, Any]]:
    profile = config.quant_profile
    key = "|".join(
        [
            "noise-table",
            RECIPE_VERSION,
            config.noise_scheme,
            str(config.latent_dim),
            str(config.seed),
            str(frame_number),
            str(seed_offset),
            str(step),
            profile.name,
            config.exchange_version,
        ]
    )
    prng = HashPRNG(key)
    amplitude = max(1, profile.scale // 16)
    values = prng.integers(config.latent_dim, -amplitude, amplitude)
    digest = content_digest(
        {
            "scheme": config.noise_scheme,
            "code": config.noise_schemes[config.noise_scheme],
            "values": values,
            "profile": profile.name,
            "step": step,
        }
    )
    metadata = {
        "scheme": config.noise_scheme,
        "code": config.noise_schemes[config.noise_scheme],
        "table_id": content_digest({"key": key, "version": "sha256-counter-v1"}),
        "amplitude": amplitude,
        "digest": digest,
    }
    return values, digest, metadata


def denoise_step(
    *,
    frame_number: int,
    seed_offset: int,
    step: int,
    previous: dict[str, Any],
    fusion: dict[str, Any],
    guidance_permille: int,
    config: RenderConfig,
    condition_refs: list[str],
) -> dict[str, Any]:
    profile = config.quant_profile
    previous_vector = [int(value) for value in previous["vector"]]
    fused_vector = [int(value) for value in fusion["vector"]]
    verify_vector(previous_vector, config.latent_dim, label="previous latent")
    verify_vector(fused_vector, config.condition_dim, label="fused condition")
    if step <= 0:
        fail("E_DENOISE_STEPS", "denoise step numbers are one-based positive integers", step=step)
    noise, noise_digest, noise_meta = _noise_entry(
        config=config,
        frame_number=frame_number,
        seed_offset=seed_offset,
        step=step,
    )
    next_vector: list[int] = []
    trace: list[dict[str, Any]] = []
    for dimension in range(config.latent_dim):
        condition_dimension = (dimension + step - 1) % config.condition_dim
        condition_value = fused_vector[condition_dimension]
        condition_numerator = condition_value * guidance_permille
        condition_scaled = round_div(condition_numerator, 1000)
        noise_value = noise[dimension]
        # Documented operation order: guidance scaling, noise table addition,
        # then one decay quantisation.  All operations are integer-only.
        combined_numerator = previous_vector[dimension] * 7 + condition_scaled + noise_value
        next_value = round_div(combined_numerator, 8)
        direction = rounding_direction(combined_numerator, 8)
        residual = combined_numerator - next_value * 8
        next_vector.append(next_value)
        trace.append(
            {
                "dimension": dimension,
                "condition_dimension": condition_dimension,
                "condition_numerator": condition_numerator,
                "condition_scaled": condition_scaled,
                "noise": noise_value,
                "previous": previous_vector[dimension],
                "combined_numerator": combined_numerator,
                "next": next_value,
                "round_direction": direction,
                "rounding_residual_num": residual,
            }
        )
    digest = content_digest(
        {
            "frame": frame_number,
            "step": step,
            "vector": next_vector,
            "profile": profile.name,
            "previous_digest": previous["digest"],
            "fusion_digest": fusion["digest"],
            "guidance_permille": guidance_permille,
            "noise_digest": noise_digest,
            "exchange_version": config.exchange_version,
        }
    )
    return {
        "frame": frame_number,
        "step": step,
        "previous_digest": previous["digest"],
        "fusion_digest": fusion["digest"],
        "guidance_permille": guidance_permille,
        "noise": noise_meta,
        "noise_entry": noise,
        "profile": profile.name,
        "fraction_bits": profile.fraction_bits,
        "scale": profile.scale,
        "vector": next_vector,
        "dimension_trace": trace,
        "round_directions": [item["round_direction"] for item in trace],
        "residual_abs_total": sum(abs(item["rounding_residual_num"]) for item in trace),
        "condition_refs": list(condition_refs),
        "digest": digest,
    }


@lru_cache(maxsize=16)
def _projection_plan_cached(
    seed: int,
    latent_dim: int,
    width: int,
    height: int,
    radius: int,
    projection_version: str,
    profile_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_channels = height * width * 3
    prng = HashPRNG(f"projection|{projection_version}|{seed}|{latent_dim}|{width}|{height}|{profile_name}")
    coefficients = np.empty((pixel_channels, latent_dim), dtype=np.int64)
    for output_index in range(pixel_channels):
        coefficients[output_index] = prng.integers(latent_dim, -radius, radius)
    biases = np.array(
        [128 + prng.integer(-8, 8) for _ in range(pixel_channels)], dtype=np.int64
    )
    coefficients.setflags(write=False)
    biases.setflags(write=False)
    return coefficients, biases


def projection_recipe(config: RenderConfig) -> dict[str, Any]:
    return {
        "version": config.projection_version,
        "seed": config.seed,
        "latent_dim": config.latent_dim,
        "width": config.image.width,
        "height": config.image.height,
        "coefficient_radius": config.projection_coefficient_radius,
        "denominator_scale": config.projection_denominator_scale,
        "profile": config.quant_profile.name,
        "rounding": "nearest_ties_away_from_zero",
        "bias_base": 128,
    }


def projection_plan(config: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    return _projection_plan_cached(
        config.seed,
        config.latent_dim,
        config.image.width,
        config.image.height,
        config.projection_coefficient_radius,
        config.projection_version,
        config.quant_profile.name,
    )


def chunk_partition(height: int, chunk_rows: int) -> list[dict[str, int]]:
    if chunk_rows <= 0 or chunk_rows > height:
        fail(
            "E_CHUNK_ROWS",
            "chunk_rows must be positive and no larger than image height",
            chunk_rows=chunk_rows,
            height=height,
        )
    return [
        {
            "index": index,
            "start_row": index * height // chunk_rows,
            "end_row": (index + 1) * height // chunk_rows,
        }
        for index in range(chunk_rows)
    ]


def decode_rows(
    latent: list[int],
    *,
    config: RenderConfig,
    start_row: int,
    end_row: int,
) -> np.ndarray:
    verify_vector(latent, config.latent_dim, label="decode latent")
    if not (0 <= start_row <= end_row <= config.image.height):
        fail(
            E_DECODE_OOB,
            "decode row request is outside the image",
            start_row=start_row,
            end_row=end_row,
            height=config.image.height,
        )
    if start_row == end_row:
        fail(E_DECODE_OOB, "decode row request is empty", start_row=start_row, end_row=end_row)
    coefficients, biases = projection_plan(config)
    width = config.image.width
    first = start_row * width * 3
    last = end_row * width * 3
    latent_array = np.asarray(latent, dtype=np.int64)
    numerators = coefficients[first:last] @ latent_array
    denominator = config.quant_profile.scale * config.projection_denominator_scale
    if denominator <= 0:
        fail("E_CONFIG_INVALID", "projection denominator must be positive", denominator=denominator)
    signs = np.where(numerators < 0, -1, 1).astype(np.int64)
    absolute = np.abs(numerators)
    quotient, remainder = np.divmod(absolute, denominator)
    quotient = quotient + (remainder * 2 >= denominator)
    rounded = signs * quotient
    pixels = np.clip(rounded + biases[first:last], 0, 255).astype(np.uint8)
    return pixels.reshape((end_row - start_row, width, 3))


def stage_records_for_step(artifact: dict[str, Any]) -> StageRecord:
    return StageRecord(
        stage="denoise",
        frame=int(artifact["frame"]),
        step=int(artifact["step"]),
        guidance_permille=int(artifact["guidance_permille"]),
        guidance=float(artifact["guidance_permille"]) / 1000.0,
        noise_scheme=str(artifact["noise"]["scheme"]),
        noise_code=int(artifact["noise"]["code"]),
        quant_profile=str(artifact["profile"]),
        quant_bits=int(artifact["fraction_bits"]),
        refs=list(artifact.get("condition_refs", [])),
        digest=str(artifact["digest"]),
        source_version="internal",
        precision_labels={"step_trace": "dimension_trace_internal_artifact"},
        metadata={"previous_digest": artifact["previous_digest"]},
    )
