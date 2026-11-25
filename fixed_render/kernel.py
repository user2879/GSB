from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RenderConfig
from .errors import CODES, StageFault
from .fixedpoint import (
    QuantSpec,
    assert_finite_numeric,
    quantize_rational,
    round_integer_ratio,
)
from .hashing import array_fingerprint, derive_seed
from .inputs import ConditionBundle


@dataclass
class NamedVector:
    condition_type: str
    dimension: int
    values: np.ndarray
    present: bool = True

    def sort_key(self) -> tuple[str, int]:
        return (self.condition_type, self.dimension)


def kernel_parameter_fingerprint(config: RenderConfig) -> dict[str, Any]:
    """Parameters that can change latent/pixel arithmetic, excluding cache topology."""
    return {
        "seed": config.seed,
        "frame_width": config.frame_width,
        "frame_height": config.frame_height,
        "num_frames": config.num_frames,
        "latent_scale": config.latent_scale,
        "latent_channels": config.latent_channels,
        "fusion_dim": config.fusion_dim,
        "guidance_q8": config.guidance_q8,
        "noise_amplitude": config.noise_amplitude,
        "fusion_weights": config.fusion_weights,
    }


def encode_text(
    bundle: ConditionBundle, config: RenderConfig, spec: QuantSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.zeros(len(bundle.vocab.tokens), dtype=np.int64)
    for token in bundle.prompt_tokens:
        counts[bundle.vocab.index[token]] += 1
    total = len(bundle.prompt_tokens)
    values, trace = quantize_rational(counts, total * 255 if False else total, spec)
    if values.shape[0] != config.text_dim:
        raise StageFault(
            CODES["DIM_MISMATCH"],
            "text condition vector dimension does not match config",
            field_name="text_dim",
            details={"actual": int(values.shape[0]), "configured": config.text_dim},
        )
    trace["stage"] = "encode_text"
    return values.astype(np.int64), trace


def encode_image(
    bundle: ConditionBundle, config: RenderConfig, spec: QuantSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    grid = config.image_grid
    block_h = config.image_height // grid
    block_w = config.image_width // grid
    if bundle.image is None:
        values = np.zeros(config.image_dim, dtype=np.int64)
        return values, {"stage": "encode_image", "absent": True, "zero_dimension": config.image_dim}

    image = bundle.image.astype(np.int64)
    features: list[np.ndarray] = []
    traces: list[dict[str, Any]] = []
    for br in range(grid):
        for bc in range(grid):
            y0, y1 = br * block_h, (br + 1) * block_h
            x0, x1 = bc * block_w, (bc + 1) * block_w
            block = image[y0:y1, x0:x1, :]
            area = block_h * block_w
            means, mean_trace = quantize_rational(block.sum(axis=(0, 1)), area * 255, spec)

            gx = np.abs(np.diff(block, axis=1)).sum(axis=2)
            gy = np.abs(np.diff(block, axis=0)).sum(axis=2)
            gx_sum = int(gx.sum())
            gy_sum = int(gy.sum())
            gx_values, gx_trace = quantize_rational(
                gx_sum, block_h * (block_w - 1) * 255 * 3, spec
            )
            gy_values, gy_trace = quantize_rational(
                gy_sum, (block_h - 1) * block_w * 255 * 3, spec
            )
            gx_values = np.atleast_1d(gx_values)
            gy_values = np.atleast_1d(gy_values)
            features.append(
                np.concatenate(
                    [means.astype(np.int64), gx_values.astype(np.int64), gy_values.astype(np.int64)]
                )
            )
            if br == 0 and bc == 0:
                traces.extend([mean_trace, gx_trace, gy_trace])
    values = np.concatenate(features).astype(np.int64)
    if values.shape[0] != config.image_dim:
        raise StageFault(
            CODES["DIM_MISMATCH"],
            "image condition vector dimension does not match config",
            field_name="image_dim",
            details={"actual": int(values.shape[0]), "configured": config.image_dim},
        )
    summary = {
        "stage": "encode_image",
        "absent": False,
        "grid": grid,
        "quantization_traces": traces,
        "integer_fingerprint": array_fingerprint(values),
    }
    return values, summary


def encode_motion(
    bundle: ConditionBundle, config: RenderConfig, spec: QuantSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    flat_q8 = bundle.motion.reshape(-1)
    if spec.frac_bits == 8:
        values = flat_q8.astype(np.int64) // 32
        # The table is Q8 pixels; normalize by a 32-pixel envelope and quantize.
        values, trace = quantize_rational(flat_q8, 32 * 256, spec)
    else:
        normalized, trace0 = quantize_rational(flat_q8, 32 * 256, spec)
        values = normalized.astype(np.int64)
        trace = trace0
    if values.shape[0] != config.motion_dim:
        raise StageFault(
            CODES["DIM_MISMATCH"],
            "motion condition vector dimension does not match config",
            field_name="motion_dim",
            details={"actual": int(values.shape[0]), "configured": config.motion_dim},
        )
    trace["stage"] = "encode_motion"
    return values.astype(np.int64), trace


def fuse_vectors(
    named_vectors: list[NamedVector],
    config: RenderConfig,
    spec: QuantSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    ordered = sorted(named_vectors, key=lambda item: item.sort_key())
    fused: np.ndarray | None = None
    total_weight = 0
    order_trace: list[dict[str, Any]] = []

    for item in ordered:
        weight = int(config.fusion_weights.get(item.condition_type, 0))
        projection = _deterministic_projection(
            item.values,
            config.fusion_dim,
            derive_seed("fusion-projection", config.seed, item.condition_type, item.dimension),
        )
        projected, projection_residual = round_integer_ratio(
            projection, max(1, item.dimension), spec
        )
        entry: dict[str, Any] = {
            "condition_type": item.condition_type,
            "dimension": item.dimension,
            "present": item.present,
            "weight": weight,
            "sort_key": list(item.sort_key()),
            "projected_fingerprint": array_fingerprint(projected),
            "projection_residual_fingerprint": array_fingerprint(projection_residual),
        }
        if weight == 0:
            entry["skipped"] = "zero_weight"
            order_trace.append(entry)
            continue

        if fused is None or total_weight == 0:
            fused = projected * weight
            total_weight = weight
            averaged, residual = round_integer_ratio(fused, total_weight, spec)
            fused = averaged
        else:
            numerator = fused * total_weight + projected * weight
            denominator = total_weight + weight
            averaged, residual = round_integer_ratio(numerator, denominator, spec)
            entry["quantization"] = {
                "quant_step": f"1/{spec.scale}",
                "mode": spec.mode,
                "denominator": denominator,
                "integer_fingerprint": array_fingerprint(averaged),
                "residual_fingerprint": array_fingerprint(residual),
                "residual_abs_max": int(np.max(np.abs(residual))),
            }
            fused = averaged
            total_weight = denominator
        entry["integer_fingerprint"] = array_fingerprint(fused)
        order_trace.append(entry)

    if fused is None:
        fused = np.zeros(config.fusion_dim, dtype=np.int64)
    if fused.shape[0] != config.fusion_dim:
        raise StageFault(CODES["DIM_MISMATCH"], "fusion dimension mismatch", field_name="fusion_dim")
    summary = {
        "stage": "fuse",
        "order": [(item.condition_type, item.dimension) for item in ordered],
        "total_weight": total_weight,
        "entries": order_trace,
        "integer_fingerprint": array_fingerprint(fused),
    }
    return fused.astype(np.int64), summary


def _deterministic_projection(
    values: np.ndarray, output_dim: int, seed: int
) -> np.ndarray:
    rng = np.random.Generator(np.random.SFC64(seed))
    matrix = rng.integers(-1, 2, size=(output_dim, values.shape[0]), dtype=np.int64)
    return matrix @ values.astype(np.int64)


def latent_shape(config: RenderConfig) -> tuple[int, int, int, int]:
    return (
        config.num_frames,
        config.frame_height // config.latent_scale,
        config.frame_width // config.latent_scale,
        config.latent_channels,
    )


def initialize_latent(
    fusion_fingerprint: str, config: RenderConfig, spec: QuantSpec
) -> tuple[np.ndarray, int]:
    seed = derive_seed(
        "latent-init",
        fusion_fingerprint,
        config.seed,
        kernel_parameter_fingerprint(config),
        spec.fingerprint(),
    )
    rng = np.random.Generator(np.random.SFC64(seed))
    shape = latent_shape(config)
    latent = rng.integers(-2 * spec.scale, 2 * spec.scale + 1, size=shape, dtype=np.int64)
    return latent, seed


def condition_map(
    fused: np.ndarray, config: RenderConfig, spec: QuantSpec
) -> np.ndarray:
    total = int(np.prod(latent_shape(config)))
    projected = _deterministic_projection(fused, total, derive_seed("condition-map", config.seed, config.fusion_dim))
    mapped, _ = round_integer_ratio(projected, max(1, config.fusion_dim), spec)
    return mapped.reshape(latent_shape(config)).astype(np.int64)


def noise_entry(step: int, config: RenderConfig) -> np.ndarray:
    seed = derive_seed(
        "noise-table",
        config.seed,
        kernel_parameter_fingerprint(config),
        step,
    )
    rng = np.random.Generator(np.random.SFC64(seed))
    return rng.integers(
        -config.noise_amplitude,
        config.noise_amplitude + 1,
        size=latent_shape(config),
        dtype=np.int64,
    )


def denoise_step(
    previous: np.ndarray,
    mapped_condition: np.ndarray,
    step_index: int,
    config: RenderConfig,
    spec: QuantSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    previous = assert_finite_numeric(previous, f"denoise:{step_index}:input")
    if previous.shape != latent_shape(config):
        raise StageFault(
            CODES["DIM_MISMATCH"],
            "previous latent shape mismatch",
            details={"actual": list(previous.shape), "expected": list(latent_shape(config))},
        )
    noise = noise_entry(step_index, config)
    with_noise = previous.astype(np.int64) + noise

    guidance_num = mapped_condition.astype(np.int64) * int(config.guidance_q8)
    guided, guidance_residual = round_integer_ratio(guidance_num, 256, spec)
    conditioned = with_noise + guided

    kept, keep_residual = round_integer_ratio(conditioned * 240, 256, spec)
    bound = 8 * spec.scale
    clipped = np.clip(kept, -bound, bound).astype(np.int64)
    trace = {
        "stage": f"denoise_step_{step_index}",
        "operation_order": [
            "validate_finite",
            "add_noise_table_entry",
            "add_q8_guidance_scaled_condition",
            "q8_decay_round",
            "saturating_clip",
        ],
        "noise_fingerprint": array_fingerprint(noise),
        "guidance": {
            "quant_step": f"1/{spec.scale}",
            "mode": spec.mode,
            "integer_fingerprint": array_fingerprint(guided),
            "residual_fingerprint": array_fingerprint(guidance_residual),
            "residual_abs_max": int(np.max(np.abs(guidance_residual))),
        },
        "decay": {
            "keep_q8": 240,
            "integer_fingerprint": array_fingerprint(kept),
            "residual_fingerprint": array_fingerprint(keep_residual),
            "residual_abs_max": int(np.max(np.abs(keep_residual))),
        },
        "clip_bound": bound,
        "clip_count": int(np.sum(kept != clipped)),
        "integer_fingerprint": array_fingerprint(clipped),
    }
    return clipped, trace


def tile_latent_range(tile_index: int, config: RenderConfig) -> tuple[int, int, int, int]:
    tiles = config.frame_height // config.tile_height
    if not 0 <= tile_index < tiles:
        raise StageFault(
            CODES["TILE_OUT_OF_RANGE"],
            f"tile row {tile_index} outside 0..{tiles - 1}",
            field_name="tile_index",
            details={"tile_index": tile_index, "tile_count": tiles},
        )
    pixel_y0 = tile_index * config.tile_height
    pixel_y1 = pixel_y0 + config.tile_height
    latent_y0 = pixel_y0 // config.latent_scale
    latent_y1 = pixel_y1 // config.latent_scale
    return pixel_y0, pixel_y1, latent_y0, latent_y1


def decode_tile(
    latent_rows: np.ndarray,
    frame_index: int,
    tile_index: int,
    decoder_fingerprint: str,
    config: RenderConfig,
    spec: QuantSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode one band from an already isolated latent row slice."""
    pixel_y0, pixel_y1, latent_y0, latent_y1 = tile_latent_range(tile_index, config)
    expected_latent_rows = config.tile_height // config.latent_scale
    if latent_rows.shape[0] != expected_latent_rows:
        raise StageFault(
            CODES["TILE_OUT_OF_RANGE"],
            "tile received latent rows outside its own row segment",
            details={
                "tile_index": tile_index,
                "actual_rows": int(latent_rows.shape[0]),
                "expected_rows": expected_latent_rows,
            },
        )
    latent_rows = assert_finite_numeric(latent_rows, f"tile:{frame_index}:{tile_index}")
    repeated = np.repeat(
        np.repeat(latent_rows, config.latent_scale, axis=0),
        config.latent_scale,
        axis=1,
    )
    # Horizontal repeat gives width; vertical repeat gives tile_height.
    rng = np.random.Generator(
        np.random.SFC64(
            derive_seed(
                "row-projection",
                decoder_fingerprint,
                frame_index,
                tile_index,
                pixel_y0,
                pixel_y1,
            )
        )
    )
    weights = rng.integers(
        -2, 3, size=(config.tile_height, config.frame_width, config.latent_channels, 3), dtype=np.int64
    )
    raw = np.einsum("yxc,yxck->yxk", repeated.astype(np.int64), weights, optimize=True)
    centered, residual = round_integer_ratio(raw * 6, spec.scale, spec)
    bias = rng.integers(
        -8, 9, size=(config.tile_height, config.frame_width, 3), dtype=np.int64
    )
    pixels = np.clip(128 + centered + bias, 0, 255).astype(np.uint8)
    trace = {
        "stage": f"tile_{frame_index}_{tile_index}",
        "pixel_rows": [pixel_y0, pixel_y1],
        "latent_rows": [latent_y0, latent_y1],
        "reads_only_own_latent_segment": True,
        "integer_fingerprint": array_fingerprint(pixels),
        "projection_residual_fingerprint": array_fingerprint(residual),
        "pixel_sum_q0": [int(x) for x in pixels.sum(axis=(0, 1))],
        "pixel_count": int(pixels.shape[0] * pixels.shape[1]),
    }
    return pixels, trace
