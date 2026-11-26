from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable

import numpy as np

from .config import Config, VocabularyTerm
from .errors import E_CONFIG_INVALID, fail
from .resources import ImageCondition, MotionTable, TextPrompt
from .util import decimal_to_fixed, fingerprint, quantize, round_div, stable_signed


def validate_condition_vector(
    vector: Any,
    expected_dim: int,
    entry: str,
    *,
    injection_id: str | None = None,
) -> list[int]:
    if not isinstance(vector, (list, tuple)):
        fail(
            E_CONFIG_INVALID,
            entry,
            "condition vector must be a sequence",
            injection_id=injection_id,
        )
    if len(vector) != expected_dim:
        fail(
            E_CONFIG_INVALID,
            entry,
            "condition vector dimension does not match configuration",
            detail={"expected": expected_dim, "actual": len(vector)},
            injection_id=injection_id,
        )
    result: list[int] = []
    for idx, item in enumerate(vector):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            fail(
                E_CONFIG_INVALID,
                f"{entry}[{idx}]",
                "condition vector entries must be fixed-point integers",
                injection_id=injection_id,
            )
        result.append(int(item))
    return result


def encode_text(prompt: TextPrompt, config: Config) -> dict[str, Any]:
    terms = config.vocabulary_terms
    term_to_index = {term: idx for idx, term in enumerate(terms)}
    counts = {term: 0 for term in terms}
    hit_positions: list[dict[str, Any]] = []
    for sequence_index, term in enumerate(prompt.sequence()):
        if term not in term_to_index:
            fail(
                E_CONFIG_INVALID,
                f"prompts.prompts.{prompt.name}",
                "prompt contains a term outside the controlled vocabulary",
                detail={"term": term},
            )
        counts[term] += 1
        hit_positions.append(
            {
                "sequence_index": sequence_index,
                "vocabulary_position": term_to_index[term],
                "term": term,
            }
        )

    vector = [0] * config.condition_dim
    q = config.scale
    for vocabulary_position, term in enumerate(terms):
        count = counts[term]
        if count == 0:
            continue
        for dim in range(config.condition_dim):
            coefficient = stable_signed("text-embedding-v1", term, dim, limit=q)
            vector[dim] += count * coefficient

    body = {
        "kind": "text",
        "prompt": prompt.name,
        "prompt_fingerprint": prompt.fingerprint,
        "vocabulary_fingerprint": config.vocabulary_fingerprint,
        "dimension": config.condition_dim,
        "frac_bits": config.frac_bits,
        "counts": [{"term": term, "count": counts[term]} for term in terms],
        "hit_positions": hit_positions,
        "vector": vector,
    }
    body["fingerprint"] = fingerprint(body, "encode-text-v1")
    return body


def _block_statistics(pixels: np.ndarray, block_size: int, q: int) -> list[dict[str, int]]:
    height, width = pixels.shape[:2]
    rows = height // block_size
    cols = width // block_size
    statistics: list[dict[str, int]] = []
    for block_row in range(rows):
        for block_col in range(cols):
            y0, x0 = block_row * block_size, block_col * block_size
            block = pixels[y0 : y0 + block_size, x0 : x0 + block_size].astype(np.int64)
            area = block_size * block_size
            means = []
            for channel in range(3):
                channel_sum = int(block[:, :, channel].sum())
                means.append(round_div(channel_sum * q, area))
            if block_size > 1:
                dx = int(np.abs(np.diff(block, axis=1)).sum())
                dy = int(np.abs(np.diff(block, axis=0)).sum())
                dx_count = block_size * (block_size - 1) * 3
                dy_count = block_size * (block_size - 1) * 3
                gx = round_div(dx * q, max(1, dx_count))
                gy = round_div(dy * q, max(1, dy_count))
            else:
                gx = 0
                gy = 0
            statistics.append(
                {
                    "block_row": block_row,
                    "block_col": block_col,
                    "mean_r": means[0],
                    "mean_g": means[1],
                    "mean_b": means[2],
                    "gradient_x": gx,
                    "gradient_y": gy,
                }
            )
    return statistics


def encode_image(image: ImageCondition, config: Config) -> dict[str, Any]:
    statistics = _block_statistics(image.pixels, config.image_block_size, config.scale)
    feature_names = ("mean_r", "mean_g", "mean_b", "gradient_x", "gradient_y")
    accumulators = [0] * config.condition_dim
    counts = [0] * config.condition_dim
    for block_index, stats in enumerate(statistics):
        for feature_index, feature_name in enumerate(feature_names):
            lane = (block_index * len(feature_names) + feature_index) % config.condition_dim
            accumulators[lane] += int(stats[feature_name])
            counts[lane] += 1
    vector = [
        round_div(accumulators[idx], counts[idx]) if counts[idx] else 0
        for idx in range(config.condition_dim)
    ]
    body = {
        "kind": "image",
        "image_fingerprint": image.content_fingerprint,
        "path": image.path,
        "dimension": config.condition_dim,
        "frac_bits": config.frac_bits,
        "block_size": config.image_block_size,
        "block_rows": config.image_block_rows,
        "block_cols": config.image_block_cols,
        "statistics": statistics,
        "vector": vector,
    }
    body["fingerprint"] = fingerprint(body, "encode-image-v1")
    return body


def encode_motion(motion: MotionTable, frame_index: int, config: Config) -> dict[str, Any]:
    if frame_index < 0 or frame_index >= config.frame_count:
        fail(
            E_CONFIG_INVALID,
            "motion.frame",
            "frame index is outside the motion table",
            detail={"frame": frame_index, "frame_count": config.frame_count},
        )
    frame = motion.values[frame_index]
    accumulators = [0] * config.condition_dim
    counts = [0] * config.condition_dim
    q = config.scale
    for block_row in range(config.motion_block_rows):
        for block_col in range(config.motion_block_cols):
            vx = int(frame[block_row, block_col, 0])
            vy = int(frame[block_row, block_col, 1])
            magnitude = abs(vx) + abs(vy)
            for feature_index, value in enumerate((vx, vy, magnitude)):
                lane = (
                    (block_row * config.motion_block_cols + block_col) * 3 + feature_index
                ) % config.condition_dim
                accumulators[lane] += value * q
                counts[lane] += 1
    vector = [
        round_div(accumulators[idx], counts[idx]) if counts[idx] else 0
        for idx in range(config.condition_dim)
    ]
    body = {
        "kind": "motion",
        "motion_fingerprint": motion.content_fingerprint,
        "path": motion.path,
        "frame": frame_index,
        "dimension": config.condition_dim,
        "frac_bits": config.frac_bits,
        "block_rows": config.motion_block_rows,
        "block_cols": config.motion_block_cols,
        "vector": vector,
    }
    body["fingerprint"] = fingerprint(body, "encode-motion-v1")
    return body


def fuse_conditions(
    frame_index: int,
    text_encoding: dict[str, Any],
    image_encoding: dict[str, Any] | None,
    motion_encoding: dict[str, Any],
    weights: dict[str, Decimal],
    config: Config,
    *,
    quantization_level: int | None = None,
) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {
        "text": text_encoding,
        "motion": motion_encoding,
    }
    if image_encoding is not None:
        inputs["image"] = image_encoding
    for condition_type in sorted(inputs):
        validate_condition_vector(
            inputs[condition_type]["vector"],
            config.condition_dim,
            f"condition.{condition_type}.vector",
        )

    q = config.scale
    step = config.quantization_level if quantization_level is None else int(quantization_level)
    fusion_vector = [0] * config.condition_dim
    ledger: list[dict[str, Any]] = []
    for condition_type in sorted(inputs):
        weight_fixed = decimal_to_fixed(
            weights[condition_type], config.frac_bits, f"fusion.weights.{condition_type}"
        )
        vector = [int(value) for value in inputs[condition_type]["vector"]]
        for dim in range(config.condition_dim):
            input_integer = vector[dim]
            scaled_exact = round_div(input_integer * weight_fixed, q)
            quantized, residual, direction = quantize(scaled_exact, step)
            fusion_vector[dim] += quantized
            ledger.append(
                {
                    "condition_type": condition_type,
                    "dimension": dim,
                    "input_integer": input_integer,
                    "weight_fixed": weight_fixed,
                    "scaled_integer": scaled_exact,
                    "quantization_step": step,
                    "quantized_integer": quantized,
                    "residual": residual,
                    "rounding_direction": direction,
                    "accumulator_after": fusion_vector[dim],
                }
            )

    inputs_fp = {
        "text": text_encoding["fingerprint"],
        "image": None if image_encoding is None else image_encoding["fingerprint"],
        "motion": motion_encoding["fingerprint"],
    }
    body = {
        "kind": "fusion",
        "frame": frame_index,
        "dimension": config.condition_dim,
        "quantization_step": step,
        "frac_bits": config.frac_bits,
        "weights": {
            name: str(weights[name]) for name in sorted(weights)
        },
        "inputs": inputs_fp,
        "fusion_vector": fusion_vector,
        "residual_total": sum(int(item["residual"]) for item in ledger),
        "residual_abs_max": max((abs(int(item["residual"])) for item in ledger), default=0),
        "ledger": ledger,
    }
    body["fingerprint"] = fingerprint(body, "fuse-conditions-v1")
    return body
