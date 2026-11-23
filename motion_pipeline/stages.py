"""Pure deterministic implementations of the four pipeline stages."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np

from .stable import (
    deterministic_noise_pixels,
    digest_int_array,
    div_round_half_away,
    float_to_fixed,
    round_half_away,
    sha256_json,
    trunc_div,
)


STAGE_ORDER = ("encode", "motion", "predict", "decode")


def _quantize_source_values(
    values: Iterable[float], bits: int
) -> tuple[list[int], dict[str, Any]]:
    fixed_values: list[int] = []
    residual_signed = 0.0
    residual_abs = 0.0
    directions = {"up": 0, "down": 0, "exact": 0}
    for value in values:
        fixed, residual, direction = float_to_fixed(float(value), bits)
        fixed_values.append(fixed)
        residual_signed += residual
        residual_abs += abs(residual)
        directions[direction] += 1
    count = max(1, len(fixed_values))
    metadata = {
        "quantized_value_count": len(fixed_values),
        "residual_signed_sum": round(residual_signed, 9),
        "residual_abs_sum": round(residual_abs, 9),
        "residual_mean_abs": round(residual_abs / count, 9),
        "rounding_direction_counts": directions,
    }
    return fixed_values, metadata


def _fill_rect(
    pixels: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: list[int],
) -> None:
    height, width = pixels.shape[:2]
    xa = max(0, min(width, x0))
    xb = max(0, min(width, x1))
    ya = max(0, min(height, y0))
    yb = max(0, min(height, y1))
    if xa < xb and ya < yb:
        pixels[ya:yb, xa:xb, :] = np.asarray(color, dtype=np.int16)


def encode_single_frame(
    frame: dict[str, Any], *, width: int, height: int, bits: int
) -> tuple[np.ndarray, dict[str, Any]]:
    pixels = np.zeros((height, width, 3), dtype=np.int16)
    source_values: list[float] = []
    for block in frame["color_blocks"]:
        source_values.extend(
            [block["x"], block["y"], block["width"], block["height"]]
        )
    for trajectory in frame["trajectories"]:
        source_values.extend([trajectory["x"], trajectory["y"]])
        if trajectory["shape"] == "circle":
            source_values.append(trajectory["radius"])
        else:
            source_values.extend([trajectory["width"], trajectory["height"]])

    fixed_values, quantization = _quantize_source_values(source_values, bits)
    cursor = 0
    scale = 1 << bits
    for block in frame["color_blocks"]:
        x_fixed, y_fixed, w_fixed, h_fixed = fixed_values[cursor : cursor + 4]
        cursor += 4
        x0 = x_fixed // scale
        y0 = y_fixed // scale
        x1 = (x_fixed + w_fixed) // scale
        y1 = (y_fixed + h_fixed) // scale
        _fill_rect(pixels, x0, y0, x1, y1, block["color"])

    for trajectory in frame["trajectories"]:
        x_fixed = fixed_values[cursor]
        y_fixed = fixed_values[cursor + 1]
        cursor += 2
        if trajectory["shape"] == "circle":
            radius_fixed = fixed_values[cursor]
            cursor += 1
            radius = max(0, radius_fixed)
            x0 = int((x_fixed - radius) // scale) - 1
            x1 = int((x_fixed + radius) // scale) + 2
            y0 = int((y_fixed - radius) // scale) - 1
            y1 = int((y_fixed + radius) // scale) + 2
            xa = max(0, x0)
            xb = min(width, x1)
            ya = max(0, y0)
            yb = min(height, y1)
            if xa < xb and ya < yb:
                yy, xx = np.indices((yb - ya, xb - xa), dtype=np.int64)
                pixel_x = (xx + xa) * scale + scale // 2
                pixel_y = (yy + ya) * scale + scale // 2
                mask = (pixel_x - x_fixed) ** 2 + (pixel_y - y_fixed) ** 2 <= radius**2
                region = pixels[ya:yb, xa:xb, :]
                region[mask] = np.asarray(trajectory["color"], dtype=np.int16)
        else:
            width_fixed = fixed_values[cursor]
            height_fixed = fixed_values[cursor + 1]
            cursor += 2
            x0 = x_fixed // scale
            y0 = y_fixed // scale
            x1 = (x_fixed + width_fixed) // scale
            y1 = (y_fixed + height_fixed) // scale
            _fill_rect(pixels, x0, y0, x1, y1, trajectory["color"])

    quantization["precision_bits"] = bits
    quantization["scale"] = scale
    quantization["noise_seed"] = frame["noise_seed"]
    return pixels, quantization


def encode_frames(
    frames: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    bits: int,
    sequence_id: str = "",
) -> dict[str, Any]:
    encoded_frames: list[list[list[list[int]]]] = []
    metadata: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        pixels, frame_metadata = encode_single_frame(
            frame, width=width, height=height, bits=bits
        )
        frame_metadata["task_sort_key"] = (
            f"{sequence_id}|encode|{frame_index:08d}"
        )
        encoded_frames.append(pixels.tolist())
        metadata.append(frame_metadata)
    return {"frame_pixels": encoded_frames, "frame_metadata": metadata}


def _luma(pixels: np.ndarray) -> np.ndarray:
    return (
        pixels[..., 0].astype(np.int64) * 77
        + pixels[..., 1].astype(np.int64) * 150
        + pixels[..., 2].astype(np.int64) * 29
    ) >> 8


def _vector_summary(vectors: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(vectors, dtype=np.int64).reshape(-1, 2)
    if flat.size == 0:
        return {
            "vector_count": 0,
            "mean_dx_fixed": 0,
            "mean_dy_fixed": 0,
            "mean_l1_fixed": 0,
            "max_abs_fixed": 0,
            "digest": hashlib.sha256(b"").hexdigest(),
        }
    count = int(flat.shape[0])
    abs_values = np.abs(flat)
    return {
        "vector_count": count,
        "mean_dx_fixed": trunc_div(int(flat[:, 0].sum()), count),
        "mean_dy_fixed": trunc_div(int(flat[:, 1].sum()), count),
        "mean_l1_fixed": trunc_div(int(abs_values.sum()), count),
        "max_abs_fixed": int(abs_values.max()),
        "digest": digest_int_array(flat.reshape(-1)),
    }


def estimate_motion_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    bits: int,
    block_size: int,
    search_radius: int,
    pair_index: int,
    sequence_id: str,
) -> dict[str, Any]:
    height, width = frame_a.shape[:2]
    gray_a = _luma(frame_a)
    gray_b = _luma(frame_b)
    scale = 1 << bits
    blocks_y = math.ceil(height / block_size)
    blocks_x = math.ceil(width / block_size)
    vectors = np.zeros((blocks_y, blocks_x, 2), dtype=np.int64)
    gradient_energy = np.zeros((blocks_y, blocks_x), dtype=np.int64)
    block_sad = np.zeros((blocks_y, blocks_x), dtype=np.int64)

    for by in range(blocks_y):
        y0 = by * block_size
        y1 = min(height, y0 + block_size)
        for bx in range(blocks_x):
            x0 = bx * block_size
            x1 = min(width, x0 + block_size)
            source = gray_a[y0:y1, x0:x1]
            best: tuple[int, int, int, int] | None = None
            # Stable candidate order: SAD, Manhattan radius, dy, dx.
            for dy in range(-search_radius, search_radius + 1):
                sy0 = y0 + dy
                sy1 = y1 + dy
                if sy0 < 0 or sy1 > height:
                    continue
                for dx in range(-search_radius, search_radius + 1):
                    sx0 = x0 + dx
                    sx1 = x1 + dx
                    if sx0 < 0 or sx1 > width:
                        continue
                    target = gray_b[sy0:sy1, sx0:sx1]
                    sad = int(np.abs(source - target).sum())
                    candidate = (sad, abs(dx) + abs(dy), dy, dx)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                best = (0, 0, 0, 0)
            sad_value, _, best_dy, best_dx = best
            block_sad[by, bx] = sad_value

            target = gray_b[y0 + best_dy : y1 + best_dy, x0 + best_dx : x1 + best_dx]
            temporal = target - source
            if source.shape[1] >= 3 and source.shape[0] >= 3:
                grad_x = source[:, 2:] - source[:, :-2]
                grad_y = source[2:, :] - source[:-2, :]
                temporal_x = temporal[:, 1:-1]
                temporal_y = temporal[1:-1, :]
                sum_gx2 = int((grad_x * grad_x).sum())
                sum_gy2 = int((grad_y * grad_y).sum())
                numer_x = -int((grad_x * temporal_x).sum())
                numer_y = -int((grad_y * temporal_y).sum())
                denom = max(1, sum_gx2 + sum_gy2)
                if sum_gx2 > 0:
                    delta_x = div_round_half_away(numer_x * scale, sum_gx2)
                    delta_x = max(-scale, min(scale, delta_x))
                else:
                    delta_x = 0
                if sum_gy2 > 0:
                    delta_y = div_round_half_away(numer_y * scale, sum_gy2)
                    delta_y = max(-scale, min(scale, delta_y))
                else:
                    delta_y = 0
                gradient_energy[by, bx] = denom
            else:
                delta_x = 0
                delta_y = 0
            vectors[by, bx, 0] = best_dx * scale + delta_x
            vectors[by, bx, 1] = best_dy * scale + delta_y

    summary = _vector_summary(vectors)
    summary.update(
        {
            "pair_index": pair_index,
            "mean_gradient_energy": trunc_div(int(gradient_energy.sum()), gradient_energy.size),
            "mean_sad": trunc_div(int(block_sad.sum()), block_sad.size),
            "task_sort_key": f"{sequence_id}|motion|{pair_index:08d}",
        }
    )
    return {
        "pair_index": pair_index,
        "blocks_y": blocks_y,
        "blocks_x": blocks_x,
        "block_size": block_size,
        "vectors": vectors.tolist(),
        "gradient_energy": gradient_energy.tolist(),
        "block_sad": block_sad.tolist(),
        "summary": summary,
    }


def _trunc_divide_array(numerator: np.ndarray, denominator: int) -> np.ndarray:
    quotient = np.abs(numerator) // denominator
    return np.where(numerator < 0, -quotient, quotient).astype(np.int64)


def _quantize_vectors(vectors: np.ndarray, quantization_level: int, bits: int) -> np.ndarray:
    step = max(1, int(quantization_level) * (1 << bits))
    magnitude = (np.abs(vectors) + step // 2) // step
    return np.where(vectors < 0, -magnitude, magnitude).astype(np.int64) * step


def _damping_fixed(damping: float, step: int) -> tuple[int, float]:
    damping_q = round_half_away(damping * 1000.0)
    accumulated = 1000
    for _ in range(step):
        accumulated = trunc_div(accumulated * damping_q, 1000)
    return accumulated, round(accumulated / 1000.0, 9)


def predict_from_motion(
    motion_payload: dict[str, Any],
    *,
    path: str,
    horizon: int,
    bits: int,
    linear_gain: float,
    damping: float,
    quantization_level: int,
) -> dict[str, Any]:
    if path not in ("linear", "weighted"):
        raise ValueError(f"unsupported prediction path: {path}")
    pairs = motion_payload["pairs"]
    if pairs:
        blocks_y = int(pairs[0]["blocks_y"])
        blocks_x = int(pairs[0]["blocks_x"])
        last = np.asarray(pairs[-1]["vectors"], dtype=np.int64)
        previous = np.asarray(pairs[-2]["vectors"], dtype=np.int64) if len(pairs) > 1 else np.zeros_like(last)
    else:
        blocks_y = max(1, int(motion_payload.get("blocks_y", 1)))
        blocks_x = max(1, int(motion_payload.get("blocks_x", 1)))
        last = np.zeros((blocks_y, blocks_x, 2), dtype=np.int64)
        previous = np.zeros_like(last)
    gain_q = round_half_away(max(0.0, linear_gain) * 1000.0)
    slope = last - previous
    steps: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        if path == "linear":
            raw = last + slope * step
            extrapolation_coefficient = float(step)
        else:
            count = min(4, len(pairs))
            if count == 0:
                raw = np.zeros_like(last)
            else:
                recent = [np.asarray(pairs[-index]["vectors"], dtype=np.int64) for index in range(1, count + 1)]
                weight_sum = count * (count + 1) // 2
                total = np.zeros_like(last)
                for index, vector_field in enumerate(recent, start=1):
                    total += vector_field * (count - index + 1)
                raw = _trunc_divide_array(total, weight_sum)
            extrapolation_coefficient = 0.0
        damp_q, damp_float = _damping_fixed(damping, step)
        adjusted = _trunc_divide_array(raw * gain_q, 1000)
        adjusted = _trunc_divide_array(adjusted * damp_q, 1000)
        vectors = _quantize_vectors(adjusted, quantization_level, bits)
        summary = _vector_summary(vectors)
        summary.update(
            {
                "step": step,
                "task_sort_key": f"{motion_payload.get('sequence_id', '')}|predict|{step:08d}",
            }
        )
        steps.append(
            {
                "step": step,
                "vectors": vectors.tolist(),
                "motion_summary": summary,
                "coefficients": {
                    "prediction_path": path,
                    "extrapolation_coefficient": extrapolation_coefficient,
                    "damping_coefficient": damp_float,
                    "damping_fixed_milli": damp_q,
                    "linear_gain": max(0.0, linear_gain),
                    "quantization_level": quantization_level,
                },
            }
        )
    return {
        "path": path,
        "blocks_y": blocks_y,
        "blocks_x": blocks_x,
        "horizon": horizon,
        "steps": steps,
    }


def decode_step(
    last_pixels: np.ndarray,
    predicted_vectors: np.ndarray,
    *,
    step: int,
    block_size: int,
    bits: int,
    sequence_id: str,
    noise_seed: int,
    noise_amplitude: float,
    noise_salt: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = last_pixels.shape[:2]
    scale = 1 << bits
    output = np.zeros_like(last_pixels, dtype=np.int16)
    blocks_y, blocks_x, _ = predicted_vectors.shape
    for by in range(blocks_y):
        y0 = by * block_size
        y1 = min(height, y0 + block_size)
        for bx in range(blocks_x):
            x0 = bx * block_size
            x1 = min(width, x0 + block_size)
            dx = div_round_half_away(int(predicted_vectors[by, bx, 0]), scale)
            dy = div_round_half_away(int(predicted_vectors[by, bx, 1]), scale)
            sy0 = y0 + dy
            sy1 = y1 + dy
            sx0 = x0 + dx
            sx1 = x1 + dx
            dy0 = max(0, -sy0)
            dy1 = (y1 - y0) - max(0, sy1 - height)
            dx0 = max(0, -sx0)
            dx1 = (x1 - x0) - max(0, sx1 - width)
            if dy0 < dy1 and dx0 < dx1:
                output[y0 + dy0 : y0 + dy1, x0 + dx0 : x0 + dx1, :] = last_pixels[
                    sy0 + dy0 : sy0 + dy1, sx0 + dx0 : sx0 + dx1, :
                ]
    noise = deterministic_noise_pixels(
        sequence_id=sequence_id,
        step=step,
        height=height,
        width=width,
        noise_seed=noise_seed + noise_salt,
        amplitude=noise_amplitude,
        bits=bits,
    )
    adjusted = output.astype(np.int32) + noise[..., None]
    clipped = np.clip(adjusted, 0, 255).astype(np.int16)
    metadata = {
        "step": step,
        "block_size": block_size,
        "fixed_point_bits": bits,
        "quantization_step_fixed": 1 << bits,
        "noise_amplitude": noise_amplitude,
        "noise_seed": noise_seed + noise_salt,
        "clip_range": [0, 255],
        "task_sort_key": f"{sequence_id}|decode|{step:08d}",
    }
    return clipped, metadata


def compute_prediction_error(
    predicted_steps: list[dict[str, Any]],
    truth_frames: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    bits: int,
) -> dict[str, Any] | None:
    count = min(len(predicted_steps), len(truth_frames))
    if count == 0:
        return None
    absolute_sum = 0
    squared_sum = 0
    maximum = 0
    samples = 0
    per_step: list[dict[str, Any]] = []
    for step_index in range(count):
        truth_pixels, _ = encode_single_frame(
            truth_frames[step_index], width=width, height=height, bits=bits
        )
        predicted_pixels = np.asarray(
            predicted_steps[step_index]["pixels"], dtype=np.int32
        )
        diff = predicted_pixels - truth_pixels.astype(np.int32)
        abs_diff = np.abs(diff)
        step_absolute = int(abs_diff.sum())
        step_squared = int((diff * diff).sum())
        step_maximum = int(abs_diff.max())
        step_samples = int(diff.size)
        absolute_sum += step_absolute
        squared_sum += step_squared
        maximum = max(maximum, step_maximum)
        samples += step_samples
        per_step.append(
            {
                "step": step_index + 1,
                "mae": round(step_absolute / max(1, step_samples), 9),
                "rmse": round(math.sqrt(step_squared / max(1, step_samples)), 9),
                "max_abs_error": step_maximum,
            }
        )
    return {
        "compared_steps": count,
        "sample_count": samples,
        "mae": round(absolute_sum / max(1, samples), 9),
        "rmse": round(math.sqrt(squared_sum / max(1, samples)), 9),
        "max_abs_error": maximum,
        "per_step": per_step,
    }


def params_fingerprint(stage: str, params: dict[str, Any], quantization_level: int | None) -> str:
    return sha256_json(
        {
            "stage": stage,
            "params": params,
            "quantization_level": quantization_level,
            "algorithm_version": 1,
        }
    )
