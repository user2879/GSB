from __future__ import annotations

from typing import Any

import numpy as np

from .errors import Codes, Failure, StageFailure
from .fixedpoint import BASIS, q_scale, round_div
from .pngcodec import decode_png_rgb
from .validation import STOPWORDS, validate_motion_table, validate_noise_table


def injection_for(injections: list[dict[str, Any]] | None, frame: int,
                  stage: str) -> dict[str, Any] | None:
    for injection in injections or []:
        if int(injection.get("target_frame", -1)) == frame and injection.get("stage") == stage:
            return injection
    return None


def _inject_failure(injection: dict[str, Any] | None, code: str, message: str,
                    frame: int, stage: str, details: dict[str, Any] | None = None) -> StageFailure:
    return StageFailure(Failure(
        code, message, frame=frame, stage=stage,
        injection_id=(injection or {}).get("id"), details=details,
    ))


def encode_text(frame: int, state: dict[str, Any]) -> dict[str, Any]:
    terms = list(state["terms"])
    vocab = state.get("vocab", {})
    categories: dict[str, str] = {}
    for category, names in vocab.items():
        if isinstance(names, list):
            for name in names:
                categories[str(name)] = str(category)
    term_index = {term: i for i, term in enumerate(terms)}
    text = state["prompts"][frame]
    tokens = text.split()
    counts = [0] * len(terms)
    positions: list[dict[str, Any]] = []

    cursor = 0
    for token_index, raw_token in enumerate(tokens):
        start = text.find(raw_token, cursor)
        end = start + len(raw_token)
        cursor = end
        normalized = "".join(ch for ch in raw_token.lower() if ch.isalnum())
        if normalized in STOPWORDS:
            continue
        if normalized not in term_index:
            raise StageFailure(Failure(
                Codes.VOCAB_UNKNOWN_TERM,
                f"term {normalized!r} is outside the controlled vocabulary",
                frame=frame, stage="text_encode", details={"token": raw_token},
            ))
        vocab_index = term_index[normalized]
        counts[vocab_index] += 1
        positions.append({
            "term": normalized,
            "vocab_index": vocab_index,
            "category": categories.get(normalized, "unknown"),
            "token_index": token_index,
            "char_start": start,
            "char_end": end,
        })

    total = sum(counts)
    if total == 0:
        raise StageFailure(Failure(
            Codes.VOCAB_EMPTY_PROMPT, "prompt contains no controlled terms",
            frame=frame, stage="text_encode",
        ))

    frac_bits = int(state["config"]["fixed_point"]["frac_bits"])
    mode = state["config"]["fixed_point"].get("rounding_mode", "HALF_AWAY_FROM_ZERO")
    scale = q_scale(frac_bits)
    vector: list[int] = []
    trace: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        value, direction, residual = round_div(count * scale, total, mode)
        vector.append(value)
        trace.append({
            "dim": index, "count": count, "total": total, "bow_q": value,
            "round_direction": direction, "residual_num": residual,
        })
    positions.sort(key=lambda item: (item["vocab_index"], item["token_index"]))
    return {
        "vector": vector,
        "counts": counts,
        "total_terms": total,
        "hit_positions": positions,
        "normalization_trace": trace,
        "dim": len(vector),
    }


def decode_frame_image(frame: int, state: dict[str, Any],
                       injection: dict[str, Any] | None) -> tuple[np.ndarray | None, bool]:
    ref = state["image_refs"][frame]
    if ref is None:
        return None, True
    data = state["image_bytes"][ref]
    stage = "image_decode"
    if injection and injection.get("type") == "truncated_png":
        data = data[: max(40, len(data) // 2)]
    image_cfg = state["config"]["condition_image"]
    try:
        rgb = decode_png_rgb(data, image_cfg["width"], image_cfg["height"], f"frame:{frame}:{ref}")
    except StageFailure:
        raise
    except Exception as exc:
        failure = getattr(exc, "failure", None)
        if isinstance(failure, Failure):
            raise _inject_failure(injection, failure.code, failure.message, frame, stage,
                                  failure.details) from exc
        raise _inject_failure(injection, Codes.PNG_HEADER, str(exc), frame, stage) from exc
    return rgb, False


def encode_image(frame: int, state: dict[str, Any],
                 injections: list[dict[str, Any]] | None) -> dict[str, Any]:
    decode_injection = injection_for(injections, frame, "image_decode")
    injection = injection_for(injections, frame, "image_encode")
    rgb, absent = decode_frame_image(frame, state, decode_injection)
    dim = int(state["config"]["condition_dims"]["image"])
    if absent:
        return {"vector": [0] * dim, "absent": True, "blocks": [], "dim": dim}

    assert rgb is not None
    if injection and injection.get("type") == "nonfinite_condition":
        raw_stats = np.asarray(rgb, dtype=np.float64)
        raw_stats[0, 0, 0] = np.nan
        if not np.all(np.isfinite(raw_stats)):
            raise _inject_failure(
                injection, Codes.NONFINITE_CONDITION,
                "image condition contains a non-finite statistic after fault injection",
                frame, "image_encode",
            )

    image_cfg = state["config"]["condition_image"]
    block_rows = int(image_cfg["block_rows"])
    block_cols = int(image_cfg["block_cols"])
    block_h = rgb.shape[0] // block_rows
    block_w = rgb.shape[1] // block_cols
    frac_bits = int(state["config"]["fixed_point"]["frac_bits"])
    mode = state["config"]["fixed_point"].get("rounding_mode", "HALF_AWAY_FROM_ZERO")
    scale = q_scale(frac_bits)
    vector: list[int] = []
    blocks: list[dict[str, Any]] = []

    for br in range(block_rows):
        for bc in range(block_cols):
            block = rgb[br * block_h:(br + 1) * block_h, bc * block_w:(bc + 1) * block_w, :]
            pixel_count = block_h * block_w
            means: list[int] = []
            for channel in range(3):
                total = int(block[:, :, channel].sum(dtype=np.int64))
                value, _, _ = round_div(total * scale, pixel_count * 255, mode)
                means.append(value)

            diff_x = np.abs(block[:, 1:, :].astype(np.int16) - block[:, :-1, :].astype(np.int16))
            diff_y = np.abs(block[1:, :, :].astype(np.int16) - block[:-1, :, :].astype(np.int16))
            count_x = block_h * max(0, block_w - 1) * 3
            count_y = max(0, block_h - 1) * block_w * 3
            grad_x, _, _ = round_div(int(diff_x.sum(dtype=np.int64)) * scale,
                                     max(1, count_x) * 255, mode)
            grad_y, _, _ = round_div(int(diff_y.sum(dtype=np.int64)) * scale,
                                     max(1, count_y) * 255, mode)
            values = means + [grad_x, grad_y]
            vector.extend(values)
            blocks.append({"block_row": br, "block_col": bc, "values": values})

    if len(vector) != dim:
        raise StageFailure(Failure(
            Codes.COND_DIM_MISMATCH, "image condition vector dimension is inconsistent",
            frame=frame, stage="image_encode",
            details={"actual": len(vector), "expected": dim},
        ))
    return {"vector": vector, "absent": False, "blocks": blocks, "dim": dim}


def validate_and_encode_motion(frame: int, state: dict[str, Any],
                               injection: dict[str, Any] | None) -> dict[str, Any]:
    config = state["config"]
    vectors = state["motion"]
    if injection and injection.get("type") == "motion_block_mismatch":
        bad = np.asarray(vectors, dtype=np.int64)[:, :1, :, :].tolist()
        failures = validate_motion_table(
            {"vectors": bad, "frame_width": config["width"], "frame_height": config["height"]},
            config, f"injected:{injection.get('id')}",
        )
        if failures:
            failure = failures[0]
            failure.frame = frame
            failure.stage = "motion_validate"
            failure.injection_id = injection.get("id")
            raise StageFailure(failure)

    failures = validate_motion_table(
        {"vectors": vectors, "frame_width": config["width"],
         "frame_height": config["height"]}, config, "state:motion"
    )
    if failures:
        failure = failures[0]
        failure.frame = frame
        failure.stage = "motion_validate"
        raise StageFailure(failure)

    grid = config["motion_grid"]
    arr = np.asarray(vectors[frame], dtype=np.int64)
    frac_bits = int(config["fixed_point"]["frac_bits"])
    scale = q_scale(frac_bits)
    flat = arr.reshape(-1).tolist()
    vector = [int(v * scale // 8) for v in flat]
    rows = []
    row_dim = int(config["condition_dims"]["motion_row"])
    for row in range(int(grid["rows"])):
        start = row * row_dim
        rows.append(vector[start:start + row_dim])
    return {"vector": vector, "rows": rows, "dim": len(vector), "row_dim": row_dim}


def check_noise_for_frame(frame: int, state: dict[str, Any],
                          injection: dict[str, Any] | None) -> None:
    noise = dict(state["noise"])
    if injection and injection.get("type") == "noise_version_missing":
        noise.pop("version", None)
    failures = validate_noise_table(noise, state["config"], "state:noise")
    if failures:
        failure = failures[0]
        failure.frame = frame
        failure.stage = "denoise_step"
        if injection:
            failure.injection_id = injection.get("id")
        raise StageFailure(failure)


def fusion_weights(state: dict[str, Any]) -> tuple[dict[str, int], int]:
    raw = state["config"]["fusion"]["weights"]
    weights = {
        name: int(__import__("condrender.fixedpoint", fromlist=["basis_weight"]).basis_weight(
            raw[name], f"weights.{name}"
        ))
        for name in ("text", "image", "motion")
    }
    total = sum(weights.values())
    if total <= 0:
        raise StageFailure(Failure(Codes.COMMAND_INVALID, "fusion weight total must be positive",
                                   stage="fusion"))
    return weights, total


def fuse_frame(frame: int, state: dict[str, Any], text_data: dict[str, Any],
               image_data: dict[str, Any], motion_data: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    frac_bits = int(config["fixed_point"]["frac_bits"])
    mode = config["fixed_point"].get("rounding_mode", "HALF_AWAY_FROM_ZERO")
    weights, weight_total = fusion_weights(state)
    text_vec = text_data["vector"]
    image_vec = image_data["vector"]
    motion_rows = motion_data["rows"]
    fused_dim = int(config["condition_dims"]["fused"])
    text_dim = len(text_vec)
    image_dim = len(image_vec)

    rows: list[list[int]] = []
    trace: list[dict[str, Any]] = []
    for block_row in range(len(motion_rows)):
        fused_row: list[int] = []
        for dim in range(fused_dim):
            if dim < text_dim:
                modality, source, weight = "text", text_vec[dim], weights["text"]
            elif dim < text_dim + image_dim:
                offset = dim - text_dim
                modality, source, weight = "image", image_vec[offset], weights["image"]
            else:
                offset = dim - text_dim - image_dim
                modality, source, weight = "motion", motion_rows[block_row][offset], weights["motion"]
            numerator = weight * int(source)
            value, direction, residual = round_div(numerator, weight_total, mode)
            fused_row.append(value)
            trace.append({
                "frame": frame,
                "block_row": block_row,
                "dim": dim,
                "modality": modality,
                "source_q": int(source),
                "weight_bp": weight,
                "weight_total_bp": weight_total,
                "numerator": numerator,
                "denominator": weight_total,
                "fused_q": value,
                "quantum_q": 1,
                "quantum_real": f"1/{q_scale(frac_bits)}",
                "round_direction": direction,
                "residual_num": residual,
            })
        rows.append(fused_row)
    return {
        "rows": rows,
        "flat": [value for row in rows for value in row],
        "dim": fused_dim,
        "trace": trace,
        "weights_bp": weights,
        "weight_total_bp": weight_total,
        "frac_bits": frac_bits,
    }
