"""Fixed-point multi-template glyph matching and confidence synthesis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from .config import stage_param_fingerprint
from .templates import TEMPLATE_SIZE


def _resize_binary_block(
    gray: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    threshold: int,
    scale: int,
) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
        raise ValueError("cannot match an empty character block")
    binary = (crop < int(threshold)).astype(np.uint8) * 255
    resized = Image.fromarray(binary, mode="L").resize(
        (TEMPLATE_SIZE, TEMPLATE_SIZE), resample=Image.Resampling.LANCZOS
    )
    arr = np.asarray(resized, dtype=np.int64)
    return (arr * int(scale) + 127) // 255


def _variant_metric(img_q: np.ndarray, template: np.ndarray, metric: str, scale: int) -> tuple[int, int]:
    """Return (score_q, coverage_q)."""

    scaled = prepare_templates({"__one__": [template]}, int(scale))["__one__"][0]
    img_bool = img_q >= int(scale) // 2
    img_dilated = _dilate_bool(img_bool, 1)
    return _score_prepared(img_q, img_bool, img_dilated, scaled, metric, int(scale))


def _score_prepared(
    img_q: np.ndarray,
    img_bool: np.ndarray,
    img_dilated: np.ndarray,
    prepared: dict[str, Any],
    metric: str,
    scale: int,
) -> tuple[int, int]:
    tmpl_q = prepared["q"]
    tmpl_bool = prepared["bool"]
    tmpl_dilated = prepared["dilated"]
    if metric == "l1":
        error = int(np.abs(img_q - tmpl_q).sum())
        denom = int(img_q.size) * int(scale)
        score = int(scale) - (error * int(scale) + denom // 2) // denom
        score = max(0, min(int(scale), score))
    elif metric == "overlap":
        if not img_bool.any() and not tmpl_bool.any():
            score = int(scale)
        else:
            precision_num = int(np.count_nonzero(img_bool & tmpl_dilated))
            recall_num = int(np.count_nonzero(tmpl_bool & img_dilated))
            precision = precision_num / max(1, int(np.count_nonzero(img_bool)))
            recall = recall_num / max(1, int(np.count_nonzero(tmpl_bool)))
            f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
            score = int(round(f1 * int(scale)))
        score = max(0, min(int(scale), score))
    else:  # pragma: no cover - config validation guards this
        raise ValueError(f"unsupported metric: {metric}")
    if metric == "overlap":
        recall_num = int(np.count_nonzero(tmpl_bool & img_dilated))
        coverage = int(scale) if not tmpl_bool.any() else (recall_num * int(scale) + int(tmpl_bool.sum()) // 2) // int(tmpl_bool.sum())
    else:
        template_ink = int(tmpl_q.sum())
        if template_ink == 0:
            coverage = int(scale)
        else:
            coverage = (int(np.minimum(img_q, tmpl_q).sum()) * int(scale) + template_ink // 2) // template_ink
    coverage = max(0, min(int(scale), coverage))
    return score, coverage


def prepare_templates(
    templates: dict[str, list[np.ndarray]], scale: int
) -> dict[str, list[dict[str, Any]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    for char in sorted(templates):
        variants: list[dict[str, Any]] = []
        for variant in templates[char]:
            q = _tight_template(variant, scale)
            boolean = q >= scale // 2
            variants.append(
                {
                    "q": q,
                    "bool": boolean,
                    "dilated": _dilate_bool(boolean, 1),
                    "ink_count": int(np.count_nonzero(boolean)),
                }
            )
        prepared[char] = variants
    return prepared


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, radius)):
        expanded = out.copy()
        expanded[1:, :] |= out[:-1, :]
        expanded[:-1, :] |= out[1:, :]
        expanded[:, 1:] |= out[:, :-1]
        expanded[:, :-1] |= out[:, 1:]
        out = expanded
    return out


def _tight_template(template: np.ndarray, scale: int) -> np.ndarray:
    ys, xs = np.nonzero(template)
    if ys.size == 0:
        return np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.int64)
    cropped = template[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    image = Image.fromarray((cropped.astype(np.uint8) * 255), mode="L").resize(
        (TEMPLATE_SIZE, TEMPLATE_SIZE), resample=Image.Resampling.NEAREST
    )
    arr = np.asarray(image, dtype=np.int64)
    return (arr * int(scale) + 127) // 255


def match_character(
    gray: np.ndarray,
    bbox: list[int],
    config: dict[str, Any],
    templates: dict[str, list[np.ndarray]],
    image_name: str,
    column_index: int,
    char_index: int,
    piece_id: str,
    segment_id: str,
    parent_segment_id: str,
    column_range: list[int],
    template_version: str,
    cache_key: str,
    prepared_templates: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not templates:
        raise ValueError("no templates available")
    variant_count = len(next(iter(templates.values())))
    # Convert user weights to integer micro-units before accumulation.  The
    # arithmetic below is therefore host- and order-independent fixed point.
    weights = [max(1, int(round(float(value) * 1_000_000))) for value in config["fusion_weights"]]
    if len(weights) != variant_count:
        raise ValueError("fusion weight count does not match template variant count")
    bits = int(config["fixed_point_bits"])
    scale = 1 << bits
    img_q = _resize_binary_block(gray, bbox, config["binarization_threshold"], scale)
    img_bool = img_q >= scale // 2
    img_dilated = _dilate_bool(img_bool, 1)
    prepared = prepared_templates if prepared_templates is not None else prepare_templates(templates, scale)

    scores: dict[str, dict[str, Any]] = {}
    weight_sum = sum(weights)
    for char in sorted(templates):
        variants = prepared[char]
        weighted_score = 0
        weighted_coverage = 0
        variant_scores: list[int] = []
        for weight, variant in zip(weights, variants):
            score, coverage = _score_prepared(
                img_q, img_bool, img_dilated, variant, config["match_metric"], scale
            )
            variant_scores.append(score)
            weighted_score += score * weight
            weighted_coverage += coverage * weight
        fused_score = weighted_score // weight_sum
        fused_coverage = weighted_coverage // weight_sum
        scores[char] = {
            "fused_score_fixed": int(fused_score),
            "coverage_fixed": int(fused_coverage),
            "variant_scores_fixed": variant_scores,
        }

    # Character code ascending is the final tie breaker, never insertion order.
    ranked = sorted(scores.items(), key=lambda item: (-item[1]["fused_score_fixed"], ord(item[0])))
    recognized, best = ranked[0]
    second_score = ranked[1][1]["fused_score_fixed"] if len(ranked) > 1 else 0
    margin = max(0, int(best["fused_score_fixed"]) - int(second_score))
    coverage_q = int(best["coverage_fixed"])
    raw_confidence_q = (margin * (scale // 2) + (margin * coverage_q) // 2) // scale
    raw_confidence_q = max(0, min(scale, raw_confidence_q))

    steps = int(config["confidence_quantization_steps"])
    quantum = max(1, scale // steps)
    confidence_fixed = (raw_confidence_q // quantum) * quantum
    confidence_value = confidence_fixed / scale
    confidence_level = confidence_fixed // quantum
    confidence_digits = confidence_decimal_places(bits, steps)

    top_tied = [
        char for char, item in scores.items() if item["fused_score_fixed"] == best["fused_score_fixed"]
    ]
    return {
        "kind": "match",
        "key": cache_key,
        "image_name": image_name,
        "column_index": int(column_index),
        "char_index": int(char_index),
        "piece_id": piece_id,
        "segment_id": segment_id,
        "parent_segment_id": parent_segment_id,
        "column_range": [int(value) for value in column_range],
        "bbox": [int(value) for value in bbox],
        "recognized_char": recognized,
        "top_score_fixed": int(best["fused_score_fixed"]),
        "second_score_fixed": int(second_score),
        "score_margin_fixed": int(margin),
        "coverage_fixed": int(coverage_q),
        "confidence_raw_fixed": int(raw_confidence_q),
        "confidence_fixed": int(confidence_fixed),
        "confidence_value": confidence_value,
        "confidence_level": int(confidence_level),
        "confidence_levels": steps,
        "confidence_digits": confidence_digits,
        "confidence_quantization_steps": steps,
        "confidence_quantum": int(quantum),
        "fixed_point_bits": bits,
        "fixed_point_scale": int(scale),
        "match_metric": config["match_metric"],
        "fusion_weights": weights,
        "template_version": template_version,
        "parameters_fingerprint": stage_param_fingerprint("match", config),
        "template_scores": {
            char: {
                "fused_score_fixed": int(item["fused_score_fixed"]),
                "coverage_fixed": int(item["coverage_fixed"]),
                "variant_scores_fixed": list(item["variant_scores_fixed"]),
            }
            for char, item in sorted(scores.items())
        },
        "top_tied_chars": sorted(top_tied),
        "tie_break": "char_code_ascending",
    }


def compare_points(bbox: list[int] | tuple[int, int, int, int], variant_count: int) -> int:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    pixels = max(1, (x1 - x0) * (y1 - y0))
    return pixels * max(1, int(variant_count))


def confidence_decimal_places(bits: int, steps: int) -> int:
    """Useful for presentation; the numeric score remains fixed point."""

    return max(1, min(9, int(math.ceil(bits * math.log10(2.0)))))
