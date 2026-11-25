from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
from PIL import Image

from .glyphs import TEMPLATE_SIZE, TEMPLATE_VERSION, normalize_mask, template_variants


def block_point_cost(block: dict[str, Any], variant_count: int = 3) -> int:
    x1, y1, x2, y2 = block["bbox"]
    pixel_count = max(1, int(x2 - x1) * int(y2 - y1))
    return pixel_count * max(1, int(variant_count))


def crop_to_template_mask(gray: np.ndarray, bbox: list[int], threshold: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray.shape[1], x2)
    y2 = min(gray.shape[0], y2)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), 255, dtype=np.uint8)
    mask = crop < int(threshold)
    return normalize_mask(mask, size=TEMPLATE_SIZE, ink_fit=14)


def _round_half_up(value: float) -> int:
    return int(Decimal(str(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def match_block(
    gray: np.ndarray,
    block: dict[str, Any],
    config: dict[str, Any],
    templates: dict[str, list[np.ndarray]] | None = None,
) -> dict[str, Any]:
    version = config.get("template_version", TEMPLATE_VERSION)
    templates = templates if templates is not None else template_variants(version)
    observed = crop_to_template_mask(gray, block["bbox"], config["binarize_threshold"])
    fraction_bits = int(config["fixed_point"]["fraction_bits"])
    weight_scale = 2 ** fraction_bits
    float_weights = [float(w) for w in config["fusion_weights"]]
    fixed_weights = [max(0, _round_half_up(w * weight_scale)) for w in float_weights]
    if sum(fixed_weights) <= 0:
        fixed_weights = [1] * len(fixed_weights)
    total_pixels = TEMPLATE_SIZE * TEMPLATE_SIZE

    scores: list[dict[str, Any]] = []
    best_char = None
    best_numerator = -1
    best_denominator = sum(fixed_weights[: len(next(iter(templates.values())))])
    for char in sorted(templates):
        variants = templates[char]
        numerator = 0
        variant_details = []
        for i, variant in enumerate(variants):
            weight = fixed_weights[i] if i < len(fixed_weights) else 0
            xor_count = int(np.count_nonzero(observed != variant))
            similarity_points = total_pixels - xor_count
            numerator += int(similarity_points) * int(weight)
            variant_details.append({
                "variant": i,
                "weight_fixed": weight,
                "xor_count": xor_count,
                "similarity_points": similarity_points,
            })
        denominator = sum(fixed_weights[: len(variants)]) or 1
        score01 = numerator / denominator / total_pixels
        entry = {
            "char": char,
            "numerator": int(numerator),
            "denominator": int(denominator),
            "score": round(score01, 9),
            "variants": variant_details,
        }
        scores.append(entry)
        if numerator > best_numerator:
            best_numerator = numerator
            best_char = char
            best_denominator = denominator

    confidence = best_numerator / best_denominator / total_pixels
    quant_level = int(config["fixed_point"]["quantization_level"])
    confidence_fixed = _round_half_up(confidence * quant_level)
    quantized_confidence = confidence_fixed / quant_level
    result = {
        **{k: v for k, v in block.items() if k not in {"regression"}},
        "recognized_char": best_char,
        "confidence": round(float(confidence), 9),
        "confidence_fixed": int(confidence_fixed),
        "quantization_level": quant_level,
        "quantization_step": 1.0 / quant_level,
        "quantization_residual": round(float(confidence - quantized_confidence), 12),
        "confidence_permille": int(max(0, min(1000, _round_half_up(confidence * 1000)))),
        "fraction_bits": fraction_bits,
        "weight_scale": weight_scale,
        "best_numerator": int(best_numerator),
        "best_denominator": int(best_denominator),
        "template_version": version,
        "template_size": TEMPLATE_SIZE,
        "point_cost": block_point_cost(block, len(next(iter(templates.values())))),
        "tie_rule": "equal_score_choose_lowest_unicode_codepoint",
        "top_candidates": sorted(scores, key=lambda s: (-s["numerator"], s["char"]))[:3],
    }
    result["regression"] = block.get("regression")
    return result
