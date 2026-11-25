"""Integer-only normalized similarity and confusion-candidate materialization."""

from __future__ import annotations

import math
from typing import Any

from .fingerprint import sha256_json


def fixed_norm_squared(integers: list[int]) -> int:
    total = 0
    for index in range(len(integers)):
        value = int(integers[index])
        total += value * value
    return total


def fixed_similarity_permille(integers_a: list[int], integers_b: list[int]) -> int:
    """Cosine similarity in permille using only integer arithmetic.

    Dimensions are consumed in ascending index order.  `math.isqrt` keeps the
    denominator deterministic and platform independent.
    """

    if len(integers_a) != len(integers_b):
        raise ValueError("quantized vectors have different dimensions")
    dot = 0
    for index in range(len(integers_a)):
        dot += int(integers_a[index]) * int(integers_b[index])
    if dot <= 0:
        return 0
    norm_a = fixed_norm_squared(integers_a)
    norm_b = fixed_norm_squared(integers_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    denominator = math.isqrt(norm_a) * math.isqrt(norm_b)
    if denominator == 0:
        return 0
    value = (dot * 2000 + denominator) // (2 * denominator)
    return max(0, min(1000, int(value)))


def similarity_base_key(template_a: dict[str, Any], template_b: dict[str, Any]) -> str:
    first, second = sorted(
        [
            {
                "key": template_a["template_key"],
                "integer_fingerprint": template_a["integer_fingerprint"],
            },
            {
                "key": template_b["template_key"],
                "integer_fingerprint": template_b["integer_fingerprint"],
            },
        ],
        key=lambda item: item["key"],
    )
    return "similarity-base:" + sha256_json({"a": first, "b": second})


def compute_similarity_base(
    template_a: dict[str, Any],
    template_b: dict[str, Any],
) -> dict[str, Any]:
    first, second = sorted([template_a, template_b], key=lambda item: item["template_key"])
    similarity = fixed_similarity_permille(first["integers"], second["integers"])
    pair_key = f'{first["template_key"]}|{second["template_key"]}'
    base_key = similarity_base_key(first, second)
    return {
        "pair_key": pair_key,
        "template_a": first["template_key"],
        "template_b": second["template_key"],
        "dimension": len(first["integers"]),
        "dimension_order": list(range(len(first["integers"]))),
        "comparison_points": len(first["integers"]),
        "similarity_permille": similarity,
        "references": [first["template_key"], second["template_key"]],
        "comparison_references": {
            "template_a_integer_fingerprint": first["integer_fingerprint"],
            "template_b_integer_fingerprint": second["integer_fingerprint"],
            "ordered_template_keys": [first["template_key"], second["template_key"]],
        },
        "cache_key": base_key,
    }


def materialize_similarity(base: dict[str, Any], threshold_permille: int) -> dict[str, Any]:
    result = dict(base)
    similarity = base["similarity_permille"]
    threshold_delta = similarity - threshold_permille
    candidate = similarity >= threshold_permille
    result.update(
        {
            "candidate_threshold_permille": threshold_permille,
            "threshold_delta_permille": threshold_delta,
            "candidate": candidate,
            "equality_boundary": similarity == threshold_permille,
            "boundary": candidate and similarity == threshold_permille,
            "tie_break_basis": "template_pair_key_lexicographic",
            "candidate_cache_key": "similarity-candidate:"
            + sha256_json({"base": base["cache_key"], "threshold_permille": threshold_permille}),
        }
    )
    return result


def compare_quantized_templates(
    template_a: dict[str, Any],
    template_b: dict[str, Any],
    threshold_permille: int,
) -> dict[str, Any]:
    return materialize_similarity(compute_similarity_base(template_a, template_b), threshold_permille)
