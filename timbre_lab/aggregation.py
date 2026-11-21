from __future__ import annotations

from typing import Any

from .config import AnalysisConfig, digest_json
def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (2 * numerator + denominator) // (2 * denominator)


def speaker_cache_key(
    speaker: str,
    segment_ids: list[str],
    descriptions: dict[str, dict[str, Any]],
    weights: dict[str, int],
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
) -> str:
    return digest_json(
        {
            "speaker": speaker,
            "segments": segment_ids,
            "description_keys": [descriptions[segment_id]["cache_key"] for segment_id in segment_ids],
            "weights": weights,
            "quantization": config.quantization_fingerprint(overrides),
        }
    )


def aggregate_speaker(
    group: dict[str, Any],
    features: dict[str, dict[str, Any]],
    descriptions: dict[str, dict[str, Any]],
    config: AnalysisConfig,
    overrides: dict[str, int] | None,
) -> dict[str, Any]:
    segment_ids = list(group["segment_ids"])
    weights = {segment_id: int(features[segment_id]["effective_frame_count"]) for segment_id in segment_ids}
    denominator = sum(weights.values())
    dimension_names = config.dimension_names()
    dimensions: list[dict[str, Any]] = []
    vector: list[float] = []
    integers: list[int] = []
    for dimension_index, dimension_name in enumerate(dimension_names):
        dimension_descriptions = [descriptions[segment_id]["dimensions"][dimension_index] for segment_id in segment_ids]
        bit_width = int(dimension_descriptions[0]["bit_width"])
        max_integer = (1 << bit_width) - 1
        weighted_sum = sum(int(item["integer"]) * weights[segment_id] for item, segment_id in zip(dimension_descriptions, segment_ids))
        integer = _round_ratio(weighted_sum, denominator)
        integer = min(max(integer, 0), max_integer)
        normalized = integer / max_integer
        weighted_input = sum(float(item["normalized"]) * weights[segment_id] for item, segment_id in zip(dimension_descriptions, segment_ids))
        input_mean = weighted_input / denominator if denominator else 0.0
        dimensions.append(
            {
                "name": dimension_name,
                "bit_width": bit_width,
                "weighted_sum_integer": weighted_sum,
                "weight_denominator": denominator,
                "integer": integer,
                "normalized": normalized,
                "rounding": "nearest_half_up_fixed_point",
                "residual": input_mean - normalized,
                "segment_weights": weights,
            }
        )
        vector.append(normalized)
        integers.append(integer)
    return {
        "speaker": group["speaker"],
        "cache_key": speaker_cache_key(group["speaker"], segment_ids, descriptions, weights, config, overrides),
        "segment_ids": segment_ids,
        "segments": [
            {
                "segment_id": segment_id,
                "weight": weights[segment_id],
                "description_key": descriptions[segment_id]["cache_key"],
            }
            for segment_id in segment_ids
        ],
        "total_weight": denominator,
        "degenerate_single_segment": len(segment_ids) == 1,
        "dimensions": dimensions,
        "integers": integers,
        "vector": vector,
        "upstream": {segment_id: descriptions[segment_id]["cache_key"] for segment_id in segment_ids},
    }
