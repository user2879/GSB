"""Fixed-point speaker-level description aggregation."""

from __future__ import annotations

from typing import Any

from .util import cache_key, round_div_half_away, round_half_away_from_zero, sha256_obj, stable_float


def aggregation_cache_key(
    speaker_id: str,
    segment_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    weight_manifest = [
        {
            "file": row["file"],
            "segment_id": row["segment_id"],
            "description_key": row["description_key"],
            "weight": int(row["features"]["valid_frame_count"]),
        }
        for row in sorted(segment_rows, key=lambda item: (item["file"].casefold(), item["file"], item["start_sample"]))
    ]
    return cache_key(
        "speaker_aggregation",
        speaker_id,
        weight_manifest,
        sha256_obj(config),
    )


def aggregate_speaker(
    speaker_id: str,
    segment_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        segment_rows,
        key=lambda item: (item["file"].casefold(), item["file"], item["start_sample"], item["end_sample"]),
    )
    scale = int(config["fixed_scale"])
    dimensions = len(ordered[0]["description"]["vector"]) if ordered else 0
    numerators = [0 for _ in range(dimensions)]
    participants: list[dict[str, Any]] = []
    denominator = 0
    for row in ordered:
        weight = int(row["features"]["valid_frame_count"])
        denominator += weight
        participants.append(
            {
                "file": row["file"],
                "segment_id": row["segment_id"],
                "description_key": row["description_key"],
                "valid_frame_count": weight,
                "weight": weight,
            }
        )
        for index, value in enumerate(row["description"]["vector"]):
            fixed = round_half_away_from_zero(float(value) * scale)
            numerators[index] += fixed * weight
    if denominator > 0:
        averaged_fixed = [round_div_half_away(value, denominator) for value in numerators]
    else:
        averaged_fixed = [0 for _ in numerators]
    vector = [stable_float(value / scale, 12) for value in averaged_fixed]
    return {
        "speaker_id": speaker_id,
        "segment_count": len(ordered),
        "degenerate_single_segment": len(ordered) == 1,
        "aggregation_degenerate": len(ordered) == 1,
        "vector_dimension_count": dimensions,
        "fixed_scale": scale,
        "weight_denominator": denominator,
        "numerators_fixed": numerators,
        "vector_fixed": averaged_fixed,
        "vector": vector,
        "participants": participants,
        "recalculation_basis": {
            "policy": "valid_frame_weighted_fixed_point",
            "segment_description_keys": [row["description_key"] for row in ordered],
            "weights": [int(row["features"]["valid_frame_count"]) for row in ordered],
            "rounding": config["rounding"],
        },
    }

