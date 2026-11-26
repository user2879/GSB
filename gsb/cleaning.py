from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .fixedpoint import quantize_ratio
from .samples import SampleSet, sample_content_fingerprint


CLEANING_ALGORITHM_VERSION = "clean-v1"


def cleaning_scope(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_bitmap_width": config["missing_bitmap_width"],
        "missing_rules": config["missing_rules"],
        "min_total_token_count": config["cleaning"]["min_total_token_count"],
        "fixed_point": config["fixed_point"],
        "algorithm": CLEANING_ALGORITHM_VERSION,
    }


def cleaning_config_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(cleaning_scope(config))


def normalize_length_distribution(
    raw_values: list[int],
    width: int,
    decimal_width: int,
    tier: str,
) -> list[int]:
    """Normalize bins to fixed-point parts summing exactly to one scale unit."""

    del width
    total = sum(raw_values)
    scale = 10**decimal_width
    if total <= 0:
        if not raw_values:
            return []
        base = scale // len(raw_values)
        result = [base] * len(raw_values)
        result[0] += scale - sum(result)
        return result

    floors: list[int] = []
    remainders: list[tuple[int, int, int]] = []
    for index, value in enumerate(raw_values):
        scaled = value * scale
        quotient, remainder = divmod(scaled, total)
        floors.append(quotient)
        remainders.append((remainder, -index, index))

    result = list(floors)
    missing = scale - sum(result)
    for _, _, index in sorted(remainders, reverse=True):
        if missing <= 0:
            break
        result[index] += 1
        missing -= 1
    return result


def clean_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    width = config["missing_bitmap_width"]
    fp_config = config["fixed_point"]
    bitmap = sample["missing_bitmap"]
    fill_vector = sample["batch_impute_length_distribution"]

    filled_values = list(sample["length_distribution"])
    filled_positions = [index for index, bit in enumerate(bitmap) if bit == 1]
    fill_sources: dict[str, str] = {}
    fill_values: dict[str, int] = {}
    for position in filled_positions:
        filled_values[position] = fill_vector[position]
        fill_sources[str(position)] = f"batch_header:{sample['batch_id']}#impute_values.length_distribution[{position}]"
        fill_values[str(position)] = fill_vector[position]

    rejection: dict[str, Any] | None = None
    for rule in config["missing_rules"]:
        position = rule["position"]
        if bitmap[position] == 1:
            rejection = {
                "status": "rejected",
                "reason_code": rule["code"],
                "config_field": rule["config_field"],
                "sample_field": f"missing_bitmap[{position}]",
                "hit_value": 1,
            }
            break

    min_tokens = config["cleaning"]["min_total_token_count"]
    if rejection is None and sample["term_count_total"] < min_tokens:
        rejection = {
            "status": "rejected",
            "reason_code": "LOW_TOKEN_COUNT",
            "config_field": "cleaning.min_total_token_count",
            "sample_field": "term_count_total",
            "hit_value": sample["term_count_total"],
            "threshold": min_tokens,
        }

    normalized_length = normalize_length_distribution(
        filled_values,
        width,
        fp_config["decimal_width"],
        fp_config["quantization_tier"],
    )

    record = {
        "sample_id": sample["sample_id"],
        "batch_id": sample["batch_id"],
        "domain": sample["domain"],
        "status": "retained" if rejection is None else "rejected",
        "reason_code": None if rejection is None else rejection["reason_code"],
        "config_field": None if rejection is None else rejection["config_field"],
        "sample_field": None if rejection is None else rejection["sample_field"],
        "hit_value": None if rejection is None else rejection.get("hit_value"),
        "threshold": None if rejection is None else rejection.get("threshold"),
        "filled_positions": filled_positions,
        "fill_values": fill_values,
        "fill_sources": fill_sources,
        "raw_length_distribution": list(sample["length_distribution"]),
        "filled_length_distribution": filled_values,
        "normalized_length_distribution_fp": normalized_length,
        "effective_token_count": sample["term_count_total"],
        "content_fingerprint": sample_content_fingerprint(sample),
    }
    record["cache_key"] = cleaning_cache_key(record["content_fingerprint"], config)
    record["record_fingerprint"] = fingerprint(_cleaning_hashable(record))
    return record


def cleaning_cache_key(sample_fingerprint: str, config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "stage": CLEANING_ALGORITHM_VERSION,
            "upstream_sample_fingerprint": sample_fingerprint,
            "cleaning_config_fingerprint": cleaning_config_fingerprint(config),
            "quantization_tier": config["fixed_point"]["quantization_tier"],
            "decimal_width": config["fixed_point"]["decimal_width"],
        }
    )


def _cleaning_hashable(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"cache_key", "record_fingerprint"}}


def run_cleaning(sample_set: SampleSet, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Clean in sample_id lexicographic order."""

    return [clean_sample(sample_set.samples[sample_id], config) for sample_id in sample_set.sample_order]


def cleaning_artifact_fingerprint(records: list[dict[str, Any]]) -> str:
    return fingerprint([record["record_fingerprint"] for record in records])
