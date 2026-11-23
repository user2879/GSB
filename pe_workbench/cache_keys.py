"""Content-addressed stage keys for exact invalidation."""

from __future__ import annotations

from typing import Any

from .config import WorkbenchConfig
from .encoding import scheme_fingerprint
from .hashing import sha256_json, short_hash
from .models import Sample
from .quantization import quantization_profile


def region_mask(sample: Sample, train_context_length: int) -> str:
    bits = 0
    for index, key_position in enumerate(sample.key_positions):
        distance = abs(sample.query_position - key_position)
        if distance >= train_context_length:
            bits |= 1 << index
    return format(bits, f"0{max(1, len(sample.key_positions))}x") if sample.key_positions else "0"


def stage_keys(sample: Sample, config: WorkbenchConfig) -> dict[str, Any]:
    pe_params = scheme_fingerprint(
        config.scheme,
        config.base_frequency,
        config.dimension,
        config.scale_factor,
        config.precision_level,
    )
    pe_payload = {
        "stage": "position_encoding",
        "formula_version": "pe-stage-v1",
        "sample_fingerprint": sample.content_fingerprint,
        "parameters": pe_params,
    }
    pe_key = sha256_json(pe_payload)

    quantization = quantization_profile(
        config.quantization_level, config.fixed_decimal_bits
    )
    score_payload = {
        "stage": "scoring",
        "formula_version": "score-stage-v1",
        "sample_fingerprint": sample.content_fingerprint,
        "position_encoding_key": pe_key,
        "dimension": config.dimension,
        "dimension_block_size": config.dimension_block_size,
        "quantization": {
            "level": config.quantization_level,
            "fixed_decimal_bits": config.fixed_decimal_bits,
            "effective_fractional_bits": quantization["effective_fractional_bits"],
            "scale": quantization["scale"],
            "clip": quantization["clip"],
        },
        "rounding": "nearest_even",
        "score_decimal_places": 12,
    }
    score_key = sha256_json(score_payload)

    mask = region_mask(sample, config.train_context_length)
    error_payload = {
        "stage": "error_tracking",
        "formula_version": "error-stage-v1",
        "sample_fingerprint": sample.content_fingerprint,
        "score_key": score_key,
        "region_mask": mask,
        "tolerances": config.tolerances.to_dict(),
        "reference_precision": "fp64",
        "comparison": "max_abs",
    }
    error_key = sha256_json(error_payload)
    load_key = sha256_json(
        {
            "stage": "loading",
            "sample_fingerprint": sample.content_fingerprint,
            "dimension": config.dimension,
        }
    )
    return {
        "load": load_key,
        "position_encoding": pe_key,
        "scoring": score_key,
        "error_tracking": error_key,
        "region_masks": mask,
        "pe_parameters": pe_params,
        "quantization": quantization,
        "sort_key": f"{sample.sample_id}|{config.scheme}",
        "short": {
            "load": short_hash(load_key),
            "position_encoding": short_hash(pe_key),
            "scoring": short_hash(score_key),
            "error_tracking": short_hash(error_key),
        },
    }


def load_artifact(sample: Sample, key: str) -> dict[str, Any]:
    return {
        "stage": "loading",
        "cache_key": key,
        "sample_id": sample.sample_id,
        "content_fingerprint": sample.content_fingerprint,
        "query_position": sample.query_position,
        "key_positions": list(sample.key_positions),
        "point_cost": sample.point_cost,
        "valid": True,
    }
