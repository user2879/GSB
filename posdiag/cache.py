from __future__ import annotations

from typing import Any

from .encoding import scheme_parameters
from .models import AttentionSample, PositionConfig, sha256_value


CACHE_SCHEMA_VERSION = 1
REFERENCE_RULE = "clamp-to-training-boundary-L-1"


def scale_tier(sample: AttentionSample) -> str:
    bucket = 1
    while bucket < sample.point_units:
        bucket *= 2
    return f"S{bucket}:K{len(sample.key_positions)}"


def stage_cache_key(
    stage: str,
    config: PositionConfig,
    sample: AttentionSample,
    scheme: str,
) -> str:
    encoding_material: dict[str, Any] = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage": "encoding",
        "stage_schema": "rope-rotation-v1",
        "scheme_parameters": scheme_parameters(config, scheme),
        "sample_fingerprint": sample.fingerprint(),
        "precision": config.precision,
        "scale_tier": scale_tier(sample),
    }
    if stage == "encoding":
        return sha256_value(encoding_material)

    encoding_key = sha256_value(encoding_material)
    attention_material = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage": "attention",
        "stage_schema": "dot-softmax-v1",
        "encoding_key": encoding_key,
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "precision": config.precision,
        "scale_tier": scale_tier(sample),
    }
    if stage == "attention":
        return sha256_value(attention_material)

    attention_key = sha256_value(attention_material)
    trace_material = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage": "error_trace",
        "stage_schema": "staged-relative-error-v1",
        "attention_key": attention_key,
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "precision": config.precision,
        "scale_tier": scale_tier(sample),
        "tolerance": config.tolerance,
        "reference_rule": REFERENCE_RULE,
    }
    if stage == "error_trace":
        return sha256_value(trace_material)
    raise ValueError(f"unknown cache stage: {stage}")


def stage_references(sample: AttentionSample, scheme: str, stage: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "scheme": scheme,
        "stage": stage,
        "sample_fingerprint": sample.fingerprint(),
    }
