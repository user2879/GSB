"""Stable derivation of stage cache keys and result fingerprints."""

from __future__ import annotations

from typing import Any, Mapping

from .util import sha256_json


CACHE_KEY_VERSION = 1


def derive_cache_key(
    *,
    stage: str,
    layer: str,
    content_fingerprint: str,
    parameter_fingerprint: str,
    quantization_profile: str,
    calibration_hash: str,
    quant_gear: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    return sha256_json(
        {
            "cache_key_version": CACHE_KEY_VERSION,
            "stage": stage,
            "layer": layer,
            "layer_content_fingerprint": content_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "quantization_profile": quantization_profile,
            "quant_gear": quant_gear,
            "calibration_set_fingerprint": calibration_hash,
            "extra": dict(extra or {}),
        }
    )


def parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    return sha256_json(dict(parameters))

