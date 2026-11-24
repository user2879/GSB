"""Content-addressed keys and fingerprints for every product stage."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import WeightRow
from .util import sha256_obj


FINGERPRINT_SCHEMA = "opexlab-fingerprint-v1"


def activation_row_fingerprint(layer: str, row: int, values: list[float] | tuple[float, ...]) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "kind": "activation_row",
            "layer": layer,
            "row": row,
            "values": list(values),
        }
    )


def parameter_set_fingerprint(
    weight: WeightRow,
    conversion_sidecar_hash: Optional[str],
) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "kind": "parameter_set",
            "weight": weight.source_payload(),
            "conversion_sidecar_hash": conversion_sidecar_hash,
        }
    )


def softmax_cache_key(
    *,
    layer: str,
    row: int,
    activation_fingerprint: str,
    exchange_version: str,
    fractional_bits: int,
    quant_level: int,
    exp_approximation: str,
) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "stage": "softmax",
            "layer": layer,
            "row": row,
            "upstream_activation_fingerprint": activation_fingerprint,
            "exchange_version": exchange_version,
            "fractional_bits": fractional_bits,
            "quant_level": quant_level,
            "exp_approximation": exp_approximation,
        }
    )


def layernorm_stats_cache_key(
    *,
    layer: str,
    row: int,
    activation_fingerprint: str,
    exchange_version: str,
    fractional_bits: int,
    quant_level: int,
    variance_floor: float,
) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "stage": "layernorm_stats",
            "layer": layer,
            "row": row,
            "upstream_activation_fingerprint": activation_fingerprint,
            "exchange_version": exchange_version,
            "fractional_bits": fractional_bits,
            "quant_level": quant_level,
            "variance_floor": variance_floor,
        }
    )


def layernorm_output_cache_key(
    *,
    layer: str,
    row: int,
    stats_cache_key: str,
    stats_content_hash: str,
    parameter_fingerprint: str,
    exchange_version: str,
    fractional_bits: int,
    quant_level: int,
    scheme: str,
    bias_enabled: bool,
) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "stage": "layernorm_output",
            "layer": layer,
            "row": row,
            "upstream_stats_key": stats_cache_key,
            "upstream_stats_content_hash": stats_content_hash,
            "parameter_fingerprint": parameter_fingerprint,
            "exchange_version": exchange_version,
            "fractional_bits": fractional_bits,
            "quant_level": quant_level,
            "scheme": scheme,
            "bias_enabled": bias_enabled,
        }
    )


def stage_content_hash(
    *,
    result: Dict[str, Any],
    dependencies: Dict[str, Any],
) -> str:
    return sha256_obj(
        {
            "schema": FINGERPRINT_SCHEMA,
            "result": result,
            "dependencies": dependencies,
        }
    )

