from __future__ import annotations

from typing import Any, Sequence

from .config import DiagnosticConfig
from .util import sha256_json


ALGORITHM_REVISION = "rope-diag-numeric-v1"


def scheme_parameter_fingerprint(scheme: str, config: DiagnosticConfig) -> str:
    payload: dict[str, Any] = {
        "algorithm_revision": ALGORITHM_REVISION,
        "scheme": scheme,
        "base": config.base,
        "dimension": config.dimension,
        "training_context_length": config.training_context_length,
        "extrapolation_target_length": config.extrapolation_target_length,
        "precision": config.precision,
        "quantization": config.quantization,
    }
    if scheme in ("linear_interpolation", "ntk_aware"):
        payload["scaling_factor"] = config.scaling_factor
    return sha256_json(payload)


def position_sample_key(
    scheme: str,
    sample_fingerprint: str,
    config: DiagnosticConfig,
    pruning_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "stage": "position",
            "scheme_parameters": scheme_parameter_fingerprint(scheme, config),
            "sample_content": sample_fingerprint,
            "precision": config.precision,
            "quantization": config.quantization,
            "pruning": pruning_fingerprint,
            "schema": "rope-diag-cache-key-v1",
        }
    )


def scoring_sample_key(
    position_key: str,
    sample_fingerprint: str,
    config: DiagnosticConfig,
    strategy_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "stage": "scoring",
            "position_key": position_key,
            "sample_content": sample_fingerprint,
            "precision": config.precision,
            "quantization": config.quantization,
            "strategy_fingerprint": strategy_fingerprint,
            "schema": "rope-diag-cache-key-v1",
        }
    )


def trace_key(
    scheme: str,
    dataset_fingerprint: str,
    config: DiagnosticConfig,
    pruning_fingerprint: str,
    strategy_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "stage": "trace",
            "scheme_parameters": scheme_parameter_fingerprint(scheme, config),
            "dataset": dataset_fingerprint,
            "precision": config.precision,
            "quantization": config.quantization,
            "pruning": pruning_fingerprint,
            "strategy": strategy_fingerprint,
            "tolerances": {
                "position_vector": config.tolerances.position_vector,
                "score": config.tolerances.score,
                "attention": config.tolerances.attention,
            },
            "reference_rule": "clip_each_position_to_training_context_length_minus_one",
            "schema": "rope-diag-cache-key-v1",
        }
    )


def sensitivity_key(
    scheme: str,
    dataset_fingerprint: str,
    config: DiagnosticConfig,
    pruning_fingerprint: str,
    strategy_fingerprint: str,
) -> str:
    return sha256_json(
        {
            "stage": "sensitivity",
            "scheme_parameters": scheme_parameter_fingerprint(scheme, config),
            "dataset": dataset_fingerprint,
            "precision": config.precision,
            "quantization": config.quantization,
            "pruning": pruning_fingerprint,
            "strategy": strategy_fingerprint,
            "neighborhood": {
                "base_step": config.sensitivity.base_step,
                "scale_step": config.sensitivity.scale_step,
                "radius": config.sensitivity.radius,
                "anchor_base": config.base,
                "anchor_scaling_factor": config.scaling_factor,
            },
            "schema": "rope-diag-cache-key-v1",
        }
    )


def aggregate_sample_keys_fingerprint(keys: Sequence[str]) -> str:
    return sha256_json({"sample_cache_keys": sorted(keys), "schema": "rope-diag-aggregate-v1"})


def stage_key_map(
    scheme: str,
    *,
    position_keys: Sequence[str],
    scoring_keys: Sequence[str],
    trace: str,
    sensitivity: str,
) -> dict[str, str]:
    return {
        f"{scheme}:position": aggregate_sample_keys_fingerprint(position_keys),
        f"{scheme}:scoring": aggregate_sample_keys_fingerprint(scoring_keys),
        f"{scheme}:trace": trace,
        f"{scheme}:sensitivity": sensitivity,
    }
