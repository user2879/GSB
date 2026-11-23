"""Fixed-point feature embedding and emission scoring."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

from .constants import E_EMISSION_INVALID, FIXED_ALGORITHM, LABELS
from .models import Failure, InteractionConfig, ModelParams
from .stages import StageFailure
from .util import (
    digest,
    fraction_json,
    quantize_decimal,
    quantize_fraction,
    residual_direction,
    sha256_int,
)


def _all_model_numbers(model: ModelParams):
    for key, vector in model.char_embeddings.items():
        for value in vector:
            yield f"char_embeddings[{key!r}]", value
    for row_index, row in enumerate(model.emission_weights):
        for column_index, value in enumerate(row):
            yield f"emission_weights[{row_index}][{column_index}]", value
    for row_index, row in enumerate(model.transition_weights):
        for column_index, value in enumerate(row):
            yield f"transition_weights[{row_index}][{column_index}]", value
    for index, value in enumerate(model.start_weights):
        yield f"start_weights[{index}]", value


def validate_finite_parameters(model: ModelParams, sample_id: str) -> None:
    for name, value in _all_model_numbers(model):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        if not value.is_finite():
            raise StageFailure(
                Failure(
                    code=E_EMISSION_INVALID,
                    stage="emission",
                    message="non-finite model parameter encountered during emission",
                    sample_id=sample_id,
                    details={"parameter": name, "value": str(value)},
                )
            )


def _fallback_vector(key: str, feature_dim: int) -> list[Decimal]:
    values: list[Decimal] = []
    for index in range(feature_dim):
        raw = sha256_int(f"fallback|{key}|{index}", offset=0, size=4)
        values.append(Decimal((raw % 9) - 4))
    return values


def _embedding_for(
    model: ModelParams,
    lookup_key: str,
    bucket: int,
) -> tuple[str, list[Decimal]]:
    candidates = (
        lookup_key,
        f"@bucket:{bucket}",
        f"#bucket:{bucket}",
    )
    for candidate in candidates:
        if candidate in model.char_embeddings:
            return candidate, model.char_embeddings[candidate]
    fallback_key = f"@fallback:{lookup_key}:{bucket}"
    return fallback_key, _fallback_vector(fallback_key, model.feature_dim)


def _used_embedding_keys(payload: dict[str, Any], model: ModelParams) -> list[str]:
    used: set[str] = set()
    for position in payload["positions"]:
        for feature in position["features"]:
            key, _ = _embedding_for(model, feature["lookup_key"], feature["bucket"])
            used.add(key)
    return sorted(used)


def emission_parameter_fingerprint(
    payload: dict[str, Any],
    model: ModelParams,
    *,
    fixed_decimals: int,
    level: int,
) -> str:
    used_keys = _used_embedding_keys(payload, model)
    return digest(
        {
            "schema": "emission-params-v1",
            "embedding_keys": used_keys,
            "embeddings": {
                key: [
                    value
                    for value in model.char_embeddings.get(
                        key,
                        _fallback_vector(key, model.feature_dim),
                    )
                ]
                for key in used_keys
            },
            "emission_weights": model.emission_weights,
            "feature_dim": model.feature_dim,
            "labels": list(LABELS),
            "fixed_decimals": fixed_decimals,
            "quantization_level": level,
            "algorithm": FIXED_ALGORITHM,
        }
    )


def emission_key(
    feature_output: str,
    payload: dict[str, Any],
    model: ModelParams,
    *,
    fixed_decimals: int,
    level: int,
) -> str:
    param_fp = emission_parameter_fingerprint(
        payload,
        model,
        fixed_decimals=fixed_decimals,
        level=level,
    )
    return digest(
        {
            "stage": "emission",
            "upstream": feature_output,
            "parameters": param_fp,
            "fixed_decimals": fixed_decimals,
            "quantization_level": level,
        }
    )


def reserve_to_units(value: int, scale: int, level: int) -> tuple[int, Fraction]:
    return quantize_fraction(Fraction(value, scale), level)


def score_emissions(
    payload: dict[str, Any],
    model: ModelParams,
    config: InteractionConfig,
    *,
    fixed_decimals: int | None = None,
    quantization_level: int | None = None,
) -> dict[str, Any]:
    sample_id = str(payload["sample_id"])
    validate_finite_parameters(model, sample_id)
    decimals = config.fixed_decimals if fixed_decimals is None else fixed_decimals
    level = config.quantization_level if quantization_level is None else quantization_level
    scale = 10**decimals

    embedding_cache: dict[str, tuple[list[int], Fraction]] = {}
    emission_ints: list[list[int]] = []
    emission_weight_residual = Fraction(0)
    for row in model.emission_weights:
        quantized_row: list[int] = []
        for value in row:
            try:
                units, residual = quantize_decimal(value, decimals, level)
            except ValueError as exc:
                raise StageFailure(
                    Failure(
                        code=E_EMISSION_INVALID,
                        stage="emission",
                        message="cannot quantize non-finite emission weight",
                        sample_id=sample_id,
                        details={"error": str(exc)},
                    )
                ) from exc
            quantized_row.append(units)
            emission_weight_residual += residual
        emission_ints.append(quantized_row)

    positions: list[dict[str, Any]] = []
    scores: list[list[int]] = []
    for position in payload["positions"]:
        accumulator = [0] * model.feature_dim
        embedding_residual = Fraction(0)
        event_details: list[dict[str, Any]] = []
        for feature in position["features"]:
            lookup_key = feature["lookup_key"]
            bucket = int(feature["bucket"])
            embedding_key, vector = _embedding_for(model, lookup_key, bucket)
            if embedding_key not in embedding_cache:
                quantized_vector: list[int] = []
                residual_total = Fraction(0)
                for value in vector:
                    units, residual = quantize_decimal(value, decimals, level)
                    quantized_vector.append(units)
                    residual_total += residual
                embedding_cache[embedding_key] = (quantized_vector, residual_total)
            quantized_vector, residual_total = embedding_cache[embedding_key]
            embedding_residual += residual_total
            feature_index = int(feature["feature_index"]) % model.feature_dim
            for dimension, value in enumerate(quantized_vector):
                accumulator[(dimension + feature_index) % model.feature_dim] += value
            event_details.append(
                {
                    "channel": feature["channel"],
                    "token": feature["token"],
                    "feature_index": feature_index,
                    "bucket": bucket,
                    "embedding_key": embedding_key,
                }
            )

        label_scores: dict[str, int] = {}
        label_quantization: dict[str, Any] = {}
        aggregate_residual = Fraction(0)
        for label_index, label in enumerate(LABELS):
            total = sum(
                accumulator[dimension] * emission_ints[dimension][label_index]
                for dimension in range(model.feature_dim)
            )
            raw_fixed = Fraction(total, scale)
            units, residual = quantize_fraction(raw_fixed, level)
            label_scores[label] = units
            aggregate_residual += residual
            label_quantization[label] = {
                "residual_units": fraction_json(residual),
                "rounding": residual_direction(residual),
            }
        score_row = [label_scores[label] for label in LABELS]
        scores.append(score_row)
        positions.append(
            {
                "position": position["position"],
                "char": position["char"],
                "features": event_details,
                "emission_scores": label_scores,
                "quantization": {
                    "by_label": label_quantization,
                    "residual_units": fraction_json(aggregate_residual),
                    "rounding": residual_direction(aggregate_residual),
                    "embedding_residual_units": fraction_json(embedding_residual),
                    "embedding_rounding": residual_direction(embedding_residual),
                    "emission_weight_residual_units": fraction_json(
                        emission_weight_residual
                    ),
                },
            }
        )
    return {
        "sample_id": sample_id,
        "chars": list(payload["chars"]),
        "labels": list(LABELS),
        "scores": scores,
        "positions": positions,
        "fixed_decimals": decimals,
        "scale": scale,
        "quantization_level": level,
        "algorithm": FIXED_ALGORITHM,
    }


def emission_output_fingerprint(payload: dict[str, Any]) -> str:
    return digest(
        {
            "sample_id": payload["sample_id"],
            "chars": payload["chars"],
            "labels": payload["labels"],
            "scores": payload["scores"],
            "positions": payload["positions"],
            "fixed_decimals": payload["fixed_decimals"],
            "quantization_level": payload["quantization_level"],
            "algorithm": payload["algorithm"],
        }
    )
