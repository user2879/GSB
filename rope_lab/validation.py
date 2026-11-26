"""Loading and schema validation for config, sample JSONL and commands."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .codes import (
    E_BLOCK_INCOMPATIBLE,
    E_DIM_MISMATCH,
    E_FIELD_INVALID,
    E_JSON_LINE,
    E_KEY_POSITION_DUPLICATE,
    E_POSITION_INVALID,
    E_SAMPLE_DUPLICATE,
    E_SCHEMA_UNKNOWN,
    E_VALUE_NONFINITE,
)


FRACTION_BITS_ENUM = (4, 8, 12, 16)
SIGNED_BITS_ENUM = (16, 32, 64)
ACCUMULATOR_BITS_ENUM = (16, 32, 40, 64)
QUANT_LEVEL_ENUM = ("truncate", "nearest", "ties_even")
FREQUENCY_CORRECTION_ENUM = (0, 1, 2, 3, 4)
STAGE_NAMES = ("position", "rotation", "dot", "normalization")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(float(value))


def make_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def load_json(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle), []
    except Exception as exc:  # noqa: BLE001 - reported as validation row
        return None, [make_error(E_JSON_LINE, f"JSON parse failed: {exc}", path=str(path))]


def load_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("each JSONL row must be an object")
                value["_line"] = line_number
                rows.append(value)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    make_error(
                        E_JSON_LINE,
                        f"JSONL parse failed on line {line_number}: {exc}",
                        path=str(path),
                        line=line_number,
                    )
                )
    return rows, errors


def _positive_int(errors: list[dict[str, Any]], obj: dict[str, Any], key: str, where: str) -> None:
    value = obj.get(key)
    if not is_int(value) or value <= 0:
        errors.append(make_error(E_FIELD_INVALID, f"{where}.{key} must be a positive integer", field=key, value=value))


def _nonnegative_int(errors: list[dict[str, Any]], value: Any, what: str, **details: Any) -> bool:
    if not is_int(value) or value < 0:
        errors.append(make_error(E_POSITION_INVALID, f"{what} must be a nonnegative integer", value=value, **details))
        return False
    return True


def validate_scheme(scheme: Any, dimension: int, index: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    where = f"schemes[{index}]"
    if not isinstance(scheme, dict):
        return [make_error(E_SCHEMA_UNKNOWN, f"{where} must be an object", value=scheme)]

    name = scheme.get("name")
    if not isinstance(name, str) or not name:
        errors.append(make_error(E_SCHEMA_UNKNOWN, f"{where}.name must be a nonempty string", value=name))

    for key in ("base_frequency", "angle_scale"):
        value = scheme.get(key)
        if not finite_number(value) or float(value) <= 0:
            errors.append(make_error(E_FIELD_INVALID, f"{where}.{key} must be a positive finite number", field=key, value=value))

    block_size = scheme.get("block_size")
    if not is_int(block_size) or block_size <= 0:
        errors.append(make_error(E_FIELD_INVALID, f"{where}.block_size must be a positive integer", value=block_size))
    elif is_int(dimension) and dimension > 0 and dimension % block_size != 0:
        errors.append(
            make_error(
                E_BLOCK_INCOMPATIBLE,
                f"block_size {block_size} does not divide dimension {dimension}",
                scheme=name,
                block_size=block_size,
                dimension=dimension,
            )
        )

    correction = scheme.get("frequency_correction")
    if correction not in FREQUENCY_CORRECTION_ENUM:
        errors.append(
            make_error(
                E_SCHEMA_UNKNOWN,
                f"frequency_correction must be one of {FREQUENCY_CORRECTION_ENUM}",
                scheme=name,
                value=correction,
            )
        )
    return errors


def validate_config(config: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(config, dict):
        return [make_error(E_SCHEMA_UNKNOWN, "config must be a JSON object", value=type(config).__name__)]

    dimension = config.get("dimension")
    if not is_int(dimension) or dimension <= 0:
        errors.append(make_error(E_FIELD_INVALID, "dimension must be a positive integer", value=dimension))
        dimension = None

    seed = config.get("seed")
    if not is_int(seed):
        errors.append(make_error(E_FIELD_INVALID, "seed must be an integer", value=seed))

    for key in ("train_length", "target_length"):
        value = config.get(key)
        if not is_int(value) or value < 0:
            errors.append(make_error(E_FIELD_INVALID, f"{key} must be a nonnegative integer", value=value))

    if is_int(config.get("target_length")) and is_int(config.get("train_length")):
        if config["target_length"] < config["train_length"]:
            errors.append(make_error(E_FIELD_INVALID, "target_length must be >= train_length"))

    for key in ("point_budget", "sample_quota", "concurrency", "sample_count", "key_count"):
        if key in config:
            _positive_int(errors, config, key, "config")

    bins = config.get("extrapolation_bins", [])
    if not isinstance(bins, list) or not bins or not all(is_int(x) and x > 0 for x in bins):
        errors.append(make_error(E_FIELD_INVALID, "extrapolation_bins must be a nonempty positive integer list", value=bins))
    elif bins != sorted(bins):
        errors.append(make_error(E_FIELD_INVALID, "extrapolation_bins must be ascending", value=bins))

    schemes = config.get("schemes")
    if not isinstance(schemes, list) or not schemes:
        errors.append(make_error(E_SCHEMA_UNKNOWN, "schemes must be a nonempty list", value=schemes))
    else:
        names: set[str] = set()
        for index, scheme in enumerate(schemes):
            errors.extend(validate_scheme(scheme, dimension or -1, index))
            if isinstance(scheme, dict) and isinstance(scheme.get("name"), str):
                if scheme["name"] in names:
                    errors.append(make_error(E_SCHEMA_UNKNOWN, "duplicate scheme name", scheme=scheme["name"]))
                names.add(scheme["name"])

    fixed = config.get("fixed_point")
    if not isinstance(fixed, dict):
        errors.append(make_error(E_SCHEMA_UNKNOWN, "fixed_point must be an object", value=fixed))
    else:
        if fixed.get("fraction_bits") not in FRACTION_BITS_ENUM:
            errors.append(
                make_error(
                    E_SCHEMA_UNKNOWN,
                    "fixed_point.fraction_bits is outside allowed enum",
                    allowed=FRACTION_BITS_ENUM,
                    value=fixed.get("fraction_bits"),
                )
            )
        if fixed.get("signed_bits") not in SIGNED_BITS_ENUM:
            errors.append(
                make_error(
                    E_SCHEMA_UNKNOWN,
                    "fixed_point.signed_bits is outside allowed enum",
                    allowed=SIGNED_BITS_ENUM,
                    value=fixed.get("signed_bits"),
                )
            )
        if fixed.get("accumulator_bits") not in ACCUMULATOR_BITS_ENUM:
            errors.append(
                make_error(
                    E_SCHEMA_UNKNOWN,
                    "fixed_point.accumulator_bits is outside allowed enum",
                    allowed=ACCUMULATOR_BITS_ENUM,
                    value=fixed.get("accumulator_bits"),
                )
            )
        if fixed.get("quant_level") not in QUANT_LEVEL_ENUM:
            errors.append(
                make_error(
                    E_SCHEMA_UNKNOWN,
                    "fixed_point.quant_level is outside allowed enum",
                    allowed=QUANT_LEVEL_ENUM,
                    value=fixed.get("quant_level"),
                )
            )
        fb = fixed.get("fraction_bits")
        sb = fixed.get("signed_bits")
        if is_int(fb) and is_int(sb) and fb >= sb:
            errors.append(make_error(E_FIELD_INVALID, "fraction_bits must be smaller than signed_bits"))

    tolerances = config.get("tolerances")
    if not isinstance(tolerances, dict):
        errors.append(make_error(E_FIELD_INVALID, "tolerances must be an object", value=tolerances))
    else:
        for stage in STAGE_NAMES:
            value = tolerances.get(stage)
            if finite_number(value):
                if float(value) <= 0:
                    errors.append(make_error(E_FIELD_INVALID, f"tolerances.{stage} must be positive finite", value=value))
            elif isinstance(value, dict):
                for metric in ("abs", "rel"):
                    metric_value = value.get(metric)
                    if not finite_number(metric_value) or float(metric_value) <= 0:
                        errors.append(
                            make_error(
                                E_FIELD_INVALID,
                                f"tolerances.{stage}.{metric} must be positive finite",
                                value=metric_value,
                            )
                        )
            else:
                errors.append(
                    make_error(
                        E_FIELD_INVALID,
                        f"tolerances.{stage} must be a positive number or {{abs, rel}}",
                        value=value,
                    )
                )
    return errors


def validate_vector(
    errors: list[dict[str, Any]],
    vector: Any,
    dimension: int,
    what: str,
    **details: Any,
) -> None:
    if not isinstance(vector, list):
        errors.append(make_error(E_DIM_MISMATCH, f"{what} must be a list of length {dimension}", value=vector, **details))
        return
    if len(vector) != dimension:
        errors.append(
            make_error(
                E_DIM_MISMATCH,
                f"{what} length {len(vector)} != dimension {dimension}",
                expected=dimension,
                actual=len(vector),
                **details,
            )
        )
    for index, value in enumerate(vector):
        if not finite_number(value):
            errors.append(
                make_error(
                    E_VALUE_NONFINITE,
                    f"{what}[{index}] must be a finite number",
                    dimension_index=index,
                    value=repr(value),
                    **details,
                )
            )


def validate_sample(
    sample: Any,
    dimension: int,
    known_schemes: Iterable[str],
    *,
    check_id_uniqueness: bool = False,
    seen_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(sample, dict):
        return [make_error(E_SCHEMA_UNKNOWN, "sample must be an object", value=type(sample).__name__)]

    line = sample.get("_line")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        errors.append(make_error(E_FIELD_INVALID, "sample_id must be a nonempty string", line=line))
    elif check_id_uniqueness and seen_ids is not None:
        if sample_id in seen_ids:
            errors.append(make_error(E_SAMPLE_DUPLICATE, "duplicate sample_id", sample_id=sample_id, line=line))
        seen_ids.add(sample_id)

    details = {"sample_id": sample_id, "line": line}
    _nonnegative_int(errors, sample.get("query_position"), "query_position", **details)
    _nonnegative_int(errors, sample.get("train_length"), "train_length", **details)

    key_positions = sample.get("key_positions")
    if not isinstance(key_positions, list) or not key_positions:
        errors.append(make_error(E_FIELD_INVALID, "key_positions must be a nonempty list", **details))
    else:
        seen_positions: set[int] = set()
        for key_index, position in enumerate(key_positions):
            if _nonnegative_int(errors, position, "key_position", key_index=key_index, **details):
                if position in seen_positions:
                    errors.append(
                        make_error(
                            E_KEY_POSITION_DUPLICATE,
                            "key_positions must not contain duplicates",
                            key_index=key_index,
                            position=position,
                            **details,
                        )
                    )
                seen_positions.add(position)

    validate_vector(errors, sample.get("query_vector"), dimension, "query_vector", **details)

    key_vectors = sample.get("key_vectors")
    if not isinstance(key_vectors, list):
        errors.append(make_error(E_DIM_MISMATCH, "key_vectors must be a list", **details))
    else:
        expected_keys = len(key_positions) if isinstance(key_positions, list) else 0
        if expected_keys and len(key_vectors) != expected_keys:
            errors.append(
                make_error(
                    E_DIM_MISMATCH,
                    "key_vectors count must equal key_positions count",
                    expected=expected_keys,
                    actual=len(key_vectors),
                    **details,
                )
            )
        for key_index, vector in enumerate(key_vectors):
            validate_vector(
                errors,
                vector,
                dimension,
                f"key_vectors[{key_index}]",
                key_index=key_index,
                **details,
            )

    declared = sample.get("schemes")
    known = set(known_schemes)
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            errors.append(make_error(E_SCHEMA_UNKNOWN, "sample.schemes must be a nonempty list when provided", **details))
        else:
            for name in declared:
                if name not in known:
                    errors.append(make_error(E_SCHEMA_UNKNOWN, "sample references undeclared scheme", scheme=name, **details))
    return errors


def validate_workspace(config: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = validate_config(config)
    if errors and (not isinstance(config, dict) or not is_int(config.get("dimension"))):
        return errors
    dimension = config["dimension"]
    scheme_names = [s.get("name") for s in config.get("schemes", []) if isinstance(s, dict)]
    seen: set[str] = set()
    for sample in samples:
        errors.extend(
            validate_sample(
                sample,
                dimension,
                scheme_names,
                check_id_uniqueness=True,
                seen_ids=seen,
            )
        )
    return errors
