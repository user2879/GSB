from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import codes
from .config import is_finite_number, is_int
from .hashing import fingerprint


@dataclass
class LoadResult:
    samples: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    code_counts: dict[str, int] = field(default_factory=dict)

    def count_code(self, code: str) -> None:
        self.code_counts[code] = self.code_counts.get(code, 0) + 1


def sample_content_fingerprint(sample: dict[str, Any]) -> str:
    # Length markers affect statistics only and are intentionally excluded here.
    content = {
        "id": sample["id"],
        "dimension": sample["dimension"],
        "query_position": sample["query_position"],
        "query_vector": sample["query_vector"],
        "key_positions": sample["key_positions"],
        "key_vectors": sample["key_vectors"],
        "content_kernel": "integer-seed-v1",
    }
    return fingerprint("sample-content:v1", content)


def _failure(line_number: int, sample_id: str | None,
             errors: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "sample_id": sample_id or f"<line:{line_number}>",
        "accepted": False,
        "errors": errors,
    }


def _validate_vector(value: Any, dim: int, field_name: str,
                     errors: list[dict[str, str]]) -> None:
    if not isinstance(value, list):
        errors.append({"code": codes.E_FEATURE_VECTOR, "field": field_name,
                       "message": "must be an array"})
        return
    if len(value) != dim:
        errors.append({"code": codes.E_FEATURE_VECTOR, "field": field_name,
                       "message": f"expected length {dim}, got {len(value)}"})
    bad_indexes = []
    for index, item in enumerate(value):
        if not is_finite_number(item):
            bad_indexes.append(index)
    if bad_indexes:
        errors.append({
            "code": codes.E_FEATURE_VECTOR,
            "field": f"{field_name}[{bad_indexes[0]}]",
            "message": f"{len(bad_indexes)} non-finite/non-numeric elements",
        })


def validate_sample_record(
    raw: Any,
    line_number: int,
    seen_ids: set[str],
    default_dimension: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    errors: list[dict[str, str]] = []
    preliminary_id: str | None = None

    if not isinstance(raw, dict):
        errors.append({"code": codes.E_SAMPLE_SCHEMA, "field": "$",
                       "message": "sample must be a JSON object"})
        return None, _failure(line_number, None, errors)

    raw_id = raw.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        errors.append({"code": codes.E_SAMPLE_ID, "field": "id",
                       "message": "id must be a non-empty string"})
    else:
        preliminary_id = raw_id
        if raw_id in seen_ids:
            errors.append({"code": codes.E_SAMPLE_ID_DUPLICATE, "field": "id",
                           "message": f"duplicate sample id {raw_id!r}"})

    dim = raw.get("dimension", default_dimension)
    if not is_int(dim) or dim <= 0 or dim % 2 != 0:
        errors.append({"code": codes.E_DIMENSION, "field": "dimension",
                       "message": "dimension must be a positive even integer"})
        dim = None
    elif default_dimension is not None and dim != default_dimension:
        errors.append({"code": codes.E_DIMENSION, "field": "dimension",
                       "message": f"must equal configured dimension {default_dimension}"})

    qpos = raw.get("query_position")
    if not is_int(qpos) or qpos < 0:
        errors.append({"code": codes.E_QUERY_POSITION, "field": "query_position",
                       "message": "must be a non-negative integer"})

    train_length = raw.get("train_length")
    target_length = raw.get("target_length")
    if not is_int(train_length) or train_length <= 0:
        errors.append({"code": codes.E_TRAIN_LENGTH, "field": "train_length",
                       "message": "must be a positive integer"})
    if not is_int(target_length) or target_length <= 0:
        errors.append({"code": codes.E_TARGET_LENGTH, "field": "target_length",
                       "message": "must be a positive integer"})
    if is_int(train_length) and is_int(target_length) and train_length > target_length:
        errors.append({"code": codes.E_LENGTH_ORDER, "field": "train_length",
                       "message": "train_length must not exceed target_length"})

    key_positions = raw.get("key_positions")
    valid_key_positions: list[int] = []
    if not isinstance(key_positions, list) or not key_positions:
        errors.append({"code": codes.E_KEY_POSITIONS, "field": "key_positions",
                       "message": "must be a non-empty array"})
    else:
        seen_positions: set[int] = set()
        for index, position in enumerate(key_positions):
            if not is_int(position) or position < 0:
                errors.append({"code": codes.E_KEY_POSITION,
                               "field": f"key_positions[{index}]",
                               "message": "must be a non-negative integer"})
                continue
            if position in seen_positions:
                errors.append({"code": codes.E_KEY_POSITION_DUPLICATE,
                               "field": f"key_positions[{index}]",
                               "message": f"duplicate key position {position}"})
            else:
                seen_positions.add(position)
                valid_key_positions.append(position)

    if dim is not None:
        _validate_vector(raw.get("query_vector"), dim, "query_vector", errors)
        key_vectors = raw.get("key_vectors")
        if not isinstance(key_vectors, list):
            errors.append({"code": codes.E_KEY_VECTORS, "field": "key_vectors",
                           "message": "must be an array"})
        else:
            expected_keys = len(key_positions) if isinstance(key_positions, list) else 0
            if len(key_vectors) != expected_keys:
                errors.append({"code": codes.E_KEY_VECTORS, "field": "key_vectors",
                               "message": (
                                   f"length {len(key_vectors)} does not match "
                                   f"key_positions length {expected_keys}"
                               )})
            for index, vector in enumerate(key_vectors):
                _validate_vector(vector, dim, f"key_vectors[{index}]", errors)

    if errors:
        return None, _failure(line_number, preliminary_id, errors)

    assert preliminary_id is not None
    seen_ids.add(preliminary_id)
    sample = {
        "id": preliminary_id,
        "dimension": int(dim),
        "query_position": int(qpos),
        "train_length": int(train_length),
        "target_length": int(target_length),
        "query_vector": [float(x) for x in raw["query_vector"]],
        "key_positions": [int(x) for x in key_positions],
        "key_vectors": [[float(x) for x in vector] for vector in raw["key_vectors"]],
    }
    sample["content_fingerprint"] = sample_content_fingerprint(sample)
    return sample, None


def load_samples_jsonl(
    path: str | Path,
    default_dimension: int | None = None,
) -> LoadResult:
    result = LoadResult()
    path = Path(path)
    seen_ids: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failure = _failure(line_number, None, [{
                "code": codes.E_SAMPLE_PARSE,
                "field": f"line:{line_number}",
                "message": f"JSON parse error: {exc.msg}",
            }])
            result.failures.append(failure)
            result.count_code(codes.E_SAMPLE_PARSE)
            continue
        sample, failure = validate_sample_record(
            raw, line_number, seen_ids, default_dimension,
        )
        if failure is not None:
            result.failures.append(failure)
            for item in failure["errors"]:
                result.count_code(item["code"])
        else:
            assert sample is not None
            result.samples.append(sample)
    return result


def samples_by_id(samples: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {sample["id"]: sample for sample in sorted(samples, key=lambda item: item["id"])}


def business_code_counts(failures: Iterable[dict[str, Any]],
                         extra: dict[str, int] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        for item in failure.get("errors", []):
            code = item["code"]
            counts[code] = counts.get(code, 0) + 1
    for code, count in (extra or {}).items():
        counts[code] = counts.get(code, 0) + count
    return dict(sorted(counts.items()))
