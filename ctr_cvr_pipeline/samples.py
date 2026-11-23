from __future__ import annotations

import json
import math
from typing import Any, Iterable

from .config import FeatureConfig
from .errors import Failure, sample_identity
from .state import Sample


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _failure(
    request_id: Any,
    item_id: Any,
    field: str,
    message: str,
    line_no: int | None = None,
    command_index: int | None = None,
    details: dict[str, Any] | None = None,
) -> Failure:
    return Failure(
        code="E_SAMPLE_INVALID",
        message=message,
        request_id=str(request_id) if request_id not in (None, "") else None,
        item_id=str(item_id) if item_id not in (None, "") else None,
        sample_id=sample_identity(request_id, item_id, line_no),
        field=field,
        command_index=command_index,
        details=details or {},
    )


def validate_sample_record(
    raw: Any,
    config: FeatureConfig,
    *,
    event_time_ms: int,
    batch_id: str,
    seen_keys: set[tuple[str, str]],
    line_no: int | None = None,
    command_index: int | None = None,
) -> tuple[Sample | None, list[Failure]]:
    if not isinstance(raw, dict):
        return None, [
            _failure(
                None,
                None,
                "record",
                "sample must be a JSON object",
                line_no,
                command_index,
            )
        ]

    request_id = raw.get("request_id")
    item_id = raw.get("item_id")
    failures: list[Failure] = []
    if not isinstance(request_id, str) or not request_id.strip():
        failures.append(
            _failure(request_id, item_id, "request_id", "must be a non-empty string", line_no, command_index)
        )
    if not isinstance(item_id, str) or not item_id.strip():
        failures.append(
            _failure(request_id, item_id, "item_id", "must be a non-empty string", line_no, command_index)
        )

    dense_raw = raw.get("dense", raw.get("dense_features", raw.get("dense_feature_vector", {})))
    dense: dict[str, float] = {}
    if isinstance(dense_raw, list):
        dense_names = [feature.name for feature in config.dense_features]
        if len(dense_raw) != len(dense_names):
            failures.append(
                _failure(
                    request_id,
                    item_id,
                    "dense",
                    f"dense vector length {len(dense_raw)} does not match configured {len(dense_names)}",
                    line_no,
                    command_index,
                )
            )
        else:
            dense = dict(zip(dense_names, dense_raw))
    elif isinstance(dense_raw, dict):
        dense = dict(dense_raw)
    else:
        failures.append(
            _failure(request_id, item_id, "dense", "must be an object or vector", line_no, command_index)
        )
    for name, value in dense.items():
        if not isinstance(name, str) or not name:
            failures.append(
                _failure(request_id, item_id, "dense.<name>", "feature name must be non-empty", line_no, command_index)
            )
            continue
        if not _is_finite_nonnegative(value):
            failures.append(
                _failure(
                    request_id,
                    item_id,
                    f"dense.{name}",
                    "must be finite non-negative number",
                    line_no,
                    command_index,
                )
            )

    sparse_raw = raw.get("sparse", raw.get("sparse_counts", {}))
    if not isinstance(sparse_raw, dict):
        failures.append(
            _failure(request_id, item_id, "sparse_counts", "must be an object", line_no, command_index)
        )
        sparse_raw = {}
    sparse: dict[str, float] = {}
    for name, value in sparse_raw.items():
        if not isinstance(name, str) or not name:
            failures.append(
                _failure(
                    request_id,
                    item_id,
                    "sparse.<name>",
                    "feature name must be non-empty",
                    line_no,
                    command_index,
                )
            )
            continue
        if not _is_finite_nonnegative(value):
            failures.append(
                _failure(
                    request_id,
                    item_id,
                    f"sparse.{name}",
                    "count must be finite non-negative number",
                    line_no,
                    command_index,
                )
            )
            continue
        sparse[name] = float(value)

    click = raw.get("click", raw.get("click_label"))
    conversion = raw.get("conversion", raw.get("conversion_label"))
    delay = raw.get("conversion_delay_ms", raw.get("conversion_delay"))
    if not _is_int(click) or click not in (0, 1):
        failures.append(
            _failure(request_id, item_id, "click", "must be integer 0 or 1", line_no, command_index)
        )
    if not _is_int(conversion) or conversion not in (0, 1):
        failures.append(
            _failure(request_id, item_id, "conversion", "must be integer 0 or 1", line_no, command_index)
        )
    if delay is not None and (not _is_int(delay) or delay < 0):
        failures.append(
            _failure(
                request_id,
                item_id,
                "conversion_delay_ms",
                "must be non-negative integer or null",
                line_no,
                command_index,
            )
        )

    if failures:
        return None, failures

    key = (request_id, item_id)
    if key in seen_keys:
        return None, [
            _failure(
                request_id,
                item_id,
                "item_id",
                "candidate item id must be unique within request",
                line_no,
                command_index,
            )
        ]

    normalized_dense = {
        name: float(value) for name, value in dense.items() if _is_finite_nonnegative(value)
    }
    normalized_sparse = {
        name: float(value) for name, value in sparse.items() if _is_finite_nonnegative(value)
    }
    return (
        Sample(
            request_id=request_id,
            item_id=item_id,
            dense=normalized_dense,
            sparse=normalized_sparse,
            click=int(click),
            conversion=int(conversion),
            conversion_delay_ms=int(delay) if delay is not None else None,
            event_time_ms=event_time_ms,
            batch_id=batch_id,
        ),
        [],
    )


def load_samples_jsonl(
    path: str,
    config: FeatureConfig,
    *,
    event_time_ms: int,
    batch_id: str = "b0000",
) -> tuple[list[Sample], list[Failure]]:
    samples: list[Sample] = []
    failures: list[Failure] = []
    seen: set[tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(
                    _failure(
                        None,
                        None,
                        "line",
                        f"invalid JSON: {exc.msg}",
                        line_no=line_no,
                    )
                )
                continue
            sample, problems = validate_sample_record(
                raw,
                config,
                event_time_ms=event_time_ms,
                batch_id=batch_id,
                seen_keys=seen,
                line_no=line_no,
            )
            failures.extend(problems)
            if sample is not None:
                samples.append(sample)
                seen.add(sample.key)
    return samples, failures


def validate_label_update(
    request_id: Any,
    item_id: Any,
    *,
    click: Any,
    conversion: Any,
    conversion_delay_ms: Any,
    command_index: int,
) -> list[Failure]:
    failures: list[Failure] = []
    if not _is_int(click) or click not in (0, 1):
        failures.append(
            _failure(request_id, item_id, "click", "must be integer 0 or 1", command_index=command_index)
        )
    if not _is_int(conversion) or conversion not in (0, 1):
        failures.append(
            _failure(
                request_id,
                item_id,
                "conversion",
                "must be integer 0 or 1",
                command_index=command_index,
            )
        )
    if conversion_delay_ms is not None and (
        not _is_int(conversion_delay_ms) or conversion_delay_ms < 0
    ):
        failures.append(
            _failure(
                request_id,
                item_id,
                "conversion_delay_ms",
                "must be non-negative integer or null",
                command_index=command_index,
            )
        )
    return failures


def effective_dense_value(sample: Sample, name: str) -> float:
    return float(sample.dense.get(name, 0.0))


def effective_sparse_value(sample: Sample, name: str) -> float:
    return float(sample.sparse.get(name, 0.0))
