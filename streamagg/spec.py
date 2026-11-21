from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import FORMAT_VERSION, BusinessError
from .model import (
    SUPPORTED_AGGREGATIONS,
    RuleSpec,
    SessionSpec,
    SourceSpec,
)


def _fail(field: str, message: str, code: str = "E_SPEC") -> None:
    raise BusinessError(code=code, message=message, field=field)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_int(obj: dict[str, Any], field: str, path: str) -> int:
    if field not in obj:
        _fail(f"{path}.{field}", "missing required integer field")
    value = obj[field]
    if not _is_int(value):
        _fail(f"{path}.{field}", "must be an integer")
    return value


def _positive_int(obj: dict[str, Any], field: str, path: str) -> int:
    value = _required_int(obj, field, path)
    if value <= 0:
        _fail(f"{path}.{field}", "must be a positive integer")
    return value


def _string_list(obj: dict[str, Any], field: str, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    if field not in obj:
        _fail(f"{path}.{field}", "missing required string array")
    value = obj[field]
    if not isinstance(value, list):
        _fail(f"{path}.{field}", "must be an array")
    if not value and not allow_empty:
        _fail(f"{path}.{field}", "must not be empty")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            _fail(f"{path}.{field}[{index}]", "must be a non-empty string")
        if item in seen:
            _fail(f"{path}.{field}[{index}]", "duplicate value")
        seen.add(item)
        result.append(item)
    return tuple(result)


def _optional_int(obj: dict[str, Any], field: str, path: str, default: int) -> int:
    if field not in obj:
        return default
    value = obj[field]
    if not _is_int(value):
        _fail(f"{path}.{field}", "must be an integer")
    return value


def parse_spec(data: Any) -> SessionSpec:
    if not isinstance(data, dict):
        _fail("$", "specification must be a JSON object")

    format_version = _required_int(data, "format_version", "$")
    if format_version != FORMAT_VERSION:
        _fail("$.format_version", f"unsupported format version {format_version}")

    sources_raw = data.get("sources")
    if not isinstance(sources_raw, list):
        _fail("$.sources", "must be an array")
    if not sources_raw:
        _fail("$.sources", "must contain at least one source")

    rules_raw = data.get("rules")
    if not isinstance(rules_raw, list):
        _fail("$.rules", "must be an array")
    if not rules_raw:
        _fail("$.rules", "must contain at least one rule")

    source_ids: set[str] = set()
    sources: list[SourceSpec] = []
    for index, raw in enumerate(sources_raw):
        path = f"$.sources[{index}]"
        if not isinstance(raw, dict):
            _fail(path, "must be an object")
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            _fail(f"{path}.source_id", "must be a non-empty string")
        if source_id in source_ids:
            _fail(f"{path}.source_id", "duplicate source_id")
        source_ids.add(source_id)
        seed = _positive_int(raw, "seed", path)
        event_count = _positive_int(raw, "event_count", path)
        ratio = _required_int(raw, "out_of_order_ratio_bp", path)
        if ratio < 0 or ratio > 10_000:
            _fail(f"{path}.out_of_order_ratio_bp", "must be between 0 and 10000")
        partition_keys = (
            _string_list(raw, "partition_keys", path, allow_empty=False)
            if "partition_keys" in raw
            else ()
        )
        base_time_ms = _optional_int(raw, "base_time_ms", path, 1_700_000_000_000)
        time_step_ms = _optional_int(raw, "time_step_ms", path, 20)
        if time_step_ms <= 0:
            _fail(f"{path}.time_step_ms", "must be a positive integer")
        measure_min = _optional_int(raw, "measure_min", path, -1000)
        measure_max = _optional_int(raw, "measure_max", path, 1000)
        if measure_min > measure_max:
            _fail(f"{path}.measure_min", "must not exceed measure_max")
        reorder_delay_events = _optional_int(raw, "reorder_delay_events", path, 1)
        if reorder_delay_events < 0:
            _fail(f"{path}.reorder_delay_events", "must be non-negative")
        sources.append(
            SourceSpec(
                source_id=source_id,
                seed=seed,
                event_count=event_count,
                out_of_order_ratio_bp=ratio,
                partition_keys=partition_keys,
                base_time_ms=base_time_ms,
                time_step_ms=time_step_ms,
                measure_min=measure_min,
                measure_max=measure_max,
                reorder_delay_events=reorder_delay_events,
            )
        )

    rule_ids: set[str] = set()
    rules: list[RuleSpec] = []
    for index, raw in enumerate(rules_raw):
        path = f"$.rules[{index}]"
        if not isinstance(raw, dict):
            _fail(path, "must be an object")
        rule_id = raw.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            _fail(f"{path}.rule_id", "must be a non-empty string")
        if rule_id in rule_ids:
            _fail(f"{path}.rule_id", "duplicate rule_id")
        rule_ids.add(rule_id)
        width_ms = _positive_int(raw, "width_ms", path)
        lateness_ms = _positive_int(raw, "lateness_ms", path)
        partition_keys = _string_list(raw, "partition_keys", path, allow_empty=False)
        aggregation_functions = _string_list(
            raw, "aggregation_functions", path, allow_empty=False
        )
        normalized: list[str] = []
        for agg_index, aggregation in enumerate(aggregation_functions):
            upper = aggregation.upper()
            if upper not in SUPPORTED_AGGREGATIONS:
                _fail(
                    f"{path}.aggregation_functions[{agg_index}]",
                    f"unsupported aggregation {aggregation!r}",
                )
            if upper in normalized:
                _fail(
                    f"{path}.aggregation_functions[{agg_index}]",
                    "duplicate aggregation function",
                )
            normalized.append(upper)
        rule_version = _optional_int(raw, "rule_version", path, 1)
        if rule_version <= 0:
            _fail(f"{path}.rule_version", "must be a positive integer")
        rules.append(
            RuleSpec(
                rule_id=rule_id,
                width_ms=width_ms,
                lateness_ms=lateness_ms,
                partition_keys=partition_keys,
                aggregation_functions=tuple(normalized),
                rule_version=rule_version,
            )
        )

    session_raw = data.get("session")
    if not isinstance(session_raw, dict):
        _fail("$.session", "must be an object")
    fixed_scale = _required_int(session_raw, "fixed_scale", "$.session")
    if fixed_scale <= 0:
        _fail("$.session.fixed_scale", "must be a positive integer", code="E_GRID")
    budget_count = _required_int(session_raw, "lateness_budget_count", "$.session")
    if budget_count < 0:
        _fail("$.session.lateness_budget_count", "must be non-negative")
    budget_amount = _required_int(session_raw, "lateness_budget_amount", "$.session")
    if budget_amount < 0:
        _fail("$.session.lateness_budget_amount", "must be non-negative")
    dominant_threshold_bp = _optional_int(
        session_raw, "dominant_threshold_bp", "$.session", 6000
    )
    if dominant_threshold_bp < 0 or dominant_threshold_bp > 10_000:
        _fail("$.session.dominant_threshold_bp", "must be between 0 and 10000")
    order_backward_limit_ms = _optional_int(
        session_raw, "order_backward_limit_ms", "$.session", 86_400_000
    )
    if order_backward_limit_ms <= 0:
        _fail("$.session.order_backward_limit_ms", "must be a positive integer")

    return SessionSpec(
        format_version=format_version,
        sources=tuple(sources),
        rules=tuple(rules),
        fixed_scale=fixed_scale,
        lateness_budget_count=budget_count,
        lateness_budget_amount=budget_amount,
        dominant_threshold_bp=dominant_threshold_bp,
        order_backward_limit_ms=order_backward_limit_ms,
    )


def load_spec(path: str | Path) -> SessionSpec:
    spec_path = Path(path)
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail("$", f"specification file not found: {spec_path}")
    except json.JSONDecodeError as exc:
        _fail("$", f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")
    except OSError as exc:
        _fail("$", f"cannot read specification: {exc}")
    return parse_spec(raw)
