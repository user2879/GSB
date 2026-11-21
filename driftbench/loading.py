from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigAbort, Failure, sorted_failures
from .fixed import decimal_from
from .models import (
    Config,
    EVENT_FIELDS,
    FILTER_OPS,
    FilterCondition,
    METRIC_ALIASES,
    NORMALIZATION_ALIASES,
    SOURCE_ORDER,
    Exposure,
    Spec,
    WorkspaceState,
)

DEFAULT_FILES = {
    "exposures": "exposures.jsonl",
    "specs": "specs.jsonl",
    "config": "config.json",
    "properties": "user_properties.json",
    "edit_plan": "edit_plan.json",
}

CONFIG_FIELD_ALIASES = {
    "control_group": ("control_group", "control_id"),
    "experiment_group": ("experiment_group", "experiment_id", "treatment_group"),
    "bucket_hash_seed": ("bucket_hash_seed", "seed"),
    "bucket_count": ("bucket_count", "buckets"),
    "active_spec_version": ("active_spec_version", "current_spec_version", "spec_version"),
    "fixed_decimal_places": ("fixed_decimal_places", "decimal_places"),
    "quantization_step": ("quantization_step", "quantum"),
    "concurrency": ("concurrency", "max_concurrency"),
    "batch_size": ("batch_size",),
    "snapshot_commit_interval_batches": (
        "snapshot_commit_interval_batches",
        "snapshot_interval_batches",
    ),
    "bucket_property_key": ("bucket_property_key", "bucket_attribute"),
    "fallback_policy": ("fallback_policy",),
    "missing_attribute_group": ("missing_attribute_group", "fallback_group"),
    "drift_tolerance": ("drift_tolerance", "metric_tolerance"),
    "min_affected_ratio": ("min_affected_ratio", "significance_min_affected_ratio"),
    "attribution_order": ("attribution_order", "attribution_priority"),
    "snapshot_version": ("snapshot_version", "global_version"),
}


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path, *, code: str, failures: list[Failure]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(Failure(code, f"missing input file: {path}", field_path="$"))
    except json.JSONDecodeError as exc:
        failures.append(
            Failure(code, f"invalid JSON: {exc.msg}", field_path="$", line=exc.lineno)
        )
    except OSError as exc:
        failures.append(Failure(code, f"cannot read input file: {exc}", field_path="$"))
    return None


def _read_jsonl(path: Path) -> tuple[list[tuple[int, Any]], list[Failure]]:
    rows: list[tuple[int, Any]] = []
    failures: list[Failure] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], [Failure("E_INPUT", f"cannot open JSONL file: {exc}", field_path="$")]
    with handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    Failure(
                        "E_LOG_INVALID" if path.name.startswith("exposure") else "E_SPEC_INVALID",
                        f"invalid JSONL row: {exc.msg}",
                        line=line_no,
                        field_path="$",
                    )
                )
                continue
            rows.append((line_no, value))
    return rows, failures


def _pick(raw: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def parse_sample_filter(value: Any) -> tuple[tuple[FilterCondition, ...], list[str]]:
    errors: list[str] = []
    conditions: list[FilterCondition] = []
    if value in (None, "", []):
        return (), errors
    if isinstance(value, dict):
        for field, expression in value.items():
            if isinstance(expression, dict):
                for op, operand in expression.items():
                    if op not in FILTER_OPS:
                        errors.append(f"sample_filter.{field}.{op}: unsupported operator")
                        continue
                    conditions.append(FilterCondition(field, op, operand))
            else:
                conditions.append(FilterCondition(field, "eq", expression))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"sample_filter[{index}]: condition must be an object")
                continue
            field = item.get("field")
            op = item.get("op", "eq")
            if "value" not in item:
                errors.append(f"sample_filter[{index}].value: missing value")
                continue
            if not isinstance(field, str):
                errors.append(f"sample_filter[{index}].field: must be a string")
                continue
            if op not in FILTER_OPS:
                errors.append(f"sample_filter[{index}].op: unsupported operator {op!r}")
                continue
            conditions.append(FilterCondition(field, str(op), item["value"]))
    else:
        errors.append("sample_filter: must be an object, array, or null")
    for index, condition in enumerate(conditions):
        if condition.field not in EVENT_FIELDS:
            errors.append(
                f"sample_filter[{index}].field: unknown exposure field {condition.field!r}"
            )
    return tuple(conditions), errors


def validate_spec(raw: Any, line_no: int | None = None) -> tuple[Spec | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return None, [Failure("E_SPEC_INVALID", "spec row must be an object", line=line_no, field_path="$")]

    version = _pick(raw, ("spec_version", "version", "spec_id"))
    metric_name = _pick(raw, ("metric_name", "metric"))
    truncation_rank = _pick(raw, ("truncation_rank", "truncate_at", "top_k"))
    normalization = _pick(raw, ("normalization", "normalisation"))
    dedup_count = _pick(raw, ("dedup_count", "deduplicate"))
    sample_filter = raw.get("sample_filter", raw.get("filter", []))

    if not isinstance(version, str) or not version.strip():
        failures.append(Failure("E_SPEC_INVALID", "spec_version must be a non-empty string", line=line_no, field_path="spec_version"))
    metric_key = str(metric_name).lower() if isinstance(metric_name, str) else ""
    if metric_key not in METRIC_ALIASES:
        failures.append(Failure("E_SPEC_INVALID", f"unsupported metric_name {metric_name!r}", line=line_no, field_path="metric_name"))
    if not is_int(truncation_rank) or truncation_rank <= 0:
        failures.append(Failure("E_SPEC_INVALID", "truncation_rank must be a positive integer", line=line_no, field_path="truncation_rank"))
    normalization_key = str(normalization).lower() if isinstance(normalization, str) else ""
    if normalization_key not in NORMALIZATION_ALIASES:
        failures.append(Failure("E_SPEC_INVALID", f"unsupported normalization {normalization!r}", line=line_no, field_path="normalization"))
    if not isinstance(dedup_count, bool):
        failures.append(Failure("E_SPEC_INVALID", "dedup_count must be a boolean", line=line_no, field_path="dedup_count"))
    conditions, condition_errors = parse_sample_filter(sample_filter)
    for message in condition_errors:
        failures.append(Failure("E_SPEC_INVALID", message, line=line_no, field_path=message.split(":", 1)[0]))
    if failures:
        return None, failures
    assert isinstance(version, str)
    assert isinstance(truncation_rank, int)
    assert isinstance(dedup_count, bool)
    return (
        Spec(
            version=version,
            metric_name=METRIC_ALIASES[metric_key],
            truncation_rank=truncation_rank,
            normalization=NORMALIZATION_ALIASES[normalization_key],
            dedup_count=dedup_count,
            sample_filter=conditions,
        ),
        [],
    )


def validate_config(raw: Any, specs: dict[str, Spec]) -> Config:
    failures: list[Failure] = []

    def invalid(path: str, message: str) -> None:
        failures.append(Failure("E_CONFIG_INVALID", message, field_path=path))

    if not isinstance(raw, dict):
        raise ConfigAbort(Failure("E_CONFIG_INVALID", "config must be an object", field_path="$"))

    values = {key: _pick(raw, aliases) for key, aliases in CONFIG_FIELD_ALIASES.items()}

    for key in ("control_group", "experiment_group"):
        if not isinstance(values[key], str) or not values[key].strip():
            invalid(key, f"{key} must be a non-empty string")
    control = values["control_group"]
    experiment = values["experiment_group"]
    if control == experiment and isinstance(control, str):
        invalid("experiment_group", "control_group and experiment_group must differ")

    if not isinstance(values["bucket_hash_seed"], str) or not values["bucket_hash_seed"].strip():
        invalid("bucket_hash_seed", "bucket_hash_seed must be a non-empty string")
    if not is_int(values["bucket_count"]) or values["bucket_count"] <= 0:
        invalid("bucket_count", "bucket_count must be a positive integer")

    active = values["active_spec_version"]
    if not isinstance(active, str) or not active.strip():
        invalid("active_spec_version", "active_spec_version must be a non-empty string")
    elif active not in specs:
        invalid("active_spec_version", f"active_spec_version {active!r} is absent from the spec catalogue")

    places = values["fixed_decimal_places"]
    if not is_int(places) or not (0 <= places <= 12):
        invalid("fixed_decimal_places", "fixed_decimal_places must be an integer in [0, 12]")
        places = 6
    step = values["quantization_step"]
    if not is_int(step) or step <= 0:
        invalid("quantization_step", "quantization_step must be a positive integer")
    elif is_int(places) and step > 10**places:
        invalid("quantization_step", "quantization_step cannot exceed one unit at the configured scale")

    for key in ("concurrency", "batch_size", "snapshot_commit_interval_batches"):
        if not is_int(values[key]) or values[key] <= 0:
            invalid(key, f"{key} must be a positive integer")

    if not isinstance(values["bucket_property_key"], str) or not values["bucket_property_key"].strip():
        invalid("bucket_property_key", "bucket_property_key must be a non-empty string")

    policy = values["fallback_policy"]
    if policy not in ("user_id_only", "fixed_group"):
        invalid("fallback_policy", "fallback_policy must be 'user_id_only' or 'fixed_group'")
    missing_group = values["missing_attribute_group"]
    if policy == "fixed_group":
        if missing_group not in (control, experiment):
            invalid("missing_attribute_group", "missing_attribute_group must name a configured group")
    elif missing_group is not None and missing_group not in (control, experiment):
        invalid("missing_attribute_group", "missing_attribute_group must name a configured group")

    try:
        tolerance = decimal_from(values["drift_tolerance"])
        if tolerance <= 0:
            invalid("drift_tolerance", "drift_tolerance must be greater than zero")
    except (InvalidOperation, ValueError):
        invalid("drift_tolerance", "drift_tolerance must be a decimal number")
        tolerance = Decimal("0.001")
    try:
        min_ratio = decimal_from(values["min_affected_ratio"])
        if min_ratio < 0 or min_ratio > 1:
            invalid("min_affected_ratio", "min_affected_ratio must be in [0, 1]")
    except (InvalidOperation, ValueError):
        invalid("min_affected_ratio", "min_affected_ratio must be a decimal number")
        min_ratio = Decimal("0.05")

    order = values["attribution_order"]
    order_is_valid = (
        isinstance(order, list)
        and len(order) == len(SOURCE_ORDER)
        and all(isinstance(item, str) for item in order)
        and set(order) == set(SOURCE_ORDER)
    )
    if not order_is_valid:
        invalid(
            "attribution_order",
            "attribution_order must be a permutation of " + ", ".join(SOURCE_ORDER),
        )
        order = list(SOURCE_ORDER)

    if not is_int(values["snapshot_version"]) or values["snapshot_version"] < 0:
        invalid("snapshot_version", "snapshot_version must be a non-negative integer")

    if failures:
        raise ConfigAbort(sorted_failures(failures))

    return Config(
        control_group=control,
        experiment_group=experiment,
        bucket_hash_seed=values["bucket_hash_seed"],
        bucket_count=values["bucket_count"],
        active_spec_version=active,
        fixed_decimal_places=places,
        quantization_step=step,
        concurrency=values["concurrency"],
        batch_size=values["batch_size"],
        snapshot_commit_interval_batches=values["snapshot_commit_interval_batches"],
        bucket_property_key=values["bucket_property_key"],
        fallback_policy=policy,
        missing_attribute_group=missing_group,
        drift_tolerance=tolerance,
        min_affected_ratio=min_ratio,
        attribution_order=tuple(order),
        snapshot_version=values["snapshot_version"],
    )


def validate_exposure(
    raw: Any,
    config: Config,
    seen_request_ids: set[str],
    *,
    line_no: int | None = None,
    injection_id: str | None = None,
) -> tuple[Exposure | None, list[Failure]]:
    failures: list[Failure] = []

    def invalid(path: str, message: str) -> None:
        failures.append(
            Failure(
                "E_LOG_INVALID",
                message,
                line=line_no,
                request_id=raw.get("request_id") if isinstance(raw, dict) and isinstance(raw.get("request_id"), str) else None,
                injection_id=injection_id,
                field_path=path,
            )
        )

    if not isinstance(raw, dict):
        invalid("$", "exposure row must be an object")
        return None, failures

    for field in ("request_id", "experiment_group", "user_id", "item_id"):
        if not isinstance(raw.get(field), str) or not raw.get(field, "").strip():
            invalid(field, f"{field} must be a non-empty string")
    if isinstance(raw.get("experiment_group"), str) and raw["experiment_group"] not in (
        config.control_group,
        config.experiment_group,
    ):
        invalid("experiment_group", "experiment_group is not declared by the experiment config")
    request_id = raw.get("request_id")
    if isinstance(request_id, str) and request_id in seen_request_ids:
        invalid("request_id", "request_id must be unique")
    if not is_int(raw.get("rank")) or raw.get("rank") <= 0:
        invalid("rank", "rank must be a positive integer")
    score = raw.get("model_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        invalid("model_score", "model_score must be numeric")
    elif not math.isfinite(float(score)):
        invalid("model_score", "model_score must be finite")
    for field in ("clicked", "converted"):
        value = raw.get(field)
        if not is_int(value) or value not in (0, 1):
            invalid(field, f"{field} must be integer 0 or 1")
    if not is_int(raw.get("event_ms")) or raw.get("event_ms") < 0:
        invalid("event_ms", "event_ms must be a non-negative integer")

    if failures:
        return None, failures
    return (
        Exposure(
            request_id=raw["request_id"],
            experiment_group=raw["experiment_group"],
            user_id=raw["user_id"],
            item_id=raw["item_id"],
            rank=raw["rank"],
            model_score=float(raw["model_score"]),
            clicked=raw["clicked"],
            converted=raw["converted"],
            event_ms=raw["event_ms"],
        ),
        [],
    )


def load_workspace(sample_dir: str | Path) -> WorkspaceState:
    root = Path(sample_dir)
    failures: list[Failure] = []

    spec_rows, spec_read_failures = _read_jsonl(root / DEFAULT_FILES["specs"])
    failures.extend(spec_read_failures)
    specs: dict[str, Spec] = {}
    for line_no, row in spec_rows:
        spec, row_failures = validate_spec(row, line_no)
        if row_failures:
            failures.extend(row_failures)
            continue
        assert spec is not None
        if spec.version in specs:
            failures.append(
                Failure(
                    "E_SPEC_INVALID",
                    f"duplicate spec_version {spec.version!r}",
                    line=line_no,
                    field_path="spec_version",
                )
            )
            continue
        specs[spec.version] = spec

    raw_config = _read_json(root / DEFAULT_FILES["config"], code="E_CONFIG_INVALID", failures=failures)
    if raw_config is None:
        raise ConfigAbort(sorted_failures(failures))
    try:
        config = validate_config(raw_config, specs)
    except ConfigAbort as exc:
        raise ConfigAbort(sorted_failures(failures + exc.failures)) from exc

    exposure_rows, exposure_read_failures = _read_jsonl(root / DEFAULT_FILES["exposures"])
    failures.extend(exposure_read_failures)
    exposures: dict[str, Exposure] = {}
    seen: set[str] = set()
    for line_no, row in exposure_rows:
        exposure, row_failures = validate_exposure(row, config, seen, line_no=line_no)
        if row_failures:
            failures.extend(row_failures)
            continue
        assert exposure is not None
        exposures[exposure.request_id] = exposure
        seen.add(exposure.request_id)

    properties: dict[str, dict[str, Any]] = {}
    raw_properties = _read_json(root / DEFAULT_FILES["properties"], code="E_PROPERTIES_INVALID", failures=failures)
    if raw_properties is not None:
        users = raw_properties.get("users") if isinstance(raw_properties, dict) and "users" in raw_properties else raw_properties
        if not isinstance(users, dict):
            failures.append(Failure("E_PROPERTIES_INVALID", "user properties must be an object", field_path="$"))
        else:
            for user_id, values in users.items():
                if not isinstance(user_id, str) or not isinstance(values, dict):
                    failures.append(
                        Failure(
                            "E_PROPERTIES_INVALID",
                            "each user property entry must map a string user ID to an object",
                            field_path="$",
                        )
                    )
                    continue
                properties[user_id] = dict(values)

    plan: dict[str, Any] = {}
    plan_path = root / DEFAULT_FILES["edit_plan"]
    if plan_path.exists():
        raw_plan = _read_json(plan_path, code="E_PLAN_INVALID", failures=failures)
        if isinstance(raw_plan, dict):
            plan = raw_plan
            plan_failures = validate_plan(raw_plan)
            failures.extend(plan_failures)
        elif raw_plan is not None:
            failures.append(Failure("E_PLAN_INVALID", "edit plan must be an object", field_path="$"))

    return WorkspaceState(
        config=config,
        specs=specs,
        exposures=exposures,
        properties=properties,
        plan=plan,
        failures=sorted_failures(failures),
        input_paths={key: str(root / name) for key, name in DEFAULT_FILES.items()},
    )


def validate_plan(plan: dict[str, Any]) -> list[Failure]:
    failures: list[Failure] = []
    batches = plan.get("batches")
    if not isinstance(batches, list):
        return [Failure("E_PLAN_INVALID", "edit plan batches must be an array", field_path="batches")]
    seen_batches: set[int] = set()
    allowed = {
        "add_exposures",
        "fix_exposure",
        "switch_spec_version",
        "change_bucket_seed",
    }
    for index, batch in enumerate(batches):
        path = f"batches[{index}]"
        if not isinstance(batch, dict):
            failures.append(Failure("E_PLAN_INVALID", "batch must be an object", field_path=path))
            continue
        number = batch.get("batch")
        if not is_int(number) or number <= 0:
            failures.append(Failure("E_PLAN_INVALID", "batch must be a positive integer", field_path=f"{path}.batch"))
        elif number in seen_batches:
            failures.append(Failure("E_PLAN_INVALID", "batch number must be unique", field_path=f"{path}.batch"))
        else:
            seen_batches.add(number)
        operations = batch.get("operations", [])
        if not isinstance(operations, list):
            failures.append(Failure("E_PLAN_INVALID", "operations must be an array", field_path=f"{path}.operations"))
            continue
        for op_index, operation in enumerate(operations):
            op_path = f"{path}.operations[{op_index}]"
            if not isinstance(operation, dict):
                failures.append(Failure("E_PLAN_INVALID", "operation must be an object", field_path=op_path))
                continue
            op_type = operation.get("type")
            if op_type not in allowed:
                failures.append(Failure("E_PLAN_INVALID", f"unsupported operation type {op_type!r}", field_path=f"{op_path}.type"))
            elif op_type == "add_exposures" and not isinstance(operation.get("events"), list):
                failures.append(Failure("E_PLAN_INVALID", "add_exposures.events must be an array", field_path=f"{op_path}.events"))
            elif op_type == "fix_exposure":
                if not isinstance(operation.get("request_id"), str):
                    failures.append(Failure("E_PLAN_INVALID", "fix_exposure.request_id must be a string", field_path=f"{op_path}.request_id"))
                if not isinstance(operation.get("patch"), dict):
                    failures.append(Failure("E_PLAN_INVALID", "fix_exposure.patch must be an object", field_path=f"{op_path}.patch"))
            elif op_type == "switch_spec_version" and not isinstance(operation.get("spec_version"), str):
                failures.append(Failure("E_PLAN_INVALID", "switch_spec_version.spec_version must be a string", field_path=f"{op_path}.spec_version"))
            elif op_type == "change_bucket_seed" and not isinstance(operation.get("seed"), str):
                failures.append(Failure("E_PLAN_INVALID", "change_bucket_seed.seed must be a string", field_path=f"{op_path}.seed"))
    return failures


def fresh_failures_without_loader_errors(state: WorkspaceState) -> list[Failure]:
    return [failure for failure in state.failures if failure.code not in {"E_LOG_INVALID", "E_SPEC_INVALID"}]
