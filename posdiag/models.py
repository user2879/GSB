from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALL_SCHEMES: tuple[str, ...] = ("rope_linear", "rope_ntk", "rope_standard")

SCHEME_ALIASES = {
    "rope": "rope_standard",
    "rope_standard": "rope_standard",
    "standard": "rope_standard",
    "rope_linear": "rope_linear",
    "rope_pi": "rope_linear",
    "linear": "rope_linear",
    "linear_interpolation": "rope_linear",
    "rope_ntk": "rope_ntk",
    "ntk": "rope_ntk",
    "ntk_aware": "rope_ntk",
}

PRECISION_ALIASES = {
    "strict": "strict",
    "fp64": "strict",
    "float64": "strict",
    "high": "strict",
    "balanced": "balanced",
    "medium": "balanced",
    "compact": "compact",
    "fp32": "compact",
    "low": "compact",
}

PRECISION_DECIMALS = {"strict": 12, "balanced": 9, "compact": 6}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    field: str | None = None
    line: int | None = None
    sample_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            out["field"] = self.field
        if self.line is not None:
            out["line"] = self.line
        if self.sample_id is not None:
            out["sample_id"] = self.sample_id
        return out


class ConfigValidationError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


def _is_int(value: Any) -> bool:
    return type(value) is int


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PositionConfig:
    scheme: str
    base_frequency: float
    training_context_length: int
    extrapolation_target_length: int
    dimension: int
    scaling_factor: float
    precision: str
    batch_point_budget: int
    global_mac_budget: int
    thread_pool_concurrency: int
    snapshot_commit_interval: int
    tolerance: float
    global_version: int = 1
    comparison_schemes: tuple[str, ...] = ()

    @property
    def active_schemes(self) -> tuple[str, ...]:
        return tuple(sorted(set((self.scheme,) + self.comparison_schemes)))

    @property
    def precision_decimals(self) -> int:
        return PRECISION_DECIMALS[self.precision]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "base_frequency": self.base_frequency,
            "training_context_length": self.training_context_length,
            "extrapolation_target_length": self.extrapolation_target_length,
            "dimension": self.dimension,
            "scaling_factor": self.scaling_factor,
            "precision": self.precision,
            "batch_point_budget": self.batch_point_budget,
            "global_mac_budget": self.global_mac_budget,
            "thread_pool_concurrency": self.thread_pool_concurrency,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "tolerance": self.tolerance,
            "global_version": self.global_version,
            "comparison_schemes": list(self.comparison_schemes),
            "active_schemes": list(self.active_schemes),
        }

    def fingerprint(self) -> str:
        return sha256_value(self.as_dict())


def _normalise_scheme(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return SCHEME_ALIASES.get(value.strip().lower())


def _normalise_precision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return PRECISION_ALIASES.get(value.strip().lower())


def parse_config_dict(raw: Mapping[str, Any], *, cpu_count: int | None = None) -> PositionConfig:
    issues: list[ValidationIssue] = []

    def add(field: str, message: str) -> None:
        issues.append(ValidationIssue("E_CONFIG_INVALID", message, field=field))

    scheme = _normalise_scheme(raw.get("scheme"))
    if scheme is None:
        add("scheme", "scheme must be one of rope_standard, rope_linear, rope_ntk")

    base_frequency = raw.get("base_frequency")
    if not _finite_number(base_frequency) or float(base_frequency) <= 0.0:
        add("base_frequency", "base_frequency must be a positive finite number")

    training_length = raw.get("training_context_length")
    if not _positive_int(training_length):
        add("training_context_length", "training_context_length must be a positive integer")

    target_length = raw.get("extrapolation_target_length")
    if not _positive_int(target_length):
        add("extrapolation_target_length", "extrapolation_target_length must be a positive integer")
    elif _positive_int(training_length) and int(training_length) >= int(target_length):
        add("extrapolation_target_length", "training_context_length must be smaller than extrapolation_target_length")

    dimension = raw.get("dimension")
    if not _positive_int(dimension) or int(dimension) % 2 != 0:
        add("dimension", "dimension must be a positive even integer")

    scaling_factor = raw.get("scaling_factor")
    if not _finite_number(scaling_factor) or float(scaling_factor) <= 0.0:
        add("scaling_factor", "scaling_factor must be a positive finite number")

    precision = _normalise_precision(raw.get("precision", "strict"))
    if precision is None:
        add("precision", "precision must be strict, balanced or compact")

    batch_point_budget = raw.get("batch_point_budget")
    if not _positive_int(batch_point_budget):
        add("batch_point_budget", "batch_point_budget must be a positive integer")

    global_mac_budget = raw.get("global_mac_budget")
    if not _positive_int(global_mac_budget):
        add("global_mac_budget", "global_mac_budget must be a positive integer")

    concurrency = raw.get("thread_pool_concurrency")
    if not _positive_int(concurrency):
        add("thread_pool_concurrency", "thread_pool_concurrency must be a positive integer")
    elif concurrency > (cpu_count if cpu_count is not None else (os.cpu_count() or 1)):
        add("thread_pool_concurrency", "thread_pool_concurrency must not exceed available CPU cores")

    commit_interval = raw.get("snapshot_commit_interval")
    if not _positive_int(commit_interval):
        add("snapshot_commit_interval", "snapshot_commit_interval must be a positive integer")

    tolerance = raw.get("tolerance")
    if not _finite_number(tolerance) or float(tolerance) <= 0.0:
        add("tolerance", "tolerance must be a positive finite number")

    global_version = raw.get("global_version", 1)
    if not _positive_int(global_version):
        add("global_version", "global_version must be a positive integer")

    comparison_raw = raw.get("comparison_schemes", [])
    comparison_schemes: list[str] = []
    if not isinstance(comparison_raw, list):
        add("comparison_schemes", "comparison_schemes must be a list of scheme names")
    else:
        for index, item in enumerate(comparison_raw):
            normalised = _normalise_scheme(item)
            if normalised is None:
                add(f"comparison_schemes[{index}]", "comparison_schemes contains an unknown scheme")
            else:
                comparison_schemes.append(normalised)

    if issues:
        raise ConfigValidationError(issues)

    return PositionConfig(
        scheme=scheme or "rope_standard",
        base_frequency=float(base_frequency),
        training_context_length=int(training_length),
        extrapolation_target_length=int(target_length),
        dimension=int(dimension),
        scaling_factor=float(scaling_factor),
        precision=precision or "strict",
        batch_point_budget=int(batch_point_budget),
        global_mac_budget=int(global_mac_budget),
        thread_pool_concurrency=int(concurrency),
        snapshot_commit_interval=int(commit_interval),
        tolerance=float(tolerance),
        global_version=int(global_version),
        comparison_schemes=tuple(sorted(set(comparison_schemes))),
    )


def load_config(path: str | Path, *, cpu_count: int | None = None) -> PositionConfig:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(
            [ValidationIssue("E_CONFIG_INVALID", f"cannot load config {path}: {exc}", field="config")]
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            [ValidationIssue("E_CONFIG_INVALID", "config root must be a JSON object", field="config")]
        )
    return parse_config_dict(raw, cpu_count=cpu_count)


def with_active_schemes(config: PositionConfig, schemes: Iterable[str]) -> PositionConfig:
    canonical: list[str] = []
    for item in schemes:
        normalised = _normalise_scheme(item)
        if normalised is None:
            raise ValueError(f"unknown scheme: {item}")
        canonical.append(normalised)
    unique = sorted(set(canonical))
    if not unique:
        raise ValueError("at least one scheme is required")
    primary = config.scheme if config.scheme in unique else unique[0]
    return replace(config, scheme=primary, comparison_schemes=tuple(item for item in unique if item != primary))


@dataclass(frozen=True)
class AttentionSample:
    sample_id: str
    query_position: int
    key_positions: tuple[int, ...]
    query_vector: tuple[float, ...]
    key_vectors: tuple[tuple[float, ...], ...]
    source_line: int | None = None

    @property
    def point_units(self) -> int:
        return 1 + 2 * len(self.key_positions)

    @property
    def empty_keys(self) -> bool:
        return not self.key_positions

    @property
    def self_attention(self) -> bool:
        return self.query_position in self.key_positions

    def content_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_position": self.query_position,
            "key_positions": list(self.key_positions),
            "query_vector": list(self.query_vector),
            "key_vectors": [list(vector) for vector in self.key_vectors],
        }

    def fingerprint(self) -> str:
        return sha256_value(self.content_dict())


def _coerce_vector(value: Any, dimension: int | None, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if dimension is not None and len(value) != dimension:
        raise ValueError(f"{field} length {len(value)} does not match dimension {dimension}")
    if not value:
        raise ValueError(f"{field} must not be empty")
    out: list[float] = []
    for index, item in enumerate(value):
        if not _finite_number(item):
            raise ValueError(f"{field}[{index}] must be finite")
        out.append(float(item))
    return tuple(out)


def parse_sample_object(raw: Mapping[str, Any], *, dimension: int | None, source_line: int | None = None) -> AttentionSample:
    sample_id_raw = raw.get("sample_id", raw.get("id"))
    if not isinstance(sample_id_raw, str) or not sample_id_raw.strip():
        raise ValueError("sample_id must be a non-empty string")
    sample_id = sample_id_raw.strip()

    query_position = raw.get("query_position", raw.get("query_pos"))
    if not _is_int(query_position) or query_position < 0:
        raise ValueError("query_position must be a non-negative integer")

    key_positions_raw = raw.get("key_positions", raw.get("keys"))
    if not isinstance(key_positions_raw, list):
        raise ValueError("key_positions must be a list")
    key_positions: list[int] = []
    for index, item in enumerate(key_positions_raw):
        if not _is_int(item) or item < 0:
            raise ValueError(f"key_positions[{index}] must be a non-negative integer")
        key_positions.append(item)

    query_vector_raw = raw.get("query_vector", raw.get("query_features"))
    query_vector = _coerce_vector(query_vector_raw, dimension, field="query_vector")

    if "key_vectors" in raw or "key_features" in raw:
        key_vectors_raw = raw.get("key_vectors", raw.get("key_features"))
        if not isinstance(key_vectors_raw, list):
            raise ValueError("key_vectors must be a list")
        if len(key_vectors_raw) != len(key_positions):
            raise ValueError("key_vectors length must match key_positions length")
        key_vectors = tuple(
            _coerce_vector(item, dimension, field=f"key_vectors[{index}]")
            for index, item in enumerate(key_vectors_raw)
        )
    else:
        shared = raw.get("key_vector")
        if shared is None and len(key_positions) == 0:
            key_vectors = ()
        elif shared is None:
            raise ValueError("key_vectors is required when key_positions is not empty")
        else:
            vector = _coerce_vector(shared, dimension, field="key_vector")
            key_vectors = tuple(vector for _ in key_positions)

    return AttentionSample(
        sample_id=sample_id,
        query_position=int(query_position),
        key_positions=tuple(key_positions),
        query_vector=query_vector,
        key_vectors=key_vectors,
        source_line=source_line,
    )


@dataclass(frozen=True)
class SampleLoadResult:
    samples: tuple[AttentionSample, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def by_id(self) -> dict[str, AttentionSample]:
        return {sample.sample_id: sample for sample in self.samples}


def load_samples(path: str | Path, *, dimension: int) -> SampleLoadResult:
    path = Path(path)
    samples: list[AttentionSample] = []
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return SampleLoadResult(
            samples=(),
            issues=(ValidationIssue("E_SAMPLE_INVALID", f"cannot open samples file {path}: {exc}", line=0),),
        )

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    ValidationIssue(
                        "E_SAMPLE_INVALID",
                        f"invalid JSON: {exc.msg}",
                        line=line_number,
                    )
                )
                continue
            if not isinstance(raw, dict):
                issues.append(
                    ValidationIssue("E_SAMPLE_INVALID", "sample line must be a JSON object", line=line_number)
                )
                continue

            declared_id = raw.get("sample_id", raw.get("id"))
            sample_id = declared_id.strip() if isinstance(declared_id, str) and declared_id.strip() else None
            if sample_id is not None and sample_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        "E_SAMPLE_INVALID",
                        "duplicate sample_id",
                        line=line_number,
                        sample_id=sample_id,
                    )
                )
                continue
            if sample_id is not None:
                seen_ids.add(sample_id)

            try:
                sample = parse_sample_object(raw, dimension=dimension, source_line=line_number)
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        "E_SAMPLE_INVALID",
                        str(exc),
                        line=line_number,
                        sample_id=sample_id,
                    )
                )
                continue
            samples.append(sample)

    samples.sort(key=lambda item: item.sample_id)
    return SampleLoadResult(samples=tuple(samples), issues=tuple(issues))


@dataclass(frozen=True)
class EditPlanResult:
    config: PositionConfig
    samples: tuple[AttentionSample, ...]
    issues: tuple[ValidationIssue, ...]
    applied_operations: tuple[dict[str, Any], ...]
    plan_fingerprint: str | None


def apply_edit_plan(
    config: PositionConfig,
    samples: Sequence[AttentionSample],
    plan: Mapping[str, Any] | None,
) -> EditPlanResult:
    current_config = config
    sample_map = {sample.sample_id: sample for sample in samples}
    issues: list[ValidationIssue] = []
    applied: list[dict[str, Any]] = []

    operations: Any = []
    if plan is not None:
        operations = plan.get("operations", plan.get("edits", []))
        if not isinstance(operations, list):
            issues.append(
                ValidationIssue("E_EDIT_INVALID", "edit plan operations must be a list", field="operations")
            )
            operations = []

    for index, raw_op in enumerate(operations):
        if not isinstance(raw_op, dict):
            issues.append(
                ValidationIssue("E_EDIT_INVALID", "edit operation must be an object", field=f"operations[{index}]")
            )
            continue
        op = raw_op.get("op")
        try:
            if op == "switch_scheme":
                normalised = _normalise_scheme(raw_op.get("scheme"))
                if normalised is None:
                    raise ValueError("switch_scheme requires a known scheme")
                comparison = raw_op.get("comparison_schemes", [])
                if not isinstance(comparison, list):
                    raise ValueError("comparison_schemes must be a list")
                comparison_normalised: list[str] = []
                for item in comparison:
                    item_normalised = _normalise_scheme(item)
                    if item_normalised is None:
                        raise ValueError("comparison_schemes contains an unknown scheme")
                    comparison_normalised.append(item_normalised)
                current_config = replace(
                    current_config,
                    scheme=normalised,
                    comparison_schemes=tuple(sorted(set(comparison_normalised))),
                )
            elif op == "adjust_base_frequency":
                has_factor = "factor" in raw_op
                has_value = "base_frequency" in raw_op
                if has_factor == has_value:
                    raise ValueError("adjust_base_frequency requires exactly one of factor or base_frequency")
                if has_factor:
                    factor = raw_op.get("factor")
                    if not _finite_number(factor) or float(factor) <= 0.0:
                        raise ValueError("factor must be a positive finite number")
                    current_config = replace(
                        current_config,
                        base_frequency=current_config.base_frequency * float(factor),
                    )
                else:
                    value = raw_op.get("base_frequency")
                    if not _finite_number(value) or float(value) <= 0.0:
                        raise ValueError("base_frequency must be a positive finite number")
                    current_config = replace(current_config, base_frequency=float(value))
            elif op == "replace_sample":
                target_id_raw = raw_op.get("sample_id")
                if not isinstance(target_id_raw, str) or not target_id_raw.strip():
                    raise ValueError("replace_sample requires sample_id")
                target_id = target_id_raw.strip()
                if target_id not in sample_map:
                    raise ValueError(f"replace_sample target {target_id} does not exist")
                raw_sample = raw_op.get("sample")
                if not isinstance(raw_sample, dict):
                    raise ValueError("replace_sample requires a sample object")
                replacement_id = raw_sample.get("sample_id", raw_sample.get("id", target_id))
                if replacement_id != target_id:
                    raise ValueError("replacement sample_id must match replace_sample sample_id")
                replacement = parse_sample_object(raw_sample, dimension=current_config.dimension)
                sample_map[target_id] = replacement
            elif op == "change_precision":
                precision = _normalise_precision(raw_op.get("precision"))
                if precision is None:
                    raise ValueError("change_precision requires strict, balanced or compact")
                current_config = replace(current_config, precision=precision)
            elif op == "change_scaling_factor":
                value = raw_op.get("scaling_factor")
                if not _finite_number(value) or float(value) <= 0.0:
                    raise ValueError("scaling_factor must be a positive finite number")
                current_config = replace(current_config, scaling_factor=float(value))
            elif op == "change_training_context":
                value = raw_op.get("training_context_length")
                if not _positive_int(value):
                    raise ValueError("training_context_length must be a positive integer")
                if int(value) >= current_config.extrapolation_target_length:
                    raise ValueError("training_context_length must remain smaller than extrapolation_target_length")
                current_config = replace(current_config, training_context_length=int(value))
            elif op == "change_extrapolation_target":
                value = raw_op.get("extrapolation_target_length")
                if not _positive_int(value) or int(value) <= current_config.training_context_length:
                    raise ValueError("extrapolation_target_length must be greater than training_context_length")
                current_config = replace(current_config, extrapolation_target_length=int(value))
            elif op == "change_tolerance":
                value = raw_op.get("tolerance")
                if not _finite_number(value) or float(value) <= 0.0:
                    raise ValueError("tolerance must be a positive finite number")
                current_config = replace(current_config, tolerance=float(value))
            elif op == "change_budgets":
                updates: dict[str, Any] = {}
                for field_name in ("batch_point_budget", "global_mac_budget", "snapshot_commit_interval"):
                    if field_name in raw_op:
                        value = raw_op[field_name]
                        if not _positive_int(value):
                            raise ValueError(f"{field_name} must be a positive integer")
                        updates[field_name] = int(value)
                current_config = replace(current_config, **updates)
            elif op == "change_concurrency":
                value = raw_op.get("thread_pool_concurrency")
                cpu_count = os.cpu_count() or 1
                if not _positive_int(value) or int(value) > cpu_count:
                    raise ValueError("thread_pool_concurrency must be positive and not exceed available CPU cores")
                current_config = replace(current_config, thread_pool_concurrency=int(value))
            elif op == "bump_version":
                value = raw_op.get("global_version")
                if not _positive_int(value):
                    raise ValueError("global_version must be a positive integer")
                current_config = replace(current_config, global_version=int(value))
            else:
                raise ValueError(f"unknown edit operation: {op!r}")
        except ValueError as exc:
            issues.append(
                ValidationIssue("E_EDIT_INVALID", str(exc), field=f"operations[{index}]")
            )
            continue
        applied.append({"index": index, "op": op, "apply": "ok"})

    # Revalidate invariants introduced by combinations of edits.
    try:
        checked = parse_config_dict(current_config.as_dict(), cpu_count=os.cpu_count())
    except ConfigValidationError as exc:
        issues.extend(exc.issues)
        checked = current_config

    ordered_samples = tuple(sorted(sample_map.values(), key=lambda item: item.sample_id))
    fingerprint = sha256_value(plan) if plan is not None else None
    return EditPlanResult(
        config=checked,
        samples=ordered_samples,
        issues=tuple(issues),
        applied_operations=tuple(applied),
        plan_fingerprint=fingerprint,
    )


def load_edit_plan(config: PositionConfig, samples: Sequence[AttentionSample], path: str | Path | None) -> EditPlanResult:
    if path is None:
        return EditPlanResult(
            config=config,
            samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
            issues=(),
            applied_operations=(),
            plan_fingerprint=None,
        )
    plan_path = Path(path)
    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return EditPlanResult(
            config=config,
            samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
            issues=(ValidationIssue("E_EDIT_INVALID", f"cannot load edit plan {plan_path}: {exc}", field="plan"),),
            applied_operations=(),
            plan_fingerprint=None,
        )
    if not isinstance(raw, dict):
        return EditPlanResult(
            config=config,
            samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
            issues=(ValidationIssue("E_EDIT_INVALID", "edit plan root must be a JSON object", field="plan"),),
            applied_operations=(),
            plan_fingerprint=None,
        )
    return apply_edit_plan(config, samples, raw)


def merge_sample_issues(*groups: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    merged: list[ValidationIssue] = []
    for group in groups:
        merged.extend(group)
    merged.sort(key=lambda issue: (issue.line if issue.line is not None else -1, issue.code, issue.message))
    return tuple(merged)
