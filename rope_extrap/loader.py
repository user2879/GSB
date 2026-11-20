from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    E_BASE_INVALID,
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_DIM_MISMATCH,
    E_INTERNAL,
    E_SAMPLE_INVALID,
    E_SCALE_INVALID,
    BusinessError,
    Failure,
)
from .models import (
    PRECISION_F32,
    PRECISION_F64,
    PRUNING_ENERGY,
    PRUNING_FREQUENCY,
    PRUNING_NONE,
    SCHEME_LINEAR,
    SCHEME_NTK,
    SCHEME_STANDARD,
    SUPPORTED_PRECISIONS,
    SUPPORTED_PRUNING_POLICIES,
    SUPPORTED_SCHEMES,
    AttentionSample,
    InteractionCommand,
    PositionEncodingConfig,
)


CONFIG_ALIASES = {
    "scheme": ("scheme", "position_encoding", "pe_scheme"),
    "base": ("base", "rope_base", "theta_base"),
    "training_context_length": ("training_context_length", "train_context_length", "training_length"),
    "extrapolation_target_length": ("extrapolation_target_length", "extrapolation_length", "target_length"),
    "dimension": ("dimension", "dim", "d_model", "head_dim"),
    "scale_factor": ("scale_factor", "scaling_factor", "scale"),
    "precision": ("precision", "precision_profile", "dtype"),
    "dimension_chunk_size": ("dimension_chunk_size", "chunk_size", "dimension_block"),
    "interaction_budget_ms": ("interaction_budget_ms", "latency_budget_ms", "budget_ms"),
    "concurrency": ("concurrency", "max_concurrency", "workers"),
    "snapshot_commit_interval": ("snapshot_commit_interval", "snapshot_interval", "commit_interval"),
    "pruning_tolerance": ("pruning_tolerance", "prune_tolerance"),
    "pruning_policy": ("pruning_policy", "pruning_criterion", "pruning_mode"),
    "error_tolerance": ("error_tolerance", "trace_tolerance", "diagnostic_tolerance"),
}


SCHEME_ALIASES = {
    "standard": SCHEME_STANDARD,
    "standard_rope": SCHEME_STANDARD,
    "rope": SCHEME_STANDARD,
    "rotary": SCHEME_STANDARD,
    "linear": SCHEME_LINEAR,
    "linear_interpolation": SCHEME_LINEAR,
    "position_interpolation": SCHEME_LINEAR,
    "pi": SCHEME_LINEAR,
    "ntk": SCHEME_NTK,
    "ntk_aware": SCHEME_NTK,
    "dynamic_ntk": SCHEME_NTK,
}


PRECISION_ALIASES = {
    "f64": PRECISION_F64,
    "float64": PRECISION_F64,
    "double": PRECISION_F64,
    "high": PRECISION_F64,
    "f32": PRECISION_F32,
    "float32": PRECISION_F32,
    "single": PRECISION_F32,
    "standard": PRECISION_F32,
}


PRUNING_ALIASES = {
    "none": PRUNING_NONE,
    "keep_all": PRUNING_NONE,
    "energy": PRUNING_ENERGY,
    "chunk_energy": PRUNING_ENERGY,
    "contribution": PRUNING_ENERGY,
    "frequency": PRUNING_FREQUENCY,
    "low_frequency": PRUNING_FREQUENCY,
    "frequency_band": PRUNING_FREQUENCY,
}


def _field(raw: dict[str, Any], aliases: Iterable[str]) -> tuple[Any, bool]:
    for name in aliases:
        if name in raw:
            return raw[name], True
    return None, False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _failure(code: str, message: str, *, scope: str = "config", line: int | None = None) -> Failure:
    return Failure(code=code, message=message, stage="load", scope=scope, line=line)


def _normalize_scheme(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return SCHEME_ALIASES.get(value.strip().lower())


def _normalize_precision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return PRECISION_ALIASES.get(value.strip().lower())


def _normalize_pruning(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return PRUNING_ALIASES.get(value.strip().lower())


def parse_position_config(raw: Any) -> tuple[PositionEncodingConfig | None, list[Failure]]:
    if not isinstance(raw, dict):
        return None, [_failure(E_CONFIG_INVALID, "position-encoding config must be a JSON object")]

    failures: list[Failure] = []

    def required(name: str) -> Any:
        value, present = _field(raw, CONFIG_ALIASES[name])
        if not present:
            failures.append(_failure(E_CONFIG_INVALID, f"missing required config field: {name}"))
        return value

    scheme_raw = required("scheme")
    scheme = _normalize_scheme(scheme_raw)
    if scheme_raw is not None and scheme is None:
        failures.append(
            _failure(E_CONFIG_INVALID, f"unsupported scheme: {scheme_raw!r}; expected one of {SUPPORTED_SCHEMES}")
        )

    base_raw = required("base")
    base = float(base_raw) if _is_number(base_raw) else 0.0
    if not _is_number(base_raw):
        failures.append(_failure(E_CONFIG_INVALID, "base must be a finite number"))
    elif not math.isfinite(base) or base <= 0.0:
        failures.append(_failure(E_BASE_INVALID, "base must be greater than zero and finite"))

    training = required("training_context_length")
    if not _is_int(training) or training <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "training_context_length must be a positive integer"))
        training = 0

    target = required("extrapolation_target_length")
    if not _is_int(target) or target <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "extrapolation_target_length must be a positive integer"))
        target = 0

    dimension = required("dimension")
    if not _is_int(dimension) or dimension <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "dimension must be a positive integer"))
        dimension = 0
    elif dimension % 2 != 0:
        failures.append(_failure(E_CONFIG_INVALID, "dimension must be even"))

    scale_raw = required("scale_factor")
    scale = float(scale_raw) if _is_number(scale_raw) else 0.0
    if not _is_number(scale_raw):
        failures.append(_failure(E_CONFIG_INVALID, "scale_factor must be a finite number"))
    elif not math.isfinite(scale) or scale <= 0.0:
        failures.append(_failure(E_SCALE_INVALID, "scale_factor must be greater than zero and finite"))

    precision_raw = required("precision")
    precision = _normalize_precision(precision_raw)
    if precision_raw is not None and precision is None:
        failures.append(
            _failure(E_CONFIG_INVALID, f"unsupported precision profile: {precision_raw!r}; expected {SUPPORTED_PRECISIONS}")
        )

    chunk_size = required("dimension_chunk_size")
    if not _is_int(chunk_size) or chunk_size <= 0 or chunk_size % 2 != 0:
        failures.append(_failure(E_CONFIG_INVALID, "dimension_chunk_size must be a positive even integer"))
        chunk_size = 0
    elif dimension > 0 and dimension % chunk_size != 0:
        failures.append(_failure(E_CONFIG_INVALID, "dimension_chunk_size must divide dimension"))

    budget = required("interaction_budget_ms")
    if not _is_int(budget) or budget <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "interaction_budget_ms must be a positive integer"))
        budget = 0

    concurrency = required("concurrency")
    if not _is_int(concurrency) or concurrency <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "concurrency must be a positive integer"))
        concurrency = 0

    snapshot_interval = required("snapshot_commit_interval")
    if not _is_int(snapshot_interval) or snapshot_interval <= 0:
        failures.append(_failure(E_CONFIG_INVALID, "snapshot_commit_interval must be a positive integer"))
        snapshot_interval = 0

    pruning_tolerance_raw = required("pruning_tolerance")
    pruning_tolerance = float(pruning_tolerance_raw) if _is_number(pruning_tolerance_raw) else -1.0
    if not _is_number(pruning_tolerance_raw) or not math.isfinite(pruning_tolerance):
        failures.append(_failure(E_CONFIG_INVALID, "pruning_tolerance must be a finite number"))
    elif pruning_tolerance < 0.0 or pruning_tolerance > 1.0:
        failures.append(_failure(E_CONFIG_INVALID, "pruning_tolerance must be in [0, 1]"))

    pruning_policy_raw, pruning_policy_present = _field(raw, CONFIG_ALIASES["pruning_policy"])
    pruning_policy = _normalize_pruning(pruning_policy_raw) if pruning_policy_present else PRUNING_ENERGY
    if pruning_policy_present and pruning_policy is None:
        failures.append(
            _failure(
                E_CONFIG_INVALID,
                f"unsupported pruning policy: {pruning_policy_raw!r}; expected {SUPPORTED_PRUNING_POLICIES}",
            )
        )

    error_tolerance_raw, error_tolerance_present = _field(raw, CONFIG_ALIASES["error_tolerance"])
    if error_tolerance_present:
        error_tolerance = float(error_tolerance_raw) if _is_number(error_tolerance_raw) else -1.0
    else:
        error_tolerance = pruning_tolerance if _is_number(pruning_tolerance_raw) else 0.02
    if not _is_number(error_tolerance_raw) and error_tolerance_present:
        failures.append(_failure(E_CONFIG_INVALID, "error_tolerance must be a finite number"))
    elif not math.isfinite(error_tolerance) or error_tolerance <= 0.0:
        failures.append(_failure(E_CONFIG_INVALID, "error_tolerance must be greater than zero and finite"))

    if _is_int(training) and _is_int(target) and training > 0 and target > 0 and training >= target:
        failures.append(
            _failure(E_CONFIG_INVALID, "training_context_length must be less than extrapolation_target_length")
        )

    if failures:
        return None, _dedupe_failures(failures)
    assert scheme is not None and precision is not None and pruning_policy is not None
    config = PositionEncodingConfig(
        scheme=scheme,
        base=base,
        training_context_length=int(training),
        extrapolation_target_length=int(target),
        dimension=int(dimension),
        scale_factor=scale,
        precision=precision,
        dimension_chunk_size=int(chunk_size),
        interaction_budget_ms=int(budget),
        concurrency=int(concurrency),
        snapshot_commit_interval=int(snapshot_interval),
        pruning_tolerance=float(pruning_tolerance),
        pruning_policy=pruning_policy,
        error_tolerance=float(error_tolerance),
    )
    return config, []


def load_position_config(path: str | Path) -> tuple[PositionEncodingConfig | None, list[Failure]]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [_failure(E_CONFIG_INVALID, f"config file not found: {path}")]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [_failure(E_CONFIG_INVALID, f"invalid config JSON: {exc}")]
    return parse_position_config(raw)


def _parse_feature_vector(
    value: Any,
    *,
    dimension: int,
    label: str,
    line: int,
    sample_id: str,
) -> tuple[tuple[float, ...] | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(value, list):
        return None, [
            Failure(
                E_SAMPLE_INVALID,
                f"{label} must be a JSON array",
                stage="load",
                sample_id=sample_id,
                line=line,
            )
        ]
    if len(value) != dimension:
        return None, [
            Failure(
                E_DIM_MISMATCH,
                f"{label} has length {len(value)}, expected dimension {dimension}",
                stage="load",
                sample_id=sample_id,
                line=line,
            )
        ]
    result: list[float] = []
    for index, item in enumerate(value):
        if not _is_number(item):
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"{label}[{index}] must be numeric",
                    stage="load",
                    sample_id=sample_id,
                    line=line,
                )
            )
            continue
        number = float(item)
        if not math.isfinite(number):
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"{label}[{index}] must be finite",
                    stage="load",
                    sample_id=sample_id,
                    line=line,
                )
            )
            continue
        result.append(number)
    if failures:
        return None, failures
    return tuple(result), []


def parse_sample_object(
    raw: Any,
    *,
    dimension: int,
    target_length: int,
    feature_line: int,
    source_ref: str,
) -> tuple[AttentionSample | None, list[Failure]]:
    if not isinstance(raw, dict):
        return None, [
            Failure(
                E_SAMPLE_INVALID,
                "sample must be a JSON object",
                stage="load",
                line=feature_line,
                scope="sample",
            )
        ]
    sample_id_raw = raw.get("sample_id", raw.get("id"))
    if not isinstance(sample_id_raw, str) or not sample_id_raw:
        return None, [
            Failure(
                E_SAMPLE_INVALID,
                "sample_id must be a non-empty string",
                stage="load",
                line=feature_line,
                scope="sample",
            )
        ]
    sample_id = sample_id_raw
    failures: list[Failure] = []

    query_position = raw.get("query_position", raw.get("query_pos"))
    if not _is_int(query_position) or query_position < 0:
        failures.append(
            Failure(
                E_SAMPLE_INVALID,
                "query_position must be a non-negative integer",
                stage="load",
                sample_id=sample_id,
                line=feature_line,
            )
        )
        query_position = 0
    elif query_position > target_length:
        failures.append(
            Failure(
                E_SAMPLE_INVALID,
                f"query_position {query_position} exceeds extrapolation_target_length {target_length}",
                stage="load",
                sample_id=sample_id,
                line=feature_line,
            )
        )

    key_positions_raw = raw.get("key_positions", raw.get("keys"))
    if not isinstance(key_positions_raw, list) or not key_positions_raw:
        failures.append(
            Failure(
                E_SAMPLE_INVALID,
                "key_positions must be a non-empty array of non-negative integers",
                stage="load",
                sample_id=sample_id,
                line=feature_line,
            )
        )
        key_positions_raw = []
    key_positions: list[int] = []
    for index, position in enumerate(key_positions_raw):
        if not _is_int(position) or position < 0:
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"key_positions[{index}] must be a non-negative integer",
                    stage="load",
                    sample_id=sample_id,
                    line=feature_line,
                )
            )
            continue
        if _is_int(query_position) and position > query_position:
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"key_positions[{index}]={position} is out of bounds for query_position {query_position}",
                    stage="load",
                    sample_id=sample_id,
                    line=feature_line,
                )
            )
            continue
        if position > target_length:
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"key_positions[{index}]={position} exceeds extrapolation_target_length {target_length}",
                    stage="load",
                    sample_id=sample_id,
                    line=feature_line,
                )
            )
            continue
        key_positions.append(position)

    query_features_raw = raw.get(
        "query_features",
        raw.get("query_vector", raw.get("q_features", raw.get("features"))),
    )
    query_features, query_failures = _parse_feature_vector(
        query_features_raw,
        dimension=dimension,
        label="query_features",
        line=feature_line,
        sample_id=sample_id,
    )
    failures.extend(query_failures)

    key_features_raw = raw.get("key_features", raw.get("key_vectors", raw.get("k_features")))
    key_features: list[tuple[float, ...]] = []
    if key_features_raw is None and raw.get("features") is not None and query_features is not None:
        # Convenience format: one shared deterministic feature row for every key.
        key_features = [query_features for _ in key_positions_raw]
    elif not isinstance(key_features_raw, list):
        failures.append(
            Failure(
                E_SAMPLE_INVALID,
                "key_features must be an array of dimension-length feature rows",
                stage="load",
                sample_id=sample_id,
                line=feature_line,
            )
        )
    else:
        if len(key_features_raw) != len(key_positions_raw):
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    "key_features length must equal key_positions length",
                    stage="load",
                    sample_id=sample_id,
                    line=feature_line,
                )
            )
        for index, row in enumerate(key_features_raw):
            row_features, row_failures = _parse_feature_vector(
                row,
                dimension=dimension,
                label=f"key_features[{index}]",
                line=feature_line,
                sample_id=sample_id,
            )
            failures.extend(row_failures)
            if row_features is not None:
                key_features.append(row_features)

    if failures:
        return None, _dedupe_failures(failures)
    assert query_features is not None
    return (
        AttentionSample(
            sample_id=sample_id,
            query_position=int(query_position),
            key_positions=tuple(int(item) for item in key_positions),
            query_features=query_features,
            key_features=tuple(key_features),
            feature_line=feature_line,
            source_ref=source_ref,
        ),
        [],
    )


def load_samples(
    path: str | Path,
    *,
    dimension: int,
    target_length: int,
) -> tuple[dict[str, AttentionSample], list[Failure]]:
    path = Path(path)
    failures: list[Failure] = []
    samples: dict[str, AttentionSample] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}, [_failure(E_SAMPLE_INVALID, f"sample file not found: {path}", scope="sample")]
    except (OSError, UnicodeError) as exc:
        return {}, [_failure(E_SAMPLE_INVALID, f"cannot read sample file: {exc}", scope="sample")]
    nonempty_count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        nonempty_count += 1
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"invalid sample JSON: {exc.msg}",
                    stage="load",
                    line=line_number,
                    scope="sample",
                )
            )
            continue
        sample, sample_failures = parse_sample_object(
            raw,
            dimension=dimension,
            target_length=target_length,
            feature_line=line_number,
            source_ref=f"file:{path.name}",
        )
        failures.extend(sample_failures)
        if sample is None:
            continue
        if sample.sample_id in samples:
            failures.append(
                Failure(
                    E_SAMPLE_INVALID,
                    f"duplicate sample_id: {sample.sample_id}",
                    stage="load",
                    sample_id=sample.sample_id,
                    line=line_number,
                    scope="sample",
                )
            )
            continue
        samples[sample.sample_id] = sample
    if nonempty_count == 0:
        failures.append(_failure(E_SAMPLE_INVALID, "sample file is empty", scope="sample"))
    return dict(sorted(samples.items())), _dedupe_failures(failures)


def _dedupe_failures(failures: Iterable[Failure]) -> list[Failure]:
    unique: dict[tuple[object, ...], Failure] = {}
    for failure in failures:
        key = (
            failure.code,
            failure.stage,
            failure.sample_id,
            failure.line,
            failure.scope,
            failure.message,
        )
        unique[key] = failure
    return sorted(unique.values(), key=lambda item: item.sort_key())


def _parse_positive_budget(raw: dict[str, Any]) -> tuple[int | None, list[Failure]]:
    value = raw.get("budget_ms", raw.get("interaction_budget_ms"))
    if value is None:
        return None, []
    if not _is_int(value) or value <= 0:
        return None, [
            Failure(
                E_COMMAND_INVALID,
                "command budget_ms must be a positive integer",
                stage="load",
                scope="command",
            )
        ]
    return value, []


def parse_command_object(
    raw: Any,
    *,
    index: int,
    config: PositionEncodingConfig,
) -> tuple[InteractionCommand | None, list[Failure]]:
    if not isinstance(raw, dict):
        return None, [
            Failure(E_COMMAND_INVALID, "command must be a JSON object", stage="load", line=index, scope="command")
        ]
    command_type_raw = raw.get("type", raw.get("op", raw.get("command")))
    if not isinstance(command_type_raw, str):
        return None, [
            Failure(E_COMMAND_INVALID, "command type must be a string", stage="load", line=index, scope="command")
        ]
    command_type = command_type_raw.strip().lower()
    budget_ms, budget_failures = _parse_positive_budget(raw)
    failures: list[Failure] = list(budget_failures)
    changes: dict[str, Any] = {}
    replacement: AttentionSample | None = None

    def set_scheme(value: Any) -> None:
        scheme = _normalize_scheme(value)
        if scheme is None:
            failures.append(
                Failure(
                    E_COMMAND_INVALID,
                    f"invalid scheme in command: {value!r}",
                    stage="load",
                    line=index,
                    scope="command",
                )
            )
        else:
            changes["scheme"] = scheme

    def set_base(value: Any) -> None:
        if not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0.0:
            failures.append(
                Failure(E_BASE_INVALID, "command base must be > 0", stage="load", line=index, scope="command")
            )
        else:
            changes["base"] = float(value)

    def set_scale(value: Any) -> None:
        if not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0.0:
            failures.append(
                Failure(
                    E_SCALE_INVALID,
                    "command scale_factor must be > 0",
                    stage="load",
                    line=index,
                    scope="command",
                )
            )
        else:
            changes["scale_factor"] = float(value)

    def set_training(value: Any) -> None:
        if not _is_int(value) or value <= 0 or value >= config.extrapolation_target_length:
            failures.append(
                Failure(
                    E_COMMAND_INVALID,
                    "training_context_length must be a positive integer below extrapolation_target_length",
                    stage="load",
                    line=index,
                    scope="command",
                )
            )
        else:
            changes["training_context_length"] = int(value)

    def set_pruning(policy_value: Any = None, tolerance_value: Any = None) -> None:
        if policy_value is not None:
            policy = _normalize_pruning(policy_value)
            if policy is None:
                failures.append(
                    Failure(
                        E_COMMAND_INVALID,
                        f"invalid pruning policy: {policy_value!r}",
                        stage="load",
                        line=index,
                        scope="command",
                    )
                )
            else:
                changes["pruning_policy"] = policy
        if tolerance_value is not None:
            if (
                not _is_number(tolerance_value)
                or not math.isfinite(float(tolerance_value))
                or float(tolerance_value) < 0.0
                or float(tolerance_value) > 1.0
            ):
                failures.append(
                    Failure(
                        E_COMMAND_INVALID,
                        "pruning_tolerance must be in [0, 1]",
                        stage="load",
                        line=index,
                        scope="command",
                    )
                )
            else:
                changes["pruning_tolerance"] = float(tolerance_value)

    def replace_sample(value: Any) -> None:
        nonlocal replacement
        if value is None and any(
            name in raw for name in ("sample_id", "query_position", "key_positions", "query_features")
        ):
            value = raw
        sample, sample_failures = parse_sample_object(
            value,
            dimension=config.dimension,
            target_length=config.extrapolation_target_length,
            feature_line=index,
            source_ref=f"command:{index}",
        )
        failures.extend(sample_failures)
        if sample is not None:
            replacement = sample

    if command_type in {"set_scheme", "scheme"}:
        set_scheme(raw.get("scheme", raw.get("value")))
    elif command_type in {"set_base", "base"}:
        set_base(raw.get("base", raw.get("value")))
    elif command_type in {"set_scale", "set_scaling", "scale"}:
        set_scale(raw.get("scale_factor", raw.get("scale", raw.get("value"))))
    elif command_type in {"set_context", "set_training_context_length", "training_context_length"}:
        set_training(raw.get("training_context_length", raw.get("context_length", raw.get("value"))))
    elif command_type in {"set_pruning", "set_pruning_policy"}:
        policy_value = raw.get("pruning_policy", raw.get("pruning_criterion"))
        tolerance_value = raw.get("pruning_tolerance", raw.get("prune_tolerance"))
        if policy_value is None and tolerance_value is None:
            failures.append(
                Failure(
                    E_COMMAND_INVALID,
                    "set_pruning requires pruning_policy and/or pruning_tolerance",
                    stage="load",
                    line=index,
                    scope="command",
                )
            )
        else:
            set_pruning(policy_value, tolerance_value)
    elif command_type in {"set_pruning_tolerance"}:
        set_pruning(None, raw.get("pruning_tolerance", raw.get("tolerance", raw.get("value"))))
    elif command_type in {"replace_sample", "replace"}:
        replace_sample(raw.get("sample", raw))
    elif command_type in {"poll", "advance", "resume"}:
        pass
    elif command_type in {"update", "configure", "set_parameters"}:
        if "scheme" in raw:
            set_scheme(raw["scheme"])
        if "base" in raw:
            set_base(raw["base"])
        if "scale_factor" in raw or "scale" in raw:
            set_scale(raw.get("scale_factor", raw.get("scale")))
        if "training_context_length" in raw:
            set_training(raw["training_context_length"])
        if "pruning_policy" in raw or "pruning_tolerance" in raw:
            set_pruning(raw.get("pruning_policy"), raw.get("pruning_tolerance"))
        if "sample" in raw:
            replace_sample(raw["sample"])
        if not changes and replacement is None:
            failures.append(
                Failure(
                    E_COMMAND_INVALID,
                    "update command contains no recognized changes",
                    stage="load",
                    line=index,
                    scope="command",
                )
            )
    else:
        failures.append(
            Failure(
                E_COMMAND_INVALID,
                f"unsupported command type: {command_type_raw!r}",
                stage="load",
                line=index,
                scope="command",
            )
        )

    if failures:
        return None, _dedupe_failures(failures)
    command_id_raw = raw.get("command_id", raw.get("id", str(index)))
    command_id = str(command_id_raw)
    return (
        InteractionCommand(
            index=index,
            command_id=command_id,
            command_type=command_type,
            changes=changes,
            sample=replacement,
            budget_ms=budget_ms,
            raw=raw,
        ),
        [],
    )


def load_commands(
    path: str | Path,
    *,
    config: PositionEncodingConfig,
) -> tuple[list[InteractionCommand], list[Failure]]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [
            Failure(E_COMMAND_INVALID, f"command file not found: {path}", stage="load", scope="command")
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [
            Failure(E_COMMAND_INVALID, f"invalid command JSON: {exc}", stage="load", scope="command")
        ]
    if isinstance(raw, dict) and isinstance(raw.get("commands"), list):
        raw_commands = raw["commands"]
    elif isinstance(raw, list):
        raw_commands = raw
    else:
        return [], [
            Failure(
                E_COMMAND_INVALID,
                "command file must be an array or an object with a commands array",
                stage="load",
                scope="command",
            )
        ]
    commands: list[InteractionCommand] = []
    failures: list[Failure] = []
    for index, item in enumerate(raw_commands, start=1):
        command, command_failures = parse_command_object(item, index=index, config=config)
        failures.extend(command_failures)
        if command is not None:
            commands.append(command)
    return commands, _dedupe_failures(failures)


def load_command_file_for_config(
    path: str | Path,
    config_path: str | Path,
) -> tuple[list[InteractionCommand], list[Failure]]:
    config, failures = load_position_config(config_path)
    if config is None:
        return [], failures + [_failure(E_INTERNAL, "cannot parse commands without a valid config", scope="command")]
    return load_commands(path, config=config)
