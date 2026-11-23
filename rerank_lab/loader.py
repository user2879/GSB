"""Strict deterministic loaders for the sample directory."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import yaml

from .canonical import (
    as_fraction,
    digest,
    is_finite_number,
    parse_json_constant,
    round_half_away,
)
from .errors import (
    E_BUDGET_INVALID,
    E_INPUT_INVALID,
    Failure,
    sorted_failures,
)
from .models import (
    MAX_CONCURRENCY,
    VALID_NORMALIZATIONS,
    FeatureCatalog,
    FeatureGroup,
    LabConfig,
    LoadedSample,
    RawCandidate,
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _load_json(path: Path) -> Any:
    return json.loads(
        _read_text(path),
        parse_float=Decimal,
        parse_int=int,
        parse_constant=parse_json_constant,
    )


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(_read_text(path))


def _load_structured(path: Path) -> Any:
    if path.suffix.lower() in (".yaml", ".yml"):
        return _load_yaml(path)
    return _load_json(path)


def _first_path(sample_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = sample_dir / name
        if path.is_file():
            return path
    return None


def _get_values(obj: dict[str, Any], aliases: Iterable[str], default: Any = None) -> Any:
    for name in aliases:
        if name in obj:
            return obj[name]
    return default


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _positive_int(value: Any, label: str, failures: list[Failure], code: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        failures.append(
            Failure(code, "validation", f"{label} must be a positive integer")
        )
        return None
    return value


def _nonnegative_int(value: Any, label: str, failures: list[Failure], code: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        failures.append(
            Failure(code, "validation", f"{label} must be a non-negative integer")
        )
        return None
    return value


def _fraction(value: Any) -> Fraction:
    return as_fraction(value)


def _parse_weight_value(value: Any) -> Fraction:
    return _fraction(value)


def _parse_candidate_record(
    record: Any,
    line_no: int,
    failures: list[Failure],
) -> RawCandidate | None:
    if not isinstance(record, dict):
        failures.append(
            Failure(E_INPUT_INVALID, "load", "candidate line must be a JSON object", line=line_no)
        )
        return None

    request_raw = _get_values(record, ("request_id", "requestId", "query_id", "queryId", "qid"))
    item_raw = _get_values(record, ("item_id", "itemId", "doc_id", "docId", "candidate_id"))
    request_id = request_raw if isinstance(request_raw, str) else ""
    item_id = item_raw if isinstance(item_raw, str) else ""
    if not request_id.strip():
        failures.append(
            Failure(E_INPUT_INVALID, "load", "request_id must be a non-empty string", line=line_no)
        )
        return None
    if not item_id.strip():
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "load",
                "item_id must be a non-empty string",
                request_id=request_id,
                line=line_no,
            )
        )
        return None

    recall_raw = _get_values(record, ("recall_scores", "recallScores", "recall_scores_map"), {})
    dense_raw = _get_values(
        record,
        ("dense_features", "denseFeatures", "dense_feature_vector", "dense_vector"),
        [],
    )
    sparse_raw = _get_values(
        record, ("sparse_features", "sparseFeatures", "sparse_feature_counts"), {}
    )

    recall_scores: dict[str, Fraction] = {}
    if isinstance(recall_raw, dict):
        recall_items = list(recall_raw.items())
    elif isinstance(recall_raw, list):
        recall_items = list(enumerate(recall_raw))
    else:
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "load",
                "recall_scores must be an object or array",
                request_id=request_id,
                item_id=item_id,
                line=line_no,
            )
        )
        return None
    for key, value in recall_items:
        if not is_finite_number(value):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    f"recall score {key!r} must be finite",
                    request_id=request_id,
                    item_id=item_id,
                    line=line_no,
                )
            )
            return None
        recall_scores[f"recall.{key}"] = _fraction(value)

    if not isinstance(dense_raw, list):
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "load",
                "dense_features must be an array",
                request_id=request_id,
                item_id=item_id,
                line=line_no,
            )
        )
        return None
    dense_features: list[Fraction] = []
    for index, value in enumerate(dense_raw):
        if not is_finite_number(value):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    f"dense_features[{index}] must be finite",
                    request_id=request_id,
                    item_id=item_id,
                    line=line_no,
                )
            )
            return None
        dense_features.append(_fraction(value))

    if not isinstance(sparse_raw, dict):
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "load",
                "sparse_features must be an object",
                request_id=request_id,
                item_id=item_id,
                line=line_no,
            )
        )
        return None
    sparse_features: dict[str, int] = {}
    for key, value in sparse_raw.items():
        if not isinstance(key, str) or not key:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    "sparse feature names must be non-empty strings",
                    request_id=request_id,
                    item_id=item_id,
                    line=line_no,
                )
            )
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    f"sparse count {key!r} must be a non-negative integer",
                    request_id=request_id,
                    item_id=item_id,
                    line=line_no,
                )
            )
            return None
        sparse_features[key] = value

    features: dict[str, Fraction] = {}
    features.update(recall_scores)
    features.update({f"dense.{index}": value for index, value in enumerate(dense_features)})
    features.update({f"sparse.{key}": Fraction(value) for key, value in sparse_features.items()})

    return RawCandidate(
        request_id=request_id,
        item_id=item_id,
        features=features,
        recall_scores=recall_scores,
        dense_features=tuple(dense_features),
        sparse_features=sparse_features,
        line=line_no,
        source_record=record,
    )


def _load_candidates(path: Path, failures: list[Failure]) -> list[RawCandidate]:
    candidates: list[RawCandidate] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(_read_text(path).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    parse_float=Decimal,
                    parse_int=int,
                    parse_constant=parse_json_constant,
                )
            except (json.JSONDecodeError, InvalidOperation) as exc:
                failures.append(
                    Failure(E_INPUT_INVALID, "load", f"invalid JSONL: {exc}", line=line_no)
                )
                continue
            candidate = _parse_candidate_record(record, line_no, failures)
            if candidate is not None:
                candidates.append(candidate)
    else:
        data = _load_json(path)
        if isinstance(data, dict):
            data = _get_values(data, ("candidates", "records", "items"), [])
        if not isinstance(data, list):
            failures.append(
                Failure(E_INPUT_INVALID, "load", "candidates JSON must be an array")
            )
            return []
        for index, record in enumerate(data, start=1):
            candidate = _parse_candidate_record(record, index, failures)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _parse_groups(raw: Any, failures: list[Failure]) -> tuple[list[str], list[FeatureGroup], list[str]]:
    if raw is None:
        return [], [], []
    if isinstance(raw, list):
        group_values = raw
        raw = {"groups": group_values}
    if not isinstance(raw, dict):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "feature group table must be an object or array")
        )
        return [], [], []

    declared_order_raw = _get_values(
        raw,
        ("feature_order", "features_order", "ordered_features", "dimension_names"),
        [],
    )
    declared_order: list[str] = []
    if declared_order_raw:
        if not isinstance(declared_order_raw, list) or not all(
            isinstance(name, str) and name for name in declared_order_raw
        ):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    "feature_order must be an array of non-empty strings",
                )
            )
        else:
            declared_order = list(declared_order_raw)
            if len(set(declared_order)) != len(declared_order):
                failures.append(
                    Failure(E_INPUT_INVALID, "validation", "feature_order contains duplicates")
                )

    groups_raw = _get_values(raw, ("groups", "feature_groups"), [])
    if not isinstance(groups_raw, list):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "groups must be an array")
        )
        groups_raw = []
    groups: list[FeatureGroup] = []
    seen_group_names: set[str] = set()
    seen_features: set[str] = set()
    for index, group_raw in enumerate(groups_raw):
        if not isinstance(group_raw, dict):
            failures.append(
                Failure(E_INPUT_INVALID, "validation", f"group[{index}] must be an object")
            )
            continue
        name = _get_values(group_raw, ("name", "group", "id"), f"group-{index}")
        if not isinstance(name, str) or not name:
            failures.append(
                Failure(E_INPUT_INVALID, "validation", f"group[{index}] name is invalid")
            )
            continue
        if name in seen_group_names:
            failures.append(
                Failure(E_INPUT_INVALID, "validation", f"duplicate group name {name!r}")
            )
            continue
        features_raw = _get_values(group_raw, ("features", "members"), [])
        if not isinstance(features_raw, list) or not all(
            isinstance(feature, str) and feature for feature in features_raw
        ):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"group {name!r} features must be non-empty strings",
                )
            )
            continue
        duplicate = sorted(set(features_raw) & seen_features)
        if duplicate:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"features assigned to multiple groups: {duplicate}",
                )
            )
        seen_group_names.add(name)
        seen_features.update(features_raw)
        groups.append(
            FeatureGroup(
                name=name,
                features=tuple(features_raw),
                conflict=bool(_get_values(group_raw, ("conflict", "scale_conflict"), False)),
            )
        )

    similarity_raw = _get_values(
        raw,
        ("similarity_features", "diversity_features", "sim_features"),
        [],
    )
    similarity_features: list[str] = []
    if similarity_raw:
        if not isinstance(similarity_raw, list) or not all(
            isinstance(name, str) and name for name in similarity_raw
        ):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    "similarity_features must be an array of non-empty strings",
                )
            )
        else:
            similarity_features = list(similarity_raw)
    return declared_order, groups, similarity_features


def _catalog_from(
    declared_order: list[str],
    groups: list[FeatureGroup],
    similarity_features: list[str],
    all_candidate_features: set[str],
    threshold: Fraction,
    raw: dict[str, Any],
    failures: list[Failure],
) -> FeatureCatalog:
    group_features: list[str] = []
    for group in groups:
        group_features.extend(group.features)
    if declared_order:
        feature_order = list(declared_order)
        missing = sorted(group_features + similarity_features)
        missing = [name for name in missing if name not in feature_order]
        if missing:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"features missing from feature_order: {sorted(set(missing))}",
                )
            )
    elif group_features:
        feature_order = list(dict.fromkeys(group_features))
    else:
        feature_order = sorted(all_candidate_features)

    if not feature_order:
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "feature dimension must not be empty")
        )

    if declared_order or group_features:
        unknown = sorted(all_candidate_features - set(feature_order))
        if unknown:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"candidate features are not declared by feature grouping: {unknown}",
                )
            )

    if not similarity_features:
        dense = [name for name in feature_order if name.startswith("dense.")]
        similarity_features = dense or list(feature_order)
    bad_similarity = [name for name in similarity_features if name not in feature_order]
    if bad_similarity:
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                f"similarity features are not declared: {sorted(bad_similarity)}",
            )
        )
        similarity_features = [name for name in similarity_features if name in feature_order]

    return FeatureCatalog(
        feature_order=tuple(feature_order),
        groups=tuple(groups),
        similarity_features=tuple(similarity_features),
        similarity_threshold=threshold,
        raw=raw,
    )


def _parse_config_raw(
    raw: Any,
    catalog: FeatureCatalog,
    failures: list[Failure],
) -> LabConfig | None:
    if not isinstance(raw, dict):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "config must be a JSON object")
        )
        return None
    before = len(failures)

    normalization_raw = _get_values(raw, ("normalization", "normalizer", "normalize"))
    if not isinstance(normalization_raw, str):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "normalization must be a string")
        )
        normalization = "minmax"
    else:
        normalization = normalization_raw.strip().lower().replace("-", "").replace("_", "")
        if normalization == "minmax":
            normalization = "minmax"
        elif normalization in ("zscore", "standard", "standardscore"):
            normalization = "zscore"
        else:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"normalization must be one of {VALID_NORMALIZATIONS}",
                )
            )
            normalization = "minmax"

    weights_raw = _get_values(
        raw,
        ("feature_weights", "weights", "weight_vector"),
    )
    weights: tuple[Fraction, ...] | None = None
    if isinstance(weights_raw, dict):
        unknown = sorted(set(weights_raw) - set(catalog.feature_order))
        missing = [name for name in catalog.feature_order if name not in weights_raw]
        if unknown:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"feature weight keys are unknown: {unknown}",
                )
            )
        if missing:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"feature weight keys are missing: {missing}",
                )
            )
        if not unknown and not missing:
            try:
                weights = tuple(
                    _parse_weight_value(weights_raw[name])
                    for name in catalog.feature_order
                )
            except (TypeError, ValueError):
                failures.append(
                    Failure(
                        E_INPUT_INVALID,
                        "validation",
                        "feature weights must be finite numbers",
                    )
                )
    elif isinstance(weights_raw, list):
        if len(weights_raw) != catalog.dimension:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"feature weight length {len(weights_raw)} does not match feature dimension {catalog.dimension}",
                )
            )
        try:
            weights = tuple(_parse_weight_value(value) for value in weights_raw)
        except (TypeError, ValueError):
            failures.append(
                Failure(E_INPUT_INVALID, "validation", "feature weights must be finite numbers")
            )
    else:
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                "feature_weights must be an object keyed by feature name or an array",
            )
        )

    temp_raw = _get_values(raw, ("fusion_temperature", "temperature", "fusion_temp"))
    try:
        temperature = _fraction(temp_raw)
        if temperature <= 0:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "fusion_temperature must be positive and finite")
        )
        temperature = Fraction(1)

    clip_raw = _get_values(raw, ("score_clip_interval", "score_interval", "clamping_interval"))
    if isinstance(clip_raw, (list, tuple)) and len(clip_raw) == 2:
        score_min_raw, score_max_raw = clip_raw
    else:
        score_min_raw = _get_values(raw, ("score_min", "min_score", "score_lower_bound"))
        score_max_raw = _get_values(raw, ("score_max", "max_score", "score_upper_bound"))
    try:
        score_min = _fraction(score_min_raw)
        score_max = _fraction(score_max_raw)
        if score_min > score_max:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    "score_min must be less than or equal to score_max",
                )
            )
    except (TypeError, ValueError):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "score clip bounds must be finite numbers")
        )
        score_min, score_max = Fraction(-10), Fraction(10)

    places = _positive_int(
        _get_values(raw, ("fixed_decimal_places", "fixed_point_places", "fixed_width", "decimal_places")),
        "fixed_decimal_places",
        failures,
        E_INPUT_INVALID,
    )
    quant_step = _positive_int(
        _get_values(raw, ("quantization_step", "quantization_level", "quant_level")),
        "quantization_step",
        failures,
        E_INPUT_INVALID,
    )
    window = _positive_int(
        _get_values(raw, ("rerank_window_size", "rerank_window", "window_size")),
        "rerank_window_size",
        failures,
        E_INPUT_INVALID,
    )
    penalty_raw = _get_values(
        raw,
        ("diversity_penalty", "diversity_penalty_coefficient", "penalty"),
    )
    try:
        diversity_penalty = _fraction(penalty_raw)
        if diversity_penalty < 0:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                "diversity_penalty must be a non-negative finite number",
            )
        )
        diversity_penalty = Fraction(0)

    concurrency = _positive_int(
        _get_values(raw, ("concurrency", "thread_pool_concurrency", "threads", "concurrency_limit")),
        "concurrency",
        failures,
        E_BUDGET_INVALID,
    )
    if concurrency is not None and concurrency > MAX_CONCURRENCY:
        failures.append(
            Failure(
                E_BUDGET_INVALID,
                "validation",
                f"concurrency {concurrency} exceeds upper bound {MAX_CONCURRENCY}",
            )
        )
    global_budget = _positive_int(
        _get_values(raw, ("global_mac_budget", "global_budget", "mac_budget")),
        "global_mac_budget",
        failures,
        E_BUDGET_INVALID,
    )
    request_quota = _positive_int(
        _get_values(raw, ("per_request_quota", "request_quota", "per_request_mac_quota")),
        "per_request_quota",
        failures,
        E_BUDGET_INVALID,
    )
    cache_budget = _positive_int(
        _get_values(raw, ("cache_residency_budget", "cache_budget", "cache_blocks")),
        "cache_residency_budget",
        failures,
        E_BUDGET_INVALID,
    )
    snapshot_interval = _positive_int(
        _get_values(
            raw,
            ("snapshot_commit_interval_commands", "snapshot_interval", "commit_interval_commands"),
        ),
        "snapshot_commit_interval_commands",
        failures,
        E_BUDGET_INVALID,
    )
    min_batch = _positive_int(
        _get_values(raw, ("min_command_batch", "minimum_command_batch"), 1),
        "min_command_batch",
        failures,
        E_BUDGET_INVALID,
    )
    latency_raw = _get_values(
        raw,
        ("interaction_latency_budget_ms", "latency_budget_ms", "interactive_budget_ms"),
        Decimal("50"),
    )
    try:
        latency_budget = _fraction(latency_raw)
        if latency_budget <= 0:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(
                E_BUDGET_INVALID,
                "validation",
                "interaction_latency_budget_ms must be positive and finite",
            )
        )
        latency_budget = Fraction(50)
    latency_rate_raw = _get_values(
        raw,
        ("latency_units_per_ms", "logical_units_per_ms"),
        Decimal("1000"),
    )
    try:
        latency_rate = _fraction(latency_rate_raw)
        if latency_rate <= 0:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(
                E_BUDGET_INVALID,
                "validation",
                "latency_units_per_ms must be positive and finite",
            )
        )
        latency_rate = Fraction(1000)

    threshold_raw = _get_values(
        raw,
        ("similarity_threshold", "diversity_threshold"),
        catalog.similarity_threshold,
    )
    try:
        threshold = _fraction(threshold_raw)
        if threshold < 0 or threshold > 1:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                "similarity_threshold must be between 0 and 1",
            )
        )
        threshold = Fraction(8, 10)

    if len(failures) != before:
        return None
    assert weights is not None
    assert places is not None
    assert quant_step is not None
    assert window is not None
    assert concurrency is not None
    assert global_budget is not None
    assert request_quota is not None
    assert cache_budget is not None
    assert snapshot_interval is not None
    assert min_batch is not None
    if len(weights) != catalog.dimension:
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                f"feature weight length {len(weights)} does not match feature dimension {catalog.dimension}",
            )
        )
        return None
    if round_half_away(temperature * (10**places)) == 0:
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                "fusion_temperature quantises to zero at fixed_decimal_places",
            )
        )
        return None

    known_keys = {
        "feature_weights",
        "weights",
        "weight_vector",
        "normalization",
        "normalizer",
        "normalize",
        "fusion_temperature",
        "temperature",
        "fusion_temp",
        "score_clip_interval",
        "score_interval",
        "clamping_interval",
        "score_min",
        "min_score",
        "score_lower_bound",
        "score_max",
        "max_score",
        "score_upper_bound",
        "fixed_decimal_places",
        "fixed_point_places",
        "fixed_width",
        "decimal_places",
        "quantization_step",
        "quantization_level",
        "quant_level",
        "rerank_window_size",
        "rerank_window",
        "window_size",
        "diversity_penalty",
        "diversity_penalty_coefficient",
        "penalty",
        "concurrency",
        "thread_pool_concurrency",
        "threads",
        "concurrency_limit",
        "global_mac_budget",
        "global_budget",
        "mac_budget",
        "per_request_quota",
        "request_quota",
        "per_request_mac_quota",
        "cache_residency_budget",
        "cache_budget",
        "cache_blocks",
        "snapshot_commit_interval_commands",
        "snapshot_interval",
        "commit_interval_commands",
        "interaction_latency_budget_ms",
        "latency_budget_ms",
        "interactive_budget_ms",
        "min_command_batch",
        "minimum_command_batch",
        "latency_units_per_ms",
        "logical_units_per_ms",
        "similarity_threshold",
        "diversity_threshold",
    }
    extra = {
        str(key): value
        for key, value in sorted(raw.items())
        if key not in known_keys
    }
    return LabConfig(
        feature_weights=weights,
        normalization=normalization,
        fusion_temperature=temperature,
        score_min=score_min,
        score_max=score_max,
        fixed_decimal_places=places,
        quantization_step=quant_step,
        rerank_window_size=window,
        diversity_penalty=diversity_penalty,
        concurrency=concurrency,
        global_mac_budget=global_budget,
        per_request_quota=request_quota,
        cache_residency_budget=cache_budget,
        snapshot_commit_interval_commands=snapshot_interval,
        interaction_latency_budget_ms=latency_budget,
        min_command_batch=min_batch,
        latency_units_per_ms=latency_rate,
        similarity_threshold=threshold,
        raw_extra=extra,
    )


def _load_commands(path: Path | None, failures: list[Failure]) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.suffix.lower() in (".yaml", ".yml"):
        data = _load_yaml(path)
        if isinstance(data, dict):
            data = _get_values(data, ("commands", "stream"), [])
        if data is None:
            return []
        if not isinstance(data, list):
            failures.append(
                Failure(E_INPUT_INVALID, "validation", "command stream must be an array")
            )
            return []
        return [item for item in data if isinstance(item, dict)]

    text = _read_text(path)
    if path.suffix.lower() == ".json":
        data = json.loads(
            text,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=parse_json_constant,
        )
        if isinstance(data, dict):
            data = _get_values(data, ("commands", "stream"), [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "command stream must be an array")
        )
        return []

    commands: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_float=Decimal,
                parse_int=int,
                parse_constant=parse_json_constant,
            )
        except (json.JSONDecodeError, InvalidOperation) as exc:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"invalid command JSONL: {exc}",
                    line=line_no,
                )
            )
            continue
        if not isinstance(value, dict):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    "each command must be a JSON object",
                    line=line_no,
                )
            )
            continue
        value = dict(value)
        value.setdefault("_line", line_no)
        commands.append(value)
    return commands


def _normalise_fault(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "nonfinite": "non_finite_feature",
        "non_finite": "non_finite_feature",
        "non_finite_dense_feature": "non_finite_feature",
        "nan_feature": "non_finite_feature",
        "inf_feature": "non_finite_feature",
        "weight_dimension_mismatch": "weight_length_mismatch",
        "weights_length_mismatch": "weight_length_mismatch",
        "quota_below_min_mac": "quota_below_minimum",
        "quota_below_minimum_mac": "quota_below_minimum",
        "quota_too_small": "quota_below_minimum",
    }
    return aliases.get(text, text)


def _load_injections(path: Path | None) -> tuple[list[dict[str, Any]], list[Failure]]:
    failures: list[Failure] = []
    if path is None:
        return [], failures
    try:
        data = _load_structured(path)
    except Exception as exc:
        failures.append(
            Failure(E_INPUT_INVALID, "validation", f"cannot parse injection manifest: {exc}")
        )
        return [], failures
    if data is None:
        return [], failures
    if isinstance(data, dict):
        data = _get_values(data, ("injections", "faults", "items"), [])
    if not isinstance(data, list):
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "injection manifest must be an array")
        )
        return [], failures
    result: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            failures.append(
                Failure(E_INPUT_INVALID, "validation", f"injection[{index}] must be an object")
            )
            continue
        injection_id = _get_values(item, ("id", "injection_id", "fault_id"))
        request_id = _get_values(item, ("request_id", "requestId", "query_id"))
        stage = _get_values(item, ("stage", "target_stage"), "")
        fault = _normalise_fault(_get_values(item, ("fault", "kind", "type")))
        if not isinstance(injection_id, str) or not injection_id:
            failures.append(
                Failure(E_INPUT_INVALID, "validation", f"injection[{index}] id is invalid")
            )
            continue
        if not isinstance(request_id, str) or not request_id:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"injection {injection_id!r} request_id is invalid",
                )
            )
            continue
        if fault not in (
            "non_finite_feature",
            "weight_length_mismatch",
            "quota_below_minimum",
        ):
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "validation",
                    f"injection {injection_id!r} has unsupported fault {fault!r}",
                )
            )
            continue
        if stage not in ("load", "validation", "normalize", "fusion", "rerank", "budget"):
            stage = {
                "non_finite_feature": "load",
                "weight_length_mismatch": "fusion",
                "quota_below_minimum": "budget",
            }[fault]
        result.append(
            {
                "injection_id": injection_id,
                "request_id": request_id,
                "stage": str(stage),
                "fault": fault,
            }
        )
    return result, failures


def _attach_catalog_and_dedupe(
    candidates: list[RawCandidate],
    catalog: FeatureCatalog,
    failures: list[Failure],
) -> list[RawCandidate]:
    valid: list[RawCandidate] = []
    seen: dict[tuple[str, str], RawCandidate] = {}
    declared = set(catalog.feature_order)
    for candidate in candidates:
        unknown = sorted(set(candidate.features) - declared)
        if unknown:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    f"candidate uses undeclared features: {unknown}",
                    request_id=candidate.request_id,
                    item_id=candidate.item_id,
                    line=candidate.line,
                )
            )
            continue
        key = (candidate.request_id, candidate.item_id)
        if key in seen:
            failures.append(
                Failure(
                    E_INPUT_INVALID,
                    "load",
                    "duplicate candidate item_id within request; first occurrence kept",
                    request_id=candidate.request_id,
                    item_id=candidate.item_id,
                    line=candidate.line,
                )
            )
            continue
        seen[key] = candidate
        valid.append(candidate)
    return sorted(valid, key=lambda c: (c.request_id, c.item_id, c.line))


def load_sample_dir(
    sample_dir: str | Path,
    injections_path: str | Path | None = None,
) -> LoadedSample:
    """Load, validate and deterministically sort all sample inputs."""

    root = Path(sample_dir)
    failures: list[Failure] = []

    candidates_path = _first_path(root, ("candidates.jsonl", "candidates.json"))
    config_path = _first_path(root, ("config.json", "rerank_config.json", "config.yaml", "config.yml"))
    groups_path = _first_path(root, ("feature_groups.json", "feature_groups.yaml", "feature_groups.yml", "groups.json"))
    commands_path = _first_path(root, ("commands.jsonl", "commands.json", "commands.yaml", "commands.yml"))
    if candidates_path is None:
        raise FileNotFoundError(f"no candidates.jsonl/candidates.json in {root}")
    if config_path is None:
        raise FileNotFoundError(f"no config.json in {root}")

    raw_config = _load_structured(config_path)
    raw_groups: dict[str, Any] = {}
    if groups_path is not None:
        loaded_groups = _load_structured(groups_path)
        if isinstance(loaded_groups, list):
            raw_groups = {"groups": loaded_groups}
        elif isinstance(loaded_groups, dict):
            raw_groups = loaded_groups
        else:
            failures.append(
                Failure(E_INPUT_INVALID, "validation", "feature group table must be an object")
            )
    else:
        failures.append(
            Failure(E_INPUT_INVALID, "validation", "feature group table is required")
        )

    candidates = _load_candidates(candidates_path, failures)
    declared_order, groups, similarity_features = _parse_groups(raw_groups, failures)
    all_candidate_features = {name for candidate in candidates for name in candidate.features}

    threshold_raw = _get_values(
        raw_groups,
        ("similarity_threshold", "diversity_threshold"),
        Decimal("0.8"),
    )
    try:
        threshold = _fraction(threshold_raw)
        if threshold < 0 or threshold > 1:
            raise ValueError
    except (TypeError, ValueError):
        failures.append(
            Failure(
                E_INPUT_INVALID,
                "validation",
                "group similarity_threshold must be between 0 and 1",
            )
        )
        threshold = Fraction(8, 10)

    catalog = _catalog_from(
        declared_order,
        groups,
        similarity_features,
        all_candidate_features,
        threshold,
        raw_groups,
        failures,
    )
    candidates = _attach_catalog_and_dedupe(candidates, catalog, failures)
    config = _parse_config_raw(raw_config, catalog, failures)
    commands = _load_commands(commands_path, failures)
    injection_source = Path(injections_path) if injections_path is not None else None
    if injection_source is None:
        injection_source = _first_path(
            root, ("injections.json", "injections.yaml", "injections.yml")
        )
    injections, injection_failures = _load_injections(injection_source)
    failures.extend(injection_failures)

    input_files: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for label, path in (
        ("candidates", candidates_path),
        ("config", config_path),
        ("feature_groups", groups_path),
        ("commands", commands_path),
        ("injections", injection_source),
    ):
        if path is not None:
            input_files[label] = path.name
            input_hashes[label] = digest(path.read_bytes().decode("utf-8-sig"))

    return LoadedSample(
        candidates=candidates,
        config=config,
        catalog=catalog,
        commands=commands,
        injections=injections,
        failures=sorted_failures(failures),
        config_valid=config is not None,
        raw_config=raw_config if isinstance(raw_config, dict) else {},
        raw_groups=raw_groups,
        input_files=input_files,
        input_hashes=input_hashes,
    )
