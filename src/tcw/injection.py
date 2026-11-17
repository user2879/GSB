"""Independent, reproducible fault-injection probes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import Candidate, METHOD_ALIASES, validate_candidate_params
from .data import PreparedData, SyntheticDataset, assert_finite_raw_values, encode_raw_category
from .errors import (
    E_CATEGORY_UNMAPPED,
    E_CONFIG_INVALID,
    E_DBSCAN_RADIUS_INVALID,
    E_HYPERPARAM_OUT_OF_RANGE,
    E_INJECTION_NOT_TRIGGERED,
    E_INPUT_NON_FINITE,
    BusinessError,
    Failure,
)
from .evaluation import assert_finite_matrix
from .util import canonical_json, sha256_value


FAULT_CODES = {
    "non_finite_feature": E_INPUT_NON_FINITE,
    "missing_category_mapping": E_CATEGORY_UNMAPPED,
    "hyperparameter_out_of_range": E_HYPERPARAM_OUT_OF_RANGE,
    "negative_neighborhood_radius": E_DBSCAN_RADIUS_INVALID,
}
STAGE_ALIASES = {
    "load": "data",
    "data": "data",
    "ingest": "data",
    "preprocess": "preprocess",
    "transform": "preprocess",
    "clustering": "clustering",
    "cluster": "clustering",
    "model": "clustering",
    "fit": "clustering",
    "evaluation": "evaluation",
    "evaluate": "evaluation",
    "metrics": "evaluation",
}
STAGE_ORDER = {"data": 0, "preprocess": 1, "clustering": 2, "evaluation": 3}


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    stage: str
    fault: str
    expected_code: str
    sample: int | None = None
    column: str | None = None
    value: Any = None
    method: str | None = None
    candidate_params: dict[str, Any] = field(default_factory=dict)
    candidate_sort_key: str | None = None
    parameter: str | None = None


@dataclass
class InjectionConfig:
    path: Path | None
    raw: dict[str, Any]
    hash: str
    specs: list[InjectionSpec]
    failures: list[Failure]


@dataclass
class InjectionContext:
    dataset: SyntheticDataset
    prepared: PreparedData
    matrix: np.ndarray
    candidates: list[Candidate]


def _empty_hash() -> str:
    return sha256_value({"injections": []})


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BusinessError(
            E_CONFIG_INVALID,
            f"injection config does not exist: {path}",
            stage="config",
            path=str(path),
            fatal=True,
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"cannot parse injection config: {exc}",
            stage="config",
            path=str(path),
            fatal=True,
        ) from exc
    if raw is None:
        raw = {"injections": []}
    if not isinstance(raw, dict):
        raise BusinessError(
            E_CONFIG_INVALID,
            "injection config root must be a mapping",
            stage="config",
            path=str(path),
            fatal=True,
        )
    return raw


def load_injection_config(path: Path | None) -> InjectionConfig:
    if path is None:
        return InjectionConfig(path=None, raw={"injections": []}, hash=_empty_hash(), specs=[], failures=[])
    resolved = Path(path).resolve()
    raw = _load_yaml(resolved)
    failures: list[Failure] = []
    injections_raw = raw.get("injections", [])
    if not isinstance(injections_raw, list):
        raise BusinessError(
            E_CONFIG_INVALID,
            "injection config field 'injections' must be a list",
            stage="config",
            path=str(resolved),
            fatal=True,
        )
    specs: list[InjectionSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(injections_raw):
        location = f"injections[{index}]"
        if not isinstance(item, dict):
            failures.append(Failure(E_CONFIG_INVALID, "injection must be a mapping", "config", path=location))
            continue
        injection_id = item.get("id")
        if not isinstance(injection_id, str) or not injection_id.strip():
            failures.append(
                Failure(E_CONFIG_INVALID, "injection id must be a non-empty string", "config", path=location)
            )
            continue
        injection_id = injection_id.strip()
        if injection_id in seen_ids:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"duplicate injection id: {injection_id}",
                    "config",
                    injection_id=injection_id,
                    path=location,
                )
            )
            continue
        seen_ids.add(injection_id)
        stage_raw = item.get("stage")
        stage = STAGE_ALIASES.get(str(stage_raw).strip().lower()) if isinstance(stage_raw, str) else None
        if stage is None:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"invalid injection stage: {stage_raw!r}",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.stage",
                )
            )
            continue
        fault = item.get("fault")
        if fault not in FAULT_CODES:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"invalid injection fault: {fault!r}",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.fault",
                )
            )
            continue
        expected_code = FAULT_CODES[fault]
        configured_code = item.get("code", expected_code)
        if configured_code != expected_code:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"injection code {configured_code!r} does not match fault mapping {expected_code}",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.code",
                )
            )
            continue
        sample = item.get("sample")
        if sample is not None and (
            not isinstance(sample, int) or isinstance(sample, bool) or sample < 0
        ):
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "injection sample must be a non-negative integer",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.sample",
                )
            )
            continue
        if fault in {"non_finite_feature", "missing_category_mapping"} and sample is None:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"fault {fault} requires a sample",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.sample",
                )
            )
            continue
        method_raw = item.get("method")
        method = METHOD_ALIASES.get(str(method_raw).strip().lower()) if isinstance(method_raw, str) else None
        if fault in {"hyperparameter_out_of_range", "negative_neighborhood_radius"}:
            if method is None:
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"fault {fault} requires a valid method",
                        "config",
                        injection_id=injection_id,
                        path=f"{location}.method",
                    )
                )
                continue
            parameter = item.get("parameter")
            if not isinstance(parameter, str) or not parameter:
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"fault {fault} requires a parameter name",
                        "config",
                        injection_id=injection_id,
                        path=f"{location}.parameter",
                    )
                )
                continue
            if "value" not in item:
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"fault {fault} requires a value",
                        "config",
                        injection_id=injection_id,
                        path=f"{location}.value",
                    )
                )
                continue
        else:
            parameter = item.get("parameter") if isinstance(item.get("parameter"), str) else None
        candidate_params = item.get("candidate_params", {})
        if not isinstance(candidate_params, dict):
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "injection candidate_params must be a mapping",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.candidate_params",
                )
            )
            continue
        candidate_sort_key = item.get("candidate_sort_key")
        if candidate_sort_key is not None and not isinstance(candidate_sort_key, str):
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "candidate_sort_key must be a string",
                    "config",
                    injection_id=injection_id,
                    path=f"{location}.candidate_sort_key",
                )
            )
            continue
        specs.append(
            InjectionSpec(
                injection_id=injection_id,
                stage=stage,
                fault=fault,
                expected_code=expected_code,
                sample=sample,
                column=item.get("column") if isinstance(item.get("column"), str) else None,
                value=item.get("value"),
                method=method,
                candidate_params=dict(candidate_params),
                candidate_sort_key=candidate_sort_key,
                parameter=parameter,
            )
        )
    specs.sort(key=lambda spec: (STAGE_ORDER[spec.stage], spec.injection_id))
    return InjectionConfig(
        path=resolved,
        raw=raw,
        hash=sha256_value(raw),
        specs=specs,
        failures=failures,
    )


def _numeric_probe_value(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() in {"nan", "+nan", "-nan"}:
        return math.nan
    if isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}:
        return -math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return math.nan


def _set_raw_value(dataset: SyntheticDataset, sample: int, column: str, value: Any) -> SyntheticDataset:
    if sample < 0 or sample >= dataset.rows:
        raise BusinessError(
            "E_INPUT_INVALID",
            f"sample {sample} is outside [0, {dataset.rows})",
            stage="injection",
            extra={"sample": sample},
        )
    if column not in dataset.values:
        raise BusinessError(
            "E_INPUT_INVALID",
            f"unknown injection column: {column}",
            stage="injection",
            extra={"column": column},
        )
    values = {name: list(items) for name, items in dataset.values.items()}
    values[column][sample] = value
    return replace(dataset, values=values)


def _find_candidate(context: InjectionContext, spec: InjectionSpec) -> Candidate | None:
    for candidate in context.candidates:
        if spec.candidate_sort_key is not None and candidate.sort_key != spec.candidate_sort_key:
            continue
        if spec.method is not None and candidate.method != spec.method:
            continue
        if any(candidate.params.get(key) != value for key, value in spec.candidate_params.items()):
            continue
        return candidate
    return None


def _probe(spec: InjectionSpec, context: InjectionContext) -> None:
    if spec.fault == "non_finite_feature":
        if spec.column is None:
            raise BusinessError(
                "E_INPUT_INVALID",
                "non_finite_feature requires a column",
                stage="injection",
                injection_id=spec.injection_id,
            )
        injected = _set_raw_value(
            context.dataset,
            int(spec.sample),
            spec.column,
            _numeric_probe_value(spec.value),
        )
        if spec.stage in {"data", "preprocess"}:
            probe_value = injected.values[spec.column][int(spec.sample)]
            if not math.isfinite(float(probe_value)):
                raise BusinessError(
                    E_INPUT_NON_FINITE,
                    f"injected non-finite value at sample {spec.sample}, column {spec.column}: {probe_value!r}",
                    stage=spec.stage,
                    path=f"sample[{spec.sample}].{spec.column}",
                    extra={"sample": spec.sample, "column": spec.column},
                )
            assert_finite_raw_values(injected, stage=spec.stage)
            return
        matrix = np.asarray(context.matrix, dtype=float).copy()
        matrix[int(spec.sample), 0] = math.nan
        assert_finite_matrix(matrix, stage=spec.stage)
        return

    if spec.fault == "missing_category_mapping":
        if spec.column is None:
            raise BusinessError(
                "E_INPUT_INVALID",
                "missing_category_mapping requires a column",
                stage="injection",
                injection_id=spec.injection_id,
            )
        if spec.sample is not None and not (0 <= int(spec.sample) < len(context.prepared.matrix)):
            raise BusinessError(
                "E_INPUT_INVALID",
                f"sample {spec.sample} is outside prepared rows",
                stage="injection",
                injection_id=spec.injection_id,
            )
        injected_value = spec.value if isinstance(spec.value, str) else "__INJECTED_UNSEEN_CATEGORY__"
        encode_raw_category(context.prepared, spec.column, injected_value)
        return

    if spec.fault in {"hyperparameter_out_of_range", "negative_neighborhood_radius"}:
        candidate = _find_candidate(context, spec)
        if candidate is None:
            raise BusinessError(
                "E_INPUT_INVALID",
                "no candidate matched the injection selector",
                stage="injection",
                injection_id=spec.injection_id,
            )
        modified = dict(candidate.params)
        modified[str(spec.parameter)] = spec.value
        if spec.fault == "negative_neighborhood_radius":
            if candidate.method != "dbscan":
                raise BusinessError(
                    "E_INPUT_INVALID",
                    "negative_neighborhood_radius requires a DBSCAN candidate",
                    stage="injection",
                    injection_id=spec.injection_id,
                )
            if float(spec.value) >= 0:
                modified[str(spec.parameter)] = -abs(float(spec.value) or 1.0)
        validate_candidate_params(
            candidate.method,
            modified,
            len(context.prepared.matrix),
            error_code=spec.expected_code,
        )
        return

    raise BusinessError(
        "E_INPUT_INVALID",
        f"unsupported fault: {spec.fault}",
        stage="injection",
        injection_id=spec.injection_id,
    )


def run_injections(
    config: InjectionConfig,
    context: InjectionContext,
) -> tuple[list[dict[str, Any]], list[Failure]]:
    hits: list[dict[str, Any]] = []
    failures: list[Failure] = list(config.failures)
    for spec in config.specs:
        candidate = _find_candidate(context, spec) if spec.stage == "clustering" else None
        hit: dict[str, Any] = {
            "injection_id": spec.injection_id,
            "stage": spec.stage,
            "fault": spec.fault,
            "sample": spec.sample,
            "column": spec.column,
            "method": spec.method,
            "parameter": spec.parameter,
            "candidate_key": candidate.sort_key if candidate else spec.candidate_sort_key,
            "expected_code": spec.expected_code,
            "actual_code": None,
            "hit": False,
            "detail": "",
            "selector": {
                "candidate_params": spec.candidate_params,
                "candidate_sort_key": spec.candidate_sort_key,
            },
        }
        try:
            _probe(spec, context)
        except BusinessError as exc:
            hit["actual_code"] = exc.code
            hit["detail"] = exc.failure.message
            hit["hit"] = exc.code == spec.expected_code
            failure_code = exc.code if hit["hit"] else E_INJECTION_NOT_TRIGGERED
            failures.append(
                Failure(
                    code=failure_code,
                    message=exc.failure.message,
                    stage=spec.stage,
                    injection_id=spec.injection_id,
                    candidate_key=candidate.sort_key if candidate else spec.candidate_sort_key,
                    path=exc.failure.path,
                    extra={
                        "fault": spec.fault,
                        "expected_code": spec.expected_code,
                        "actual_code": exc.code,
                        "injection_hit": hit["hit"],
                    },
                )
            )
        except Exception as exc:
            hit["actual_code"] = "E_INTERNAL"
            hit["detail"] = f"{type(exc).__name__}: {exc}"
            failures.append(
                Failure(
                    code=E_INJECTION_NOT_TRIGGERED,
                    message=hit["detail"],
                    stage=spec.stage,
                    injection_id=spec.injection_id,
                    candidate_key=candidate.sort_key if candidate else spec.candidate_sort_key,
                    extra={"fault": spec.fault, "expected_code": spec.expected_code},
                )
            )
        else:
            hit["detail"] = "injection did not raise a business error"
            failures.append(
                Failure(
                    code=E_INJECTION_NOT_TRIGGERED,
                    message=hit["detail"],
                    stage=spec.stage,
                    injection_id=spec.injection_id,
                    candidate_key=candidate.sort_key if candidate else spec.candidate_sort_key,
                    extra={"fault": spec.fault, "expected_code": spec.expected_code},
                )
            )
        hits.append(hit)
    hits.sort(key=lambda item: (STAGE_ORDER.get(item["stage"], 99), item["injection_id"]))
    return hits, failures
