"""Fault injection, isolated to individual sample/stage diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import yamlite
from .config import ALGORITHM_NAMES, DEFAULT_STORAGE
from .errors import (
    E_CONFIG_INVALID,
    E_DATA_LIMIT,
    E_NON_FINITE_FEATURE,
    E_PARAM_INVALID,
    E_THRESHOLD_BOUNDARY,
    ConfigValidationError,
    Failure,
    config_failure,
    sort_failures,
)
from .preprocess import PreprocessedData


INJECTION_KINDS = (
    "data_size_limit",
    "hyperparameter_out_of_bounds",
    "non_finite_feature",
    "threshold_boundary",
)
DEFAULT_CODES = {
    "data_size_limit": E_DATA_LIMIT,
    "hyperparameter_out_of_bounds": E_PARAM_INVALID,
    "non_finite_feature": E_NON_FINITE_FEATURE,
    "threshold_boundary": E_THRESHOLD_BOUNDARY,
}


@dataclass
class InjectionBundle:
    path: Path
    root: Path
    data: dict[str, Any]
    failures: list[dict[str, Any]]

    @property
    def valid(self) -> bool:
        return not self.failures


def _parse_injected_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.lower() in ("inf", "+inf", "infinity", "+infinity"):
        return math.inf
    if isinstance(value, str) and value.lower() in ("-inf", "-infinity"):
        return -math.inf
    if isinstance(value, str) and value.lower() in ("nan", ".nan"):
        return math.nan
    raise ValueError("injection value is not non-finite")


def _parameter_is_out_of_bounds(algorithm: str, parameter: str, value: Any) -> bool:
    if algorithm == "robust_mahalanobis" and parameter == "regularization":
        return not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0.0
    if algorithm == "histogram_density" and parameter in ("bins", "neighborhood"):
        if not isinstance(value, int) or isinstance(value, bool):
            return True
        return value < 2 if parameter == "bins" else value < 0
    if algorithm == "random_partition_depth" and parameter in ("trees", "sample_size", "max_depth"):
        return not isinstance(value, int) or isinstance(value, bool) or value < 1
    return True


def load_injection_bundle(path: str | Path, root: str | Path | None = None) -> InjectionBundle:
    injection_path = Path(path).resolve()
    injection_root = Path(root).resolve() if root is not None else injection_path.parent
    failures: list[Failure] = []
    try:
        raw = yamlite.load(injection_path)
    except FileNotFoundError:
        failures.append(config_failure("injection", f"file not found: {injection_path}"))
        raw = {}
    except (OSError, yamlite.YamlLiteError) as exc:
        failures.append(config_failure("injection", f"invalid YAML: {exc}"))
        raw = {}
    if not isinstance(raw, dict):
        failures.append(config_failure("injection", "injection root must be a mapping"))
        raw = {}
    if raw.get("version", 1) != 1:
        failures.append(config_failure("injection.version", "unsupported injection version"))
    items = raw.get("items")
    if not isinstance(items, list):
        failures.append(config_failure("injection.items", "items must be a list"))
        items = []
    seen_ids: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        field = f"injection.items[{index}]"
        if not isinstance(item, dict):
            failures.append(config_failure(field, "injection item must be a mapping"))
            continue
        item_id = item.get("id")
        kind = item.get("kind")
        if not isinstance(item_id, str) or not item_id:
            failures.append(config_failure(f"{field}.id", "injection id is required"))
            continue
        if item_id in seen_ids:
            failures.append(config_failure(f"{field}.id", f"duplicate injection id {item_id!r}"))
        seen_ids.add(item_id)
        if kind not in INJECTION_KINDS:
            failures.append(config_failure(f"{field}.kind", "unknown injection kind"))
            continue
        entry = dict(item)
        entry["business_code"] = str(item.get("business_code") or DEFAULT_CODES[kind])
        if kind in ("non_finite_feature", "hyperparameter_out_of_bounds", "threshold_boundary"):
            sample_id = item.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                failures.append(config_failure(f"{field}.sample_id", "sample_id is required"))
        if kind == "non_finite_feature":
            if not isinstance(item.get("column"), str) or not item.get("column"):
                failures.append(config_failure(f"{field}.column", "column is required"))
            try:
                entry["value"] = _parse_injected_float(item.get("value", "inf"))
            except ValueError as exc:
                failures.append(config_failure(f"{field}.value", str(exc)))
            entry.setdefault("stage", "preprocess")
        elif kind == "hyperparameter_out_of_bounds":
            algorithm = item.get("algorithm")
            parameter = item.get("parameter")
            if algorithm not in ALGORITHM_NAMES:
                failures.append(config_failure(f"{field}.algorithm", "unknown algorithm"))
            if not isinstance(parameter, str) or not parameter:
                failures.append(config_failure(f"{field}.parameter", "parameter is required"))
            if algorithm in ALGORITHM_NAMES and isinstance(parameter, str):
                if not _parameter_is_out_of_bounds(algorithm, parameter, item.get("value")):
                    failures.append(
                        config_failure(f"{field}.value", "injected hyperparameter must actually be out of bounds")
                    )
            entry.setdefault("stage", "score")
        elif kind == "threshold_boundary":
            entry.setdefault("stage", "decision")
            entry["at_boundary"] = True
        elif kind == "data_size_limit":
            maximum = item.get("max_samples")
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
                failures.append(config_failure(f"{field}.max_samples", "max_samples must be a positive integer"))
            entry.setdefault("stage", "load")
        normalized_items.append(entry)
    raw = dict(raw)
    raw["items"] = normalized_items
    report_value = raw.get("report_path", DEFAULT_STORAGE["injection_report_path"])
    raw["report_path"] = str(report_value)
    if not raw["report_path"]:
        failures.append(config_failure("injection.report_path", "report path must be non-empty"))
    return InjectionBundle(
        path=injection_path,
        root=injection_root,
        data=raw,
        failures=sort_failures(failures),
    )


def require_valid_injection(bundle: InjectionBundle) -> InjectionBundle:
    if bundle.failures:
        raise ConfigValidationError([Failure(**x) for x in bundle.failures])
    return bundle


class InjectionController:
    """Run-scoped injection overlay; base artifacts remain cacheable."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        dataset,
        config: dict[str, Any],
    ):
        self.items = sorted(items, key=lambda item: str(item["id"]))
        self.dataset = dataset
        self.config = config
        self.sample_index = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
        self.feature_index = {
            name: index for index, name in enumerate(config["data"]["feature_columns"])
        }
        self.selected = {
            name
            for name, item in config.get("algorithms", {}).items()
            if isinstance(item, dict) and item.get("enabled", True)
        }
        self.state: dict[str, dict[str, Any]] = {
            item["id"]: {
                "id": item["id"],
                "kind": item["kind"],
                "stage": item.get("stage"),
                "sample_id": item.get("sample_id"),
                "business_code": item["business_code"],
                "hit": False,
                "message": "not evaluated",
            }
            for item in self.items
        }
        self.failures: list[Failure] = []

    def _mark(self, item: dict[str, Any], hit: bool, message: str) -> None:
        state = self.state[item["id"]]
        state["hit"] = hit
        state["message"] = message

    def run_data_size_checks(self) -> None:
        n = len(self.dataset.sample_ids)
        for item in self.items:
            if item["kind"] != "data_size_limit":
                continue
            maximum = int(item["max_samples"])
            if n > maximum:
                self._mark(item, True, f"data size {n} exceeded injected limit {maximum}")
                self.failures.append(
                    Failure(
                        code=item["business_code"],
                        stage="load",
                        message=f"data size {n} exceeded limit {maximum}",
                        injection_id=item["id"],
                        details={"sample_count": n, "max_samples": maximum},
                    )
                )
            else:
                self._mark(item, False, f"data size {n} did not exceed limit {maximum}")

    def preprocess_failures(self, prepared: PreprocessedData) -> list[Failure]:
        out: list[Failure] = []
        for item in self.items:
            if item["kind"] != "non_finite_feature":
                continue
            sample_id = str(item["sample_id"])
            if sample_id not in self.sample_index:
                self._mark(item, False, "sample id not found")
                continue
            column = str(item["column"])
            if column not in self.feature_index:
                self._mark(item, False, "feature column not found")
                continue
            row_index = self.sample_index[sample_id]
            column_index = self.feature_index[column]
            probe = prepared.matrix[row_index].copy()
            probe[column_index] = float(item["value"])
            if np_all_finite(probe):
                self._mark(item, False, "injected value remained finite")
                continue
            self._mark(item, True, f"non-finite {item['value']!r} hit sample {sample_id} at preprocess")
            out.append(
                Failure(
                    code=item["business_code"],
                    stage="preprocess",
                    sample_id=sample_id,
                    exception_type="InjectedNonFiniteFeatureError",
                    injection_id=item["id"],
                    message=f"non-finite injected feature in column {column}",
                    details={"column": column, "value": str(item["value"])},
                )
            )
        self.failures.extend(out)
        return out

    def score_failures(self) -> list[Failure]:
        out: list[Failure] = []
        for item in self.items:
            if item["kind"] != "hyperparameter_out_of_bounds":
                continue
            sample_id = str(item["sample_id"])
            algorithm = str(item["algorithm"])
            if sample_id not in self.sample_index:
                self._mark(item, False, "sample id not found")
                continue
            if algorithm not in self.selected:
                self._mark(item, False, "algorithm is not enabled")
                continue
            self._mark(
                item,
                True,
                f"out-of-bounds {algorithm}.{item['parameter']}={item.get('value')!r} hit sample {sample_id}",
            )
            out.append(
                Failure(
                    code=item["business_code"],
                    stage="score",
                    sample_id=sample_id,
                    algorithm=algorithm,
                    exception_type="InjectedHyperparameterOutOfBoundsError",
                    injection_id=item["id"],
                    message=f"injected out-of-bounds hyperparameter {item['parameter']}={item.get('value')!r}",
                    details={"parameter": item["parameter"], "value": str(item.get("value"))},
                )
            )
        self.failures.extend(out)
        return out

    def decision_failures(self, sample_records: dict[str, dict[str, Any]], threshold: float) -> list[Failure]:
        out: list[Failure] = []
        for item in self.items:
            if item["kind"] != "threshold_boundary":
                continue
            sample_id = str(item["sample_id"])
            record = sample_records.get(sample_id)
            if sample_id not in self.sample_index or record is None:
                self._mark(item, False, "sample record not available")
                continue
            score = record.get("anomaly_score")
            if score is None or not math.isfinite(float(score)):
                self._mark(item, False, "sample has no finite anomaly score")
                continue
            boundary = float(score)
            effective_threshold = boundary
            if effective_threshold != boundary:
                self._mark(item, False, "boundary equality could not be established")
                continue
            self._mark(
                item,
                True,
                f"threshold {effective_threshold!r} equals sample boundary score for {sample_id}",
            )
            out.append(
                Failure(
                    code=item["business_code"],
                    stage="decision",
                    sample_id=sample_id,
                    exception_type="InjectedThresholdBoundaryError",
                    injection_id=item["id"],
                    message=f"injected threshold equals boundary anomaly score {boundary:.12f}",
                    details={"threshold": boundary, "configured_threshold": float(threshold)},
                )
            )
        self.failures.extend(out)
        return out

    def report(self) -> dict[str, Any]:
        return {
            "items": [dict(self.state[item["id"]]) for item in self.items],
            "hits": sum(1 for item in self.state.values() if item["hit"]),
            "total": len(self.items),
        }


def np_all_finite(values) -> bool:
    # Avoid importing NumPy solely for a tiny injection probe in callers.
    try:
        for value in values:
            if not math.isfinite(float(value)):
                return False
        return True
    except (TypeError, ValueError):
        return False
