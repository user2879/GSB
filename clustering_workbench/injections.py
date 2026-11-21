"""Fault-injection bookkeeping and deterministic matching rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .codes import Failure
from .utils import canonical_json, hash_object, json_safe


INJECTION_CODES = {
    "non_finite_feature": "INJ_NON_FINITE_FEATURE",
    "hyperparameter_out_of_range": "INJ_HYPERPARAMETER_OUT_OF_RANGE",
    "threshold_boundary_equal": "INJ_THRESHOLD_BOUNDARY_EQUAL",
    "data_size_over_limit": "INJ_DATA_SIZE_OVER_LIMIT",
}


@dataclass
class InjectionReport:
    injection_id: str
    injection_type: str
    stage: str
    business_code: str
    target: Any = None
    hit: bool = False
    isolated: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.injection_id,
                "type": self.injection_type,
                "stage": self.stage,
                "business_code": self.business_code,
                "target": self.target,
                "hit": self.hit,
                "isolated": self.isolated,
                "evidence": self.evidence,
            }
        )


class InjectionManager:
    def __init__(self, items: list[dict[str, Any]] | None = None):
        self.items = [item for item in (items or []) if isinstance(item, dict)]
        self.reports: dict[str, InjectionReport] = {}
        for index, item in enumerate(self.items):
            injection_id = str(item.get("id") or f"INJ-{index + 1:03d}")
            injection_type = str(item.get("type", "unknown"))
            report = InjectionReport(
                injection_id=injection_id,
                injection_type=injection_type,
                stage=str(item.get("stage", "")),
                business_code=INJECTION_CODES.get(injection_type, "CFG_PARAMETER_OUT_OF_RANGE"),
                target=item.get("target", item.get("sample_id")),
            )
            self.reports[injection_id] = report

    @property
    def fingerprint(self) -> str:
        return hash_object("inj", self.items)

    def enabled_items(self, injection_type: str, stage: str | None = None) -> list[dict[str, Any]]:
        found = []
        for item in self.items:
            if item.get("type") != injection_type:
                continue
            if stage is not None and item.get("stage") not in {stage, "all", None}:
                continue
            found.append(item)
        return found

    def record_hit(
        self,
        injection_id: str,
        *,
        evidence: dict[str, Any],
        isolated: bool = True,
    ) -> Failure | None:
        report = self.reports.get(injection_id)
        if report is None:
            return None
        report.hit = True
        report.isolated = isolated
        report.evidence = json_safe(evidence)
        return Failure(
            code=report.business_code,
            stage=report.stage or "runtime",
            item=report.injection_id,
            message=f"Injection {report.injection_id} ({report.injection_type}) was exercised.",
            details=evidence,
        )

    def cap_rows(self, requested_rows: int, max_rows: int) -> tuple[int, list[Failure]]:
        failures: list[Failure] = []
        actual_rows = int(requested_rows)
        for item in self.enabled_items("data_size_over_limit", "load"):
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            injection_limit = int(target.get("max_rows", item.get("max_rows", max_rows)) or max_rows)
            forced_rows = int(item.get("requested_rows", injection_limit + 1) or injection_limit + 1)
            desired_rows = max(actual_rows, forced_rows)
            if desired_rows > injection_limit:
                actual_rows = injection_limit
                failure = self.record_hit(
                    str(item.get("id")),
                    evidence={
                        "requested_rows": desired_rows,
                        "max_rows": injection_limit,
                        "accepted_rows": actual_rows,
                        "action": "cap_and_continue",
                    },
                )
                if failure:
                    failures.append(failure)
        if actual_rows > max_rows:
            actual_rows = max_rows
        return max(1, actual_rows), failures

    def apply_non_finite(self, columns: dict[str, list[Any]], sample_ids: list[str]) -> list[Failure]:
        failures: list[Failure] = []
        sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        for item in self.enabled_items("non_finite_feature", "preprocess"):
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            sample_id = str(target.get("sample_id", item.get("sample_id", "")))
            column = str(target.get("column", target.get("feature", item.get("column", ""))))
            value = target.get("value", item.get("value", float("inf")))
            if sample_id not in sample_index or column not in columns:
                continue
            row = sample_index[sample_id]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float("inf")
            if not np.isfinite(numeric_value):
                numeric_value = float("inf")
            columns[column][row] = numeric_value
            failure = self.record_hit(
                str(item.get("id")),
                evidence={
                    "sample_id": sample_id,
                    "column": column,
                    "injected_value": "inf" if np.isinf(numeric_value) else "nan",
                    "row_index": row,
                },
            )
            if failure:
                failures.append(failure)
        return failures

    def candidate_matches(self, item: dict[str, Any], method: str, index: int, params: dict[str, Any]) -> bool:
        target = item.get("target") if isinstance(item.get("target"), dict) else {}
        target_method = target.get("method", item.get("method"))
        if target_method is not None and str(target_method) != method:
            return False
        target_index = target.get("candidate_index", item.get("candidate_index"))
        if target_index is not None and int(target_index) != index:
            return False
        match_params = target.get("params_match", item.get("params_match", {})) or {}
        for key, value in match_params.items():
            if params.get(key) != value:
                return False
        return True

    def inject_candidate(
        self,
        method: str,
        index: int,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
        effective = dict(params)
        hits: list[tuple[str, dict[str, Any]]] = []
        for item in self.enabled_items("hyperparameter_out_of_range", "cluster"):
            if not self.candidate_matches(item, method, index, params):
                continue
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            parameter = str(target.get("parameter", item.get("parameter", "")))
            injected_value = target.get("value", item.get("value", 0))
            if parameter:
                effective[parameter] = injected_value
                hits.append(
                    (
                        str(item.get("id")),
                        {
                            "method": method,
                            "candidate_index": index,
                            "parameter": parameter,
                            "injected_value": injected_value,
                            "original_value": params.get(parameter),
                        },
                    )
                )
        return effective, hits

    def boundary_targets(self) -> list[dict[str, Any]]:
        return self.enabled_items("threshold_boundary_equal", "evaluation")

    def boundary_matches(self, item: dict[str, Any], method: str, index: int, params: dict[str, Any]) -> bool:
        return self.candidate_matches(item, method, index, params)

    def summary(self) -> dict[str, Any]:
        reports = [self.reports[key].to_dict() for key in sorted(self.reports)]
        return {
            "fingerprint": self.fingerprint,
            "total": len(reports),
            "hit": sum(1 for item in reports if item["hit"]),
            "missed": sum(1 for item in reports if not item["hit"]),
            "items": reports,
            "canonical": canonical_json(json_safe(self.items)),
        }

    def reports_list(self) -> list[InjectionReport]:
        return [self.reports[key] for key in sorted(self.reports)]
