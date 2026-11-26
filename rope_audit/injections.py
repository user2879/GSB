from __future__ import annotations

import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import AuditError, ConfigInvalid


KNOWN_FAULTS = {
    "negative_position",
    "nonfinite_feature",
    "wrong_shape",
    "zero_normalization_sum",
}


def load_injections(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw.get("injections", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ConfigInvalid("injections must be an array", field="injections")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ConfigInvalid("injection must be an object", field=f"injections[{index}]")
        injection_id = row.get("id")
        if not isinstance(injection_id, str) or not injection_id:
            raise ConfigInvalid("injection.id must be a non-empty string", field=f"injections[{index}].id")
        if injection_id in seen:
            raise ConfigInvalid(f"duplicate injection id {injection_id}", field=f"injections[{index}].id")
        seen.add(injection_id)
        fault_type = row.get("fault_type", "nonfinite_feature")
        if fault_type not in KNOWN_FAULTS:
            raise ConfigInvalid(
                f"fault_type must be one of {sorted(KNOWN_FAULTS)}",
                field=f"injections[{index}].fault_type",
            )
        stage = row.get("stage", "rotation")
        if stage not in {"position", "rotation", "dot", "normalization"}:
            raise ConfigInvalid(
                "stage must be one of position, rotation, dot, normalization",
                field=f"injections[{index}].stage",
            )
        normalized.append(
            {
                "id": injection_id,
                "sample_id": row.get("sample_id"),
                "scheme_id": row.get("scheme_id"),
                "stage": stage,
                "fault_type": fault_type,
                "code": row.get("code", "E_INJECTED_FAULT"),
                "message": row.get("message", "injected deterministic input fault"),
            }
        )
    return normalized


class InjectionHub:
    def __init__(self, injections: list[dict[str, Any]] | None = None) -> None:
        self.injections = injections or []
        self.hits: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def active_for(self, stage: str, sample_id: str, scheme_id: str | None = None) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for injection in self.injections:
            if injection["stage"] != stage:
                continue
            if injection.get("sample_id") not in (None, sample_id):
                continue
            if injection.get("scheme_id") not in (None, scheme_id):
                continue
            active.append(injection)
        return active

    def hit(self, injection: dict[str, Any]) -> None:
        record = {
            "id": injection["id"],
            "sample_id": injection.get("sample_id"),
            "scheme_id": injection.get("scheme_id"),
            "stage": injection["stage"],
            "fault_type": injection["fault_type"],
            "code": injection["code"],
        }
        with self._lock:
            self.hits.append(record)
        print(
            f"INJECTION_HIT id={injection['id']} sample={injection.get('sample_id')} "
            f"scheme={injection.get('scheme_id') or '*'} stage={injection['stage']} code={injection['code']}",
            file=sys.stderr,
            flush=True,
        )

    def corrupt_sample(self, stage: str, sample: dict[str, Any], scheme_id: str) -> dict[str, Any]:
        corrupted = deepcopy(sample)
        for injection in self.active_for(stage, sample["id"], scheme_id):
            self.hit(injection)
            fault_type = injection["fault_type"]
            if fault_type == "negative_position":
                corrupted["query_position"] = -1
            elif fault_type == "nonfinite_feature":
                corrupted["query_vector"][0] = float("nan")
            elif fault_type == "wrong_shape":
                corrupted["query_vector"] = corrupted["query_vector"][:-1]
            elif fault_type == "zero_normalization_sum":
                corrupted["_force_zero_norm_sum"] = True
        return corrupted

    def check_runtime_fault(self, stage: str, sample_id: str, scheme_id: str, **context: Any) -> None:
        for injection in self.active_for(stage, sample_id, scheme_id):
            self.hit(injection)
            if injection["fault_type"] == "zero_normalization_sum" and context.get("sum_exp") == 0:
                raise AuditError(
                    code=injection["code"],
                    message=injection["message"],
                    sample_id=sample_id,
                    scheme_id=scheme_id,
                    stage=stage,
                )

    def forced_invalidation_stages(self) -> set[tuple[str, str, str]]:
        return {
            (item["sample_id"], item["scheme_id"] or "*", item["stage"])
            for item in self.injections
            if item.get("sample_id")
        }
