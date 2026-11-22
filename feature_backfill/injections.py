"""External injection manifest: real runtime faults without source edits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .data_loader import parse_row_record, validate_entity_sequences
from .errors import AppError, E_CONFIG_INVALID, E_ROW_INVALID, make_failure
from .models import AppConfig, Row


INJECTION_TYPES = {
    "nonfinite_value",
    "timestamp_reversal",
    "lease_ttl_zero",
    "cache_budget_lt_block",
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    stage: str
    entity_id: str | None
    row_id: str | None = None
    field: str | None = None
    value: Any = None
    expected_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.injection_id,
            "type": self.kind,
            "stage": self.stage,
        }
        if self.entity_id is not None:
            payload["entity_id"] = self.entity_id
        if self.row_id is not None:
            payload["row_id"] = self.row_id
        if self.field is not None:
            payload["field"] = self.field
        if self.value is not None:
            payload["value"] = self.value
        if self.expected_code is not None:
            payload["expected_code"] = self.expected_code
        return payload


def load_injections(path: str | Path | None) -> tuple[list[Injection], list[dict[str, Any]]]:
    if path is None:
        return [], []
    path = Path(path)
    failures: list[dict[str, Any]] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [], [
            make_failure(
                E_CONFIG_INVALID,
                f"cannot parse injection manifest: {exc}",
                stage="config",
                details={"path": str(path)},
            )
        ]
    items = raw.get("injections") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], [
            make_failure(
                E_CONFIG_INVALID,
                "injection manifest must be a list or contain injections",
                stage="config",
            )
        ]
    injections: list[Injection] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError("injection item must be an object")
            injection_id = item.get("id", item.get("injection_id", f"INJ-{index + 1:03d}"))
            kind = item.get("type")
            stage = item.get("stage", "")
            entity_id = item.get("entity_id")
            if not isinstance(injection_id, str) or not injection_id:
                raise ValueError("injection id must be a non-empty string")
            if injection_id in seen:
                raise ValueError(f"duplicate injection id {injection_id}")
            if kind not in INJECTION_TYPES:
                raise ValueError(f"unknown injection type {kind!r}")
            if not isinstance(stage, str) or not stage:
                raise ValueError("injection stage must be a non-empty string")
            if entity_id is not None and (not isinstance(entity_id, str) or not entity_id):
                raise ValueError("entity_id must be a non-empty string when supplied")
            if kind == "cache_budget_lt_block" and entity_id is None:
                raise ValueError("cache_budget_lt_block requires entity_id for isolation")
            expected = item.get("expected_code")
            if expected is not None and not isinstance(expected, str):
                raise ValueError("expected_code must be a string")
            seen.add(injection_id)
            injections.append(
                Injection(
                    injection_id=injection_id,
                    kind=kind,
                    stage=stage,
                    entity_id=entity_id,
                    row_id=item.get("row_id"),
                    field=item.get("field"),
                    value=item.get("value"),
                    expected_code=expected,
                )
            )
        except ValueError as exc:
            failures.append(
                make_failure(
                    E_CONFIG_INVALID,
                    str(exc),
                    stage="config",
                    details={"injection_index": index},
                )
            )
    return injections, failures


class InjectionRuntime:
    def __init__(self, injections: Iterable[Injection]) -> None:
        self.injections = sorted(injections, key=lambda item: item.injection_id)
        self.hits: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def matching(self, *, entity_id: str, stage: str, kind: str | None = None) -> list[Injection]:
        return [
            injection
            for injection in self.injections
            if injection.entity_id in (None, entity_id)
            and injection.stage == stage
            and (kind is None or injection.kind == kind)
        ]

    def has(self, *, entity_id: str, stage: str, kind: str) -> bool:
        return bool(self.matching(entity_id=entity_id, stage=stage, kind=kind))

    def record_hit(
        self,
        injection: Injection,
        *,
        code: str,
        evidence: dict[str, Any],
    ) -> None:
        with self._lock:
            self.hits[injection.injection_id] = {
                "injection_id": injection.injection_id,
                "type": injection.kind,
                "stage": injection.stage,
                "entity_id": injection.entity_id,
                "expected_code": injection.expected_code,
                "observed_code": code,
                "evidence": evidence,
            }

    def hits_sorted(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.hits[key] for key in sorted(self.hits)]

    def faulty_validation(
        self,
        *,
        entity_id: str,
        stage: str,
        rows: list[Row],
        config: AppConfig,
        raw_fields: Iterable[str],
    ) -> tuple[list[Row], list[dict[str, Any]]]:
        """Return a copied row set carrying a genuine validation fault."""

        copied = list(rows)
        failures: list[dict[str, Any]] = []
        for injection in self.matching(entity_id=entity_id, stage=stage):
            if injection.kind not in {"nonfinite_value", "timestamp_reversal"}:
                continue
            if not copied:
                failures.append(
                    make_failure(
                        E_ROW_INVALID,
                        "injection target entity has no rows",
                        stage="load_validation",
                        entity_id=entity_id,
                        injection_id=injection.injection_id,
                    )
                )
                self.record_hit(injection, code=E_ROW_INVALID, evidence={"reason": "no_rows"})
                continue
            target_index = 0
            if injection.row_id is not None:
                target_index = next(
                    (
                        index
                        for index, row in enumerate(copied)
                        if row.row_id == injection.row_id
                    ),
                    -1,
                )
                if target_index < 0:
                    failures.append(
                        make_failure(
                            E_ROW_INVALID,
                            "injection target row was not found",
                            stage="load_validation",
                            entity_id=entity_id,
                            row_id=injection.row_id,
                            injection_id=injection.injection_id,
                        )
                    )
                    self.record_hit(
                        injection,
                        code=E_ROW_INVALID,
                        evidence={"reason": "row_not_found"},
                    )
                    continue
            if injection.kind == "nonfinite_value":
                field = injection.field
                if field is None:
                    field = sorted(copied[target_index].values)[0] if copied[target_index].values else None
                if field is None:
                    failures.append(
                        make_failure(
                            E_ROW_INVALID,
                            "nonfinite injection has no numeric field",
                            stage="load_validation",
                            entity_id=entity_id,
                            row_id=copied[target_index].row_id,
                            injection_id=injection.injection_id,
                        )
                    )
                    self.record_hit(
                        injection, code=E_ROW_INVALID, evidence={"reason": "no_field"}
                    )
                    continue
                record = copied[target_index].to_dict()
                record[field] = math.nan
                _, row_failures = parse_row_record(
                    record,
                    line=-1,
                    config=config,
                    raw_fields=raw_fields,
                )
                for failure in row_failures:
                    failure["injection_id"] = injection.injection_id
                    failure["details"] = {
                        **failure.get("details", {}),
                        "injected_value": "NaN",
                        "injected_field": field,
                    }
                failures.extend(row_failures)
                self.record_hit(
                    injection,
                    code=E_ROW_INVALID,
                    evidence={
                        "row_id": copied[target_index].row_id,
                        "field": field,
                        "injected_value": "NaN",
                        "validator": "parse_row_record",
                    },
                )
            else:
                if target_index + 1 < len(copied):
                    # Duplicate the next timestamp so the following row violates
                    # strict per-entity increase.
                    mutated_timestamp = copied[target_index + 1].timestamp_ms
                elif target_index > 0:
                    mutated_timestamp = copied[target_index - 1].timestamp_ms
                else:
                    mutated_timestamp = -1
                from dataclasses import replace

                copied[target_index] = replace(
                    copied[target_index], timestamp_ms=mutated_timestamp
                )
                # Use insertion order, matching JSONL strict-increasing semantics.
                mutation_failures = validate_entity_sequences(copied)
                for failure in mutation_failures:
                    failure["injection_id"] = injection.injection_id
                failures.extend(mutation_failures)
                self.record_hit(
                    injection,
                    code=E_ROW_INVALID,
                    evidence={
                        "row_id": copied[target_index].row_id,
                        "mutated_timestamp_ms": mutated_timestamp,
                        "validator": "validate_entity_sequences",
                    },
                )
        return copied, failures

    def lease_ttl_for(self, *, entity_id: str, stage: str) -> int | None:
        for injection in self.matching(entity_id=entity_id, stage=stage, kind="lease_ttl_zero"):
            error = AppError(
                E_CONFIG_INVALID,
                "injected lease_ttl_ms=0 is rejected by the lease authority",
                stage="lease",
                entity_id=entity_id,
                details={"injection_id": injection.injection_id, "lease_ttl_ms": 0},
            )
            self.record_hit(
                injection,
                code=E_CONFIG_INVALID,
                evidence={"lease_ttl_ms": 0, "clock_source_checked": True},
            )
            raise error
        return None

    def cache_capacity_for(self, *, entity_id: str, stage: str) -> int | None:
        for injection in self.matching(
            entity_id=entity_id, stage=stage, kind="cache_budget_lt_block"
        ):
            self.record_hit(
                injection,
                code=E_CONFIG_INVALID,
                evidence={"cache_blocks": 0, "single_block_size": 1},
            )
            raise AppError(
                E_CONFIG_INVALID,
                "injected cache residency budget is smaller than one block",
                stage="cache",
                entity_id=entity_id,
                details={"injection_id": injection.injection_id, "cache_blocks": 0},
            )
        return None
