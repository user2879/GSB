"""Session lifecycle, crash recovery, replay verification, and deterministic runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_value
from .engine import Engine, EventResult, RuntimeState
from .errors import WorkbenchError
from .ledger import Ledger
from .spec import Declaration, TriggerEvent, load_declaration, load_events


RECORD_KEY_ORDER = (
    "format_version",
    "event_id",
    "event",
    "terminal",
    "final_matrix",
    "iterations",
    "preemptions",
    "workload_state",
    "gap",
)


def _tenant_hint(value: Any, fallback: str | None = None) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("tenant_id"), str):
        return value["tenant_id"]
    return fallback


def first_record_difference(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    tenant_id = _tenant_hint(expected, tenant_id) or _tenant_hint(actual, tenant_id)
    if type(expected) is not type(actual):
        return {
            "field": path,
            "tenant_id": tenant_id,
            "expected": expected,
            "actual": actual,
        }
    if isinstance(expected, dict):
        keys = [key for key in RECORD_KEY_ORDER if key in expected or key in actual]
        keys.extend(sorted((set(expected) | set(actual)) - set(keys)))
        for key in keys:
            if key not in expected:
                return {
                    "field": f"{path}.{key}",
                    "tenant_id": tenant_id,
                    "expected": "<missing>",
                    "actual": actual[key],
                }
            if key not in actual:
                return {
                    "field": f"{path}.{key}",
                    "tenant_id": tenant_id,
                    "expected": expected[key],
                    "actual": "<missing>",
                }
            diff = first_record_difference(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                tenant_id=tenant_id,
            )
            if diff is not None:
                return diff
        return None
    if isinstance(expected, list):
        tenant_items = all(
            isinstance(item, dict) and isinstance(item.get("tenant_id"), str)
            for item in expected + actual
        ) and bool(expected or actual) and (
            len({item["tenant_id"] for item in expected}) == len(expected)
            and len({item["tenant_id"] for item in actual}) == len(actual)
        )
        if tenant_items:
            expected_by_tenant = {item["tenant_id"]: item for item in expected}
            actual_by_tenant = {item["tenant_id"]: item for item in actual}
            for key in sorted(set(expected_by_tenant) | set(actual_by_tenant)):
                if key not in expected_by_tenant:
                    return {
                        "field": f"{path}[{key}]",
                        "tenant_id": key,
                        "expected": "<missing>",
                        "actual": actual_by_tenant[key],
                    }
                if key not in actual_by_tenant:
                    return {
                        "field": f"{path}[{key}]",
                        "tenant_id": key,
                        "expected": expected_by_tenant[key],
                        "actual": "<missing>",
                    }
                diff = first_record_difference(
                    expected_by_tenant[key],
                    actual_by_tenant[key],
                    path=f"{path}[{key}]",
                    tenant_id=key,
                )
                if diff is not None:
                    return diff
            return None
        max_length = max(len(expected), len(actual))
        for index in range(max_length):
            if index >= len(expected):
                return {
                    "field": f"{path}[{index}]",
                    "tenant_id": tenant_id,
                    "expected": "<missing>",
                    "actual": actual[index],
                }
            if index >= len(actual):
                return {
                    "field": f"{path}[{index}]",
                    "tenant_id": tenant_id,
                    "expected": expected[index],
                    "actual": "<missing>",
                }
            diff = first_record_difference(
                expected[index],
                actual[index],
                path=f"{path}[{index}]",
                tenant_id=tenant_id,
            )
            if diff is not None:
                return diff
        return None
    if expected != actual:
        return {
            "field": path,
            "tenant_id": tenant_id,
            "expected": expected,
            "actual": actual,
        }
    return None


class WorkbenchSession:
    def __init__(
        self,
        declaration: Declaration,
        events: tuple[TriggerEvent, ...],
        ledger: Ledger,
        state: RuntimeState | None = None,
    ):
        self.declaration = declaration
        self.events = events
        self.ledger = ledger
        self.engine = Engine(declaration)
        self.spec_path: str | None = None
        self.events_path: str | None = None
        self.ledger.ensure_context(declaration)
        self.recovery = self.ledger.note_startup_recovery()
        self.ledger.verify_chain()
        if self.ledger.cursor > len(events):
            raise WorkbenchError(
                "E_CONTEXT",
                "ledger cursor is beyond the supplied event stream",
                object_type="ledger",
                field="cursor",
                details={"cursor": self.ledger.cursor, "event_count": len(events)},
            )
        self._check_event_stream_matches_ledger()
        restored = self.ledger.latest_state()
        if state is not None:
            candidate = state
        elif restored is None:
            candidate = RuntimeState.initial(declaration)
        else:
            try:
                candidate = RuntimeState.from_dict(restored, declaration)
            except ValueError as exc:
                raise WorkbenchError(
                    "E_LEDGER",
                    f"cannot restore committed runtime state: {exc}",
                    object_type="ledger",
                    field="state_json",
                ) from exc
        self.state = candidate

    @classmethod
    def open(
        cls,
        declaration_path: Path | str,
        events_path: Path | str,
        db_path: Path | str,
    ) -> "WorkbenchSession":
        declaration = load_declaration(declaration_path)
        events = load_events(events_path, declaration)
        ledger = Ledger(db_path)
        try:
            session = cls(declaration, events, ledger)
        except Exception:
            ledger.close()
            raise
        session.spec_path = str(Path(declaration_path))
        session.events_path = str(Path(events_path))
        return session

    def close(self) -> None:
        self.ledger.close()

    @property
    def cursor(self) -> int:
        return self.ledger.cursor

    @property
    def completed(self) -> bool:
        return self.cursor >= len(self.events)

    def recovery_report(self) -> dict[str, Any]:
        return self.ledger.recovery_report()

    def declaration_versions(self) -> dict[str, Any]:
        return self.declaration.versions()

    def _check_event_stream_matches_ledger(self) -> None:
        records = self.ledger.committed_records()
        for item in records:
            event_id = int(item["event_id"])
            stored_event = item["record"].get("event", {})
            supplied = self.events[event_id - 1].to_dict()
            if stored_event != supplied:
                raise WorkbenchError(
                    "E_CONTEXT",
                    "event stream differs from the committed ledger context and cannot be reproduced",
                    object_type="event",
                    object_id=str(event_id),
                    field="event",
                    details={
                        "unreproducible": True,
                        "reason": "EVENT_STREAM_CHANGED",
                        "stored": stored_event,
                        "supplied": supplied,
                    },
                )

    def run_next(self, *, crash_after_prepare: bool = False) -> EventResult | None:
        if self.completed:
            return None
        event = self.events[self.cursor]
        self.ledger.prepare_event(event.event_id)
        if crash_after_prepare:
            os._exit(137)
        try:
            result = self.engine.apply(self.state, event)
        except Exception as exc:
            self.ledger.fail_attempt(event.event_id, str(exc))
            raise
        self.ledger.commit_result(result)
        self.state = result.state
        return result

    def run_all(self, *, stop_on_unsatisfied: bool = True) -> list[EventResult]:
        results: list[EventResult] = []
        while not self.completed:
            result = self.run_next()
            if result is None:
                break
            results.append(result)
            if result.unsatisfied and stop_on_unsatisfied:
                break
        return results

    def current_matrix(self) -> dict[str, Any]:
        matrix = self.ledger.latest_matrix()
        if matrix is not None:
            return matrix
        from .engine import build_matrix

        return build_matrix(self.declaration, self.state)

    def current_preemptions(self) -> list[dict[str, Any]]:
        if self.cursor == 0:
            return []
        return self.ledger.preemptions_for_event(self.cursor)

    def current_iterations(self) -> list[dict[str, Any]]:
        if self.cursor == 0:
            return []
        return self.ledger.iterations_for_event(self.cursor)

    def state_digest(self) -> str:
        return digest_value(
            {
                "versions": self.declaration.versions(),
                "cursor": self.cursor,
                "head_digest": self.ledger.head_digest,
                "state": self.state.to_dict(self.declaration),
                "matrix": self.current_matrix(),
            }
        )

    def _raise_replay(
        self,
        *,
        event_id: int,
        field: str,
        tenant_id: str | None,
        expected: Any,
        actual: Any,
        basis: str,
    ) -> None:
        raise WorkbenchError(
            "E_REPLAY",
            f"replay mismatch at event {event_id}, tenant {tenant_id or '-'}, field {field}",
            object_type="event",
            object_id=str(event_id),
            field=field,
            details={
                "first_inconsistent_event_id": event_id,
                "tenant_id": tenant_id,
                "expected": expected,
                "actual": actual,
                "basis": basis,
            },
        )

    def _compare_normalized_storage(self, event_id: int, record: dict[str, Any]) -> None:
        row = self.ledger.event_storage_row(event_id)
        if row is None:
            self._raise_replay(
                event_id=event_id,
                field="events",
                tenant_id=None,
                expected="committed row",
                actual="<missing>",
                basis="normalized ledger event",
            )
        terminal = record["terminal"]
        expected_event_fields = {
            "terminal_code": terminal["code"],
            "terminal_reason": terminal["reason"],
            "iteration_count": terminal["iteration_count"],
            "preemption_count": terminal["preemption_count"],
            "reclaimed_units": terminal["reclaimed_units"],
        }
        for field, expected in expected_event_fields.items():
            if row[field] != expected:
                self._raise_replay(
                    event_id=event_id,
                    field=f"events.{field}",
                    tenant_id=None,
                    expected=expected,
                    actual=row[field],
                    basis="normalized event ledger",
                )
        if row["event_json"] != canonical_json(record["event"]):
            self._raise_replay(
                event_id=event_id,
                field="events.event_json",
                tenant_id=None,
                expected=canonical_json(record["event"]),
                actual=row["event_json"],
                basis="normalized event ledger",
            )
        byte_fields = {
            "state_json": canonical_json(record["workload_state"]),
            "final_matrix_json": canonical_json(record["final_matrix"]),
            "gap_json": (
                canonical_json(record["gap"]) if record["gap"] is not None else None
            ),
        }
        for field, expected in byte_fields.items():
            if row[field] != expected:
                self._raise_replay(
                    event_id=event_id,
                    field=f"events.{field}",
                    tenant_id=None,
                    expected=expected,
                    actual=row[field],
                    basis="normalized byte-state ledger",
                )

        expected_iterations = record["iterations"]
        actual_iterations = self.ledger.iterations_for_event(event_id)
        if len(expected_iterations) != len(actual_iterations):
            self._raise_replay(
                event_id=event_id,
                field="iterations.length",
                tenant_id=None,
                expected=len(expected_iterations),
                actual=len(actual_iterations),
                basis="normalized iteration ledger",
            )
        for expected_iteration, actual_iteration in zip(expected_iterations, actual_iterations):
            expected_gap = (
                canonical_json(expected_iteration["gap"])
                if expected_iteration.get("gap") is not None
                else None
            )
            iteration_fields = {
                "iteration": expected_iteration["iteration"],
                "terminal_reason": expected_iteration.get("terminal_reason"),
                "matrix_digest": expected_iteration["matrix_digest"],
                "matrix_json": canonical_json(expected_iteration["matrix"]),
                "gap_json": expected_gap,
            }
            for field, expected in iteration_fields.items():
                if actual_iteration[field] != expected:
                    self._raise_replay(
                        event_id=event_id,
                        field=f"iterations[{expected_iteration['iteration']}].{field}",
                        tenant_id=None,
                        expected=expected,
                        actual=actual_iteration[field],
                        basis="normalized iteration ledger",
                    )

        expected_by_iteration_tenant: dict[tuple[int, str], dict[str, Any]] = {}
        for iteration in record["iterations"]:
            for tenant_row in iteration["matrix"]["tenants"]:
                expected_by_iteration_tenant[(iteration["iteration"], tenant_row["tenant_id"])] = tenant_row
        actual_rows = self.ledger.matrices_for_event(event_id)
        actual_by_iteration_tenant = {
            (item["iteration"], item["tenant_id"]): item for item in actual_rows
        }
        for key in sorted(set(expected_by_iteration_tenant) | set(actual_by_iteration_tenant)):
            if key not in expected_by_iteration_tenant:
                self._raise_replay(
                    event_id=event_id,
                    field=f"tenant_matrices[{key[0]}].{key[1]}",
                    tenant_id=key[1],
                    expected="<missing>",
                    actual=actual_by_iteration_tenant[key],
                    basis="normalized matrix ledger",
                )
            if key not in actual_by_iteration_tenant:
                self._raise_replay(
                    event_id=event_id,
                    field=f"tenant_matrices[{key[0]}].{key[1]}",
                    tenant_id=key[1],
                    expected=expected_by_iteration_tenant[key],
                    actual="<missing>",
                    basis="normalized matrix ledger",
                )
            expected_row = expected_by_iteration_tenant[key]
            actual_row = actual_by_iteration_tenant[key]
            for field in expected_row:
                if actual_row.get(field) != expected_row[field]:
                    self._raise_replay(
                        event_id=event_id,
                        field=f"tenant_matrices[{key[0]}].{key[1]}.{field}",
                        tenant_id=key[1],
                        expected=expected_row[field],
                        actual=actual_row.get(field),
                        basis="normalized matrix ledger",
                    )
            expected_digest = next(
                iteration["matrix_digest"]
                for iteration in record["iterations"]
                if iteration["iteration"] == key[0]
            )
            if actual_row["matrix_digest"] != expected_digest:
                self._raise_replay(
                    event_id=event_id,
                    field=f"tenant_matrices[{key[0]}].{key[1]}.matrix_digest",
                    tenant_id=key[1],
                    expected=expected_digest,
                    actual=actual_row["matrix_digest"],
                    basis="normalized matrix ledger",
                )

        expected_preemptions = record["preemptions"]
        actual_preemptions = self.ledger.preemptions_for_event(event_id)
        if len(expected_preemptions) != len(actual_preemptions):
            self._raise_replay(
                event_id=event_id,
                field="preemptions.length",
                tenant_id=None,
                expected=len(expected_preemptions),
                actual=len(actual_preemptions),
                basis="normalized preemption ledger",
            )
        for expected, actual in zip(expected_preemptions, actual_preemptions):
            for field in (
                "sequence",
                "iteration",
                "tenant_id",
                "workload_id",
                "replicas_preempted",
                "units_per_replica",
                "reclaimed_units",
                "tenant_usage_before_units",
                "tenant_usage_after_units",
                "cluster_usage_before_units",
                "cluster_usage_after_units",
                "matrix_before_digest",
                "matrix_after_digest",
            ):
                if expected[field] != actual[field]:
                    self._raise_replay(
                        event_id=event_id,
                        field=f"preemptions[{expected['sequence']}].{field}",
                        tenant_id=expected["tenant_id"],
                        expected=expected[field],
                        actual=actual[field],
                        basis="normalized preemption ledger",
                    )
            if canonical_json(expected["selection_basis"]) != actual["selection_basis_json"]:
                self._raise_replay(
                    event_id=event_id,
                    field=f"preemptions[{expected['sequence']}].selection_basis",
                    tenant_id=expected["tenant_id"],
                    expected=canonical_json(expected["selection_basis"]),
                    actual=actual["selection_basis_json"],
                    basis="normalized preemption ledger",
                )

    def verify_replay(self) -> dict[str, Any]:
        try:
            self.ledger.verify_chain()
        except WorkbenchError as exc:
            if exc.code != "E_LEDGER":
                raise
            raise WorkbenchError(
                "E_REPLAY",
                "replay cannot establish ledger integrity",
                object_type=exc.object_type or "event",
                object_id=exc.object_id,
                field=exc.field or "ledger",
                details={
                    "first_inconsistent_event_id": exc.object_id,
                    "tenant_id": None,
                    "basis": "ledger digest chain",
                    "cause": exc.to_dict(),
                },
            ) from exc
        committed = self.ledger.committed_records(verify_digests=False)
        state = RuntimeState.initial(self.declaration)
        for item in committed:
            event_id = int(item["event_id"])
            event = self.events[event_id - 1]
            digest_of_bytes = digest_value(item["record"])
            if digest_of_bytes != item["record_digest"]:
                self._raise_replay(
                    event_id=event_id,
                    field="record_digest",
                    tenant_id=None,
                    expected=item["record_digest"],
                    actual=digest_of_bytes,
                    basis="record byte digest",
                )
            stored_event = item["record"].get("event")
            if stored_event != event.to_dict():
                raise WorkbenchError(
                    "E_CONTEXT",
                    "event stream differs from ledger and cannot be reproduced",
                    object_type="event",
                    object_id=str(event_id),
                    field="event",
                    details={
                        "unreproducible": True,
                        "reason": "EVENT_STREAM_CHANGED",
                        "stored": stored_event,
                        "supplied": event.to_dict(),
                    },
                )
            generated = self.engine.apply(state, event)
            expected_record = item["record"]
            if generated.record_json != item["record_json"]:
                diff = first_record_difference(expected_record, generated.record)
                assert diff is not None
                self._raise_replay(
                    event_id=event_id,
                    field=diff["field"],
                    tenant_id=diff["tenant_id"],
                    expected=diff["expected"],
                    actual=diff["actual"],
                    basis="canonical record replay",
                )
            self._compare_normalized_storage(event_id, generated.record)
            state = generated.state
        restored_state = self.state.to_dict(self.declaration)
        replayed_state = state.to_dict(self.declaration)
        if restored_state != replayed_state:
            diff = first_record_difference(restored_state, replayed_state, path="state")
            assert diff is not None
            self._raise_replay(
                event_id=max(1, self.cursor),
                field=diff["field"],
                tenant_id=diff["tenant_id"],
                expected=diff["expected"],
                actual=diff["actual"],
                basis="runtime state replay",
            )
        return {
            "status": "REPLAY_OK",
            "event_count": len(committed),
            "cursor": self.cursor,
            "head_digest": self.ledger.head_digest,
            "state_digest": digest_value(replayed_state),
        }
