"""Workbench orchestration: resume, deterministic run, and ledger replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .engine import LeaseEngine
from .errors import DeclError, ReplayError, SnapshotError
from .ledger import Ledger, RecoveryReport
from .logging_utils import StructuredLogger
from .models import CallEvent, Declaration, SessionConfig, canonical_json, content_digest
from .validation import parse_declaration


@dataclass
class RunResult:
    batches_written: int
    final_cursor: int
    last_batch_id: str | None
    declaration_version: int
    recovery_status: str
    pending_batch_ids: list[str]
    unknown_lease_ids: list[str]
    recomputed_unknown_lease_ids: list[str]
    view: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches_written": self.batches_written,
            "final_cursor": self.final_cursor,
            "last_batch_id": self.last_batch_id,
            "declaration_version": self.declaration_version,
            "recovery_status": self.recovery_status,
            "pending_batch_ids": list(self.pending_batch_ids),
            "unknown_lease_ids": list(self.unknown_lease_ids),
            "recomputed_unknown_lease_ids": list(self.recomputed_unknown_lease_ids),
            "view": self.view,
        }


@dataclass
class ReplayResult:
    ok: bool
    batches_checked: int
    events_checked: int
    declaration_versions: list[int]
    first_mismatch_batch_id: str | None = None
    first_mismatch_event_seq: int | None = None
    first_mismatch_lease_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "batches_checked": self.batches_checked,
            "events_checked": self.events_checked,
            "declaration_versions": list(self.declaration_versions),
            "first_mismatch_batch_id": self.first_mismatch_batch_id,
            "first_mismatch_event_seq": self.first_mismatch_event_seq,
            "first_mismatch_lease_id": self.first_mismatch_lease_id,
        }


def group_events(events: Sequence[CallEvent], *, after_seq: int = 0) -> list[list[CallEvent]]:
    groups: list[list[CallEvent]] = []
    current: list[CallEvent] = []
    current_arrival: int | None = None
    for event in events:
        if event.event_seq <= after_seq:
            continue
        if current_arrival is None:
            current_arrival = event.arrival_ms
        if event.arrival_ms != current_arrival:
            groups.append(current)
            current = []
            current_arrival = event.arrival_ms
        current.append(event)
    if current:
        groups.append(current)
    return groups


def _engine_declaration(state: dict[str, Any]) -> Declaration:
    return parse_declaration(state["declaration"])


def _engine_session(state: dict[str, Any]) -> SessionConfig:
    raw = state["session"]
    return SessionConfig(
        watermark_scale=int(raw["watermark_scale"]),
        borrow_return_error_ms=int(raw.get("borrow_return_error_ms", 50)),
    )


class Workbench:
    def __init__(
        self,
        *,
        db_path: str | Path,
        declaration: Declaration,
        events: Sequence[CallEvent],
        session: SessionConfig,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.requested_declaration = declaration
        self.events = list(events)
        self.requested_session = session
        self.logger = logger or StructuredLogger(None)
        self.ledger = Ledger(self.db_path)
        self.recovery: RecoveryReport = self.ledger.recover()
        self.engine = self._build_engine()
        self.last_result: RunResult | None = None

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "Workbench":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _build_engine(self) -> LeaseEngine:
        report = self.recovery
        if report.state is None:
            return LeaseEngine(self.requested_declaration, self.requested_session)
        stored_session = _engine_session(report.state)
        if stored_session.watermark_scale != self.requested_session.watermark_scale:
            raise DeclError(
                "watermark_scale",
                "watermark scale cannot change while resuming a ledger",
                ledger=stored_session.watermark_scale,
                requested=self.requested_session.watermark_scale,
            )
        if (
            stored_session.borrow_return_error_ms
            != self.requested_session.borrow_return_error_ms
        ):
            raise DeclError(
                "borrow_return_error_ms",
                "borrow-return error window cannot change while resuming a ledger",
                ledger=stored_session.borrow_return_error_ms,
                requested=self.requested_session.borrow_return_error_ms,
            )
        stored_declaration = _engine_declaration(report.state)
        if self.requested_declaration.declaration_version < stored_declaration.declaration_version:
            raise DeclError(
                "declaration_version",
                "declaration version regressed relative to committed ledger",
                ledger=stored_declaration.declaration_version,
                requested=self.requested_declaration.declaration_version,
            )
        if (
            self.requested_declaration.declaration_version
            == stored_declaration.declaration_version
            and self.requested_declaration.digest != stored_declaration.digest
        ):
            raise DeclError(
                "declaration_version",
                "declaration content changed without incrementing version",
                version=self.requested_declaration.declaration_version,
            )
        return LeaseEngine.from_state_dict(
            stored_declaration,
            stored_session,
            report.state,
        )

    def run(
        self,
        *,
        limit_event_seq: int | None = None,
        pending_hook: Callable[[str], None] | None = None,
        inject_quota: bool = False,
    ) -> RunResult:
        if inject_quota:
            self.engine.set_quota_injection(True)
        cursor = self.ledger.current_cursor
        groups = group_events(self.events, after_seq=cursor)
        batches_written = 0
        declaration_change_pending = (
            self.requested_declaration.declaration_version
            != self.engine.declaration.declaration_version
            or self.requested_declaration.digest != self.engine.declaration.digest
        )
        declaration_applied = False
        unknown_from_recovery = list(self.recovery.unknown_lease_ids)

        for group in groups:
            if limit_event_seq is not None and group[0].event_seq > limit_event_seq:
                break
            if cursor and group[0].event_seq != cursor + 1:
                raise SnapshotError(
                    "event_seq",
                    "next event does not continue from ledger cursor",
                    cursor=cursor,
                    next_event_seq=group[0].event_seq,
                )
            self.engine.current_ops = []
            if declaration_change_pending and not declaration_applied:
                self.engine.apply_declaration(
                    self.requested_declaration, group[0].arrival_ms
                )
                declaration_applied = True
                self.logger.write(
                    "declaration.applied",
                    declaration_version=self.requested_declaration.declaration_version,
                    at_ms=group[0].arrival_ms,
                    invalidated_leases=list(self.engine.invalidations),
                )
            batch_id = self.engine.make_batch_id(group)
            self.engine.batch_id = batch_id
            candidates = self._candidate_lease_ids(group)
            self.ledger.begin_pending(
                batch_id=batch_id,
                declaration_version=self.engine.declaration.declaration_version,
                start_event_seq=group[0].event_seq,
                end_event_seq=group[-1].event_seq,
                candidate_lease_ids=candidates,
            )
            if pending_hook is not None:
                pending_hook(batch_id)
            payload = self.engine.run_batch(group)
            state = self.engine.to_state_dict()
            self.ledger.append_batch(
                payload=payload,
                state=state,
                declaration_source=self.engine.declaration.source,
            )
            batches_written += 1
            cursor = int(payload["cursor_event_seq"])
            self.logger.write(
                "batch.committed",
                batch_id=batch_id,
                cursor=cursor,
                declaration_version=payload["declaration_version"],
                operations=len(payload["operations"]),
                watermarks=len(payload["watermarks"]),
            )

        if declaration_change_pending and not declaration_applied and groups:
            raise DeclError(
                "declaration_version",
                "declaration evolution requires at least one unprocessed event batch",
            )
        if not groups:
            self.logger.write(
                "run.noop",
                cursor=cursor,
                reason="no unprocessed events",
            )
        if batches_written:
            post_recovery = self.ledger.recover()
            current_status = post_recovery.status
            current_pending = list(post_recovery.pending_batch_ids)
            current_unknown = list(post_recovery.unknown_lease_ids)
        else:
            current_status = self.recovery.status
            current_pending = list(self.recovery.pending_batch_ids)
            current_unknown = list(unknown_from_recovery)
        view = self.current_view(
            recovery_status=current_status,
            pending_batch_ids=current_pending,
            unknown_lease_ids=current_unknown,
        )
        result = RunResult(
            batches_written=batches_written,
            final_cursor=cursor,
            last_batch_id=self.ledger.last_batch_id,
            declaration_version=self.engine.declaration.declaration_version,
            recovery_status=current_status,
            pending_batch_ids=current_pending,
            unknown_lease_ids=current_unknown,
            recomputed_unknown_lease_ids=(
                unknown_from_recovery if batches_written else []
            ),
            view=view,
        )
        self.last_result = result
        return result

    def _candidate_lease_ids(self, group: Sequence[CallEvent]) -> list[str]:
        ids: list[str] = []
        for event in group:
            count = self.engine.grant_counts.get(event.event_seq, 0)
            if count == 0:
                ids.append(f"L{event.event_seq:08d}")
            else:
                ids.append(f"L{event.event_seq:08d}R{count:02d}")
            ids.append(f"L{event.event_seq:08d}R01")
        return sorted(set(ids))

    def current_view(
        self,
        *,
        recovery_status: str | None = None,
        pending_batch_ids: Iterable[str] = (),
        unknown_lease_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        rows = self.ledger.load_view_rows()
        snapshot = self.engine.snapshot()
        unknown = sorted(
            set(unknown_lease_ids)
            | {item["lease_id"] for item in rows.get("unknown_leases", [])}
        )
        view = {
            **snapshot,
            "ledger_path": str(self.db_path),
            "ledger_batch": self.ledger.last_batch_id,
            "ledger_cursor": self.ledger.current_cursor,
            "ledger_watermarks": rows["watermarks"],
            "ledger_invalidations": rows["invalidations"],
            "recovery_status": recovery_status or self.recovery.status,
            "recovery_message": self.recovery.message,
            "pending_batch_ids": sorted(set(pending_batch_ids)),
            "unknown_lease_ids": unknown,
            "unknown_requires_recompute": bool(unknown),
        }
        view["view_digest"] = content_digest(
            {key: value for key, value in view.items() if key != "view_digest"}
        )
        return view


def replay_ledger(
    *,
    db_path: str | Path,
    logger: StructuredLogger | None = None,
) -> ReplayResult:
    logger = logger or StructuredLogger(None)
    ledger = Ledger(db_path)
    try:
        batches = ledger.load_batches()
        declarations = ledger.declaration_sources()
        if not batches:
            raise ReplayError(
                "batches",
                "ledger contains no committed batch to replay",
                event_seq=None,
                lease_id=None,
            )
        first_payload = batches[0]["payload"]
        first_version = int(first_payload["declaration_version"])
        if first_version not in declarations:
            raise SnapshotError(
                "declarations",
                "ledger declaration source is missing",
                version=first_version,
            )
        declaration = parse_declaration(declarations[first_version])
        state = batches[0]["state"]
        raw_session = state["session"]
        session = SessionConfig(
            watermark_scale=int(first_payload["scale"]),
            borrow_return_error_ms=int(raw_session.get("borrow_return_error_ms", 50)),
        )
        engine = LeaseEngine(declaration, session)
        events_checked = 0
        versions: list[int] = []
        for batch in batches:
            payload = batch["payload"]
            batch_version = int(payload["declaration_version"])
            versions.append(batch_version)
            if int(payload["scale"]) != session.watermark_scale:
                raise ReplayError(
                    "scale",
                    "watermark scale changed within one ledger",
                    event_seq=int(payload["start_event_seq"]),
                    lease_id=None,
                    batch_id=batch["batch_id"],
                )
            if batch_version not in declarations:
                raise SnapshotError(
                    "declarations",
                    "ledger batch references a missing declaration",
                    version=batch_version,
                    batch_id=batch["batch_id"],
                )
            target_declaration = parse_declaration(declarations[batch_version])
            if target_declaration.digest != payload.get("declaration_digest"):
                raise ReplayError(
                    "declaration_digest",
                    "ledger declaration digest does not match stored source",
                    event_seq=int(payload["start_event_seq"]),
                    lease_id=None,
                    batch_id=batch["batch_id"],
                )
            events = [CallEvent(**item) for item in payload["events"]]
            events_checked += len(events)
            engine.current_ops = []
            engine.batch_id = batch["batch_id"]
            if target_declaration.declaration_version != engine.declaration.declaration_version:
                engine.apply_declaration(target_declaration, events[0].arrival_ms)
            replay_payload = engine.run_batch(events)
            mismatch = _first_payload_mismatch(replay_payload, payload)
            if mismatch is not None:
                event_seq, lease_id, field = mismatch
                logger.write(
                    "replay.mismatch",
                    level="ERROR",
                    code="E_REPLAY",
                    field=field,
                    event_seq=event_seq,
                    lease_id=lease_id,
                    batch_id=batch["batch_id"],
                )
                raise ReplayError(
                    field,
                    "replayed batch does not match committed ledger",
                    event_seq=event_seq,
                    lease_id=lease_id,
                    batch_id=batch["batch_id"],
                )
        logger.write(
            "replay.ok",
            batches=len(batches),
            events=events_checked,
            declaration_versions=versions,
        )
        return ReplayResult(
            ok=True,
            batches_checked=len(batches),
            events_checked=events_checked,
            declaration_versions=versions,
        )
    finally:
        ledger.close()


def _first_payload_mismatch(
    replay: dict[str, Any], committed: dict[str, Any]
) -> tuple[int, str | None, str] | None:
    replay_ops = replay.get("operations", [])
    committed_ops = committed.get("operations", [])
    for index in range(max(len(replay_ops), len(committed_ops))):
        if index >= len(replay_ops) or index >= len(committed_ops):
            operation = (
                committed_ops[index]
                if index < len(committed_ops)
                else replay_ops[index]
            )
            event_seq, lease_id = _location_from_operation(operation)
            return (
                event_seq or int(committed.get("start_event_seq", 1)),
                lease_id,
                f"operations[{index}]",
            )
        if canonical_json(replay_ops[index]) != canonical_json(committed_ops[index]):
            event_seq, lease_id = _location_from_operation(committed_ops[index])
            return (
                event_seq or int(committed.get("start_event_seq", 1)),
                lease_id,
                f"operations[{index}].{committed_ops[index].get('type', 'UNKNOWN')}",
            )
    for key in sorted(set(replay) | set(committed)):
        if key == "operations":
            continue
        if canonical_json(replay.get(key)) != canonical_json(committed.get(key)):
            event_seq = int(committed.get("start_event_seq", 1))
            return event_seq, None, key
    return None


def _location_from_operation(operation: dict[str, Any]) -> tuple[int | None, str | None]:
    event_seq = operation.get("event_seq")
    lease_id = operation.get("lease_id")
    for key in ("lease", "release", "loan", "queue", "preemption"):
        nested = operation.get(key)
        if isinstance(nested, dict):
            if event_seq is None:
                event_seq = nested.get("event_seq") or nested.get("beneficiary_event_seq")
            if lease_id is None:
                lease_id = nested.get("lease_id") or nested.get("victim_lease_id")
    return (
        int(event_seq) if isinstance(event_seq, int) else None,
        str(lease_id) if lease_id is not None else None,
    )


def load_view_from_ledger(db_path: str | Path) -> dict[str, Any]:
    normalized_path = Path(db_path).expanduser().resolve()
    ledger = Ledger(normalized_path)
    try:
        report = ledger.recover(record_unknowns=False)
        if report.state is None:
            view = {
                "declaration_version": report.declaration_version,
                "declaration_digest": None,
                "clock_ms": 0,
                "scale": report.scale,
                "global_used": 0,
                "global_budget": 0,
                "functions": [],
                "leases": [],
                "loans": [],
                "queue": [],
                "preemptions": [],
                "watermarks": [],
                "invalidations": [],
                "ledger_path": str(normalized_path),
                "ledger_batch": ledger.last_batch_id,
                "ledger_cursor": ledger.current_cursor,
                "ledger_watermarks": [],
                "ledger_invalidations": [],
                "recovery_status": report.status,
                "recovery_message": report.message,
                "pending_batch_ids": list(report.pending_batch_ids),
                "unknown_lease_ids": sorted(report.unknown_lease_ids),
                "unknown_requires_recompute": bool(report.pending_batch_ids),
            }
            view["view_digest"] = content_digest(view)
            return view
        declaration = _engine_declaration(report.state)
        session = _engine_session(report.state)
        engine = LeaseEngine.from_state_dict(declaration, session, report.state)
        rows = ledger.load_view_rows()
        view = {
            **engine.snapshot(),
            "ledger_path": str(normalized_path),
            "ledger_batch": ledger.last_batch_id,
            "ledger_cursor": ledger.current_cursor,
            "ledger_watermarks": rows["watermarks"],
            "ledger_invalidations": rows["invalidations"],
            "recovery_status": report.status,
            "recovery_message": report.message,
            "pending_batch_ids": list(report.pending_batch_ids),
            "unknown_lease_ids": sorted(
                set(report.unknown_lease_ids)
                | {item["lease_id"] for item in rows["unknown_leases"]}
            ),
            "unknown_requires_recompute": bool(report.pending_batch_ids),
        }
        view["view_digest"] = content_digest(
            {key: value for key, value in view.items() if key != "view_digest"}
        )
        return view
    finally:
        ledger.close()
