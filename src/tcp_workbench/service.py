"""Session application service: version checks, incremental ingestion, recovery."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clock import monotonic_ms
from .config import validate_peer_script, validate_session_config
from .errors import E_REPLAY, E_STATE, E_VERSION, BusinessError, version_error
from .ledger import Ledger
from .models import (
    Event,
    FieldChange,
    ItemResult,
    PeerScript,
    ReplacementReport,
    SessionConfig,
    canonical_json,
    flatten_values,
)
from .projection import (
    ITEM_ORDER,
    apply_event_to_items,
    compare_items,
    initial_items,
    replay,
    state_snapshot,
)
from .state_machine import SimulationState
from .structured_log import StructuredLogger


@dataclass(frozen=True)
class RecoveryReport:
    complete_event_seqs: tuple[int, ...]
    unknown_event_seqs: tuple[int, ...]
    rerun_required: bool
    exit_report_available: bool
    ledger_consistent: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete_event_seqs": list(self.complete_event_seqs),
            "unknown_event_seqs": list(self.unknown_event_seqs),
            "rerun_required": self.rerun_required,
            "exit_report_available": self.exit_report_available,
            "ledger_consistent": self.ledger_consistent,
            "message": self.message,
        }


@dataclass(frozen=True)
class IngestionResult:
    event: Event
    affected_items: tuple[str, ...]
    item_revisions: dict[str, int]


class SessionService:
    def __init__(
        self,
        ledger: Ledger,
        config: SessionConfig,
        script: PeerScript,
        *,
        clock_fn: Callable[[], int] = monotonic_ms,
        logger: StructuredLogger | None = None,
        crash_after_begin: Callable[[Event], None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.config = config
        self.script = script
        self.clock_fn = clock_fn
        self.logger = logger or StructuredLogger()
        self.crash_after_begin = crash_after_begin
        self.state = SimulationState(config, script)
        self.events: tuple[Event, ...] = ()
        self.items: dict[str, ItemResult] = initial_items(self.state, config, script)
        self.recovery_report: RecoveryReport | None = None
        self.needs_rerun = False

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        config: SessionConfig,
        script: PeerScript,
        *,
        clock_fn: Callable[[], int] = monotonic_ms,
        logger: StructuredLogger | None = None,
        crash_after_begin: Callable[[Event], None] | None = None,
    ) -> SessionService:
        ledger = Ledger(db_path)
        service = cls(
            ledger,
            config,
            script,
            clock_fn=clock_fn,
            logger=logger,
            crash_after_begin=crash_after_begin,
        )
        row = ledger.get_session()
        if row is not None:
            service._check_existing_session(row)
        else:
            service._register_checked("config", config.digest, config.config_version)
            service._register_checked("script", script.digest, script.script_version)
            ledger.create_session(config, script, clock_fn())
            ledger.replace_conclusions(
                service.items, reason="initial", mono_ms=clock_fn(), archive=False
            )
        service._recover()
        return service

    @classmethod
    def open_existing(
        cls,
        db_path: str | Path,
        *,
        clock_fn: Callable[[], int] = monotonic_ms,
        logger: StructuredLogger | None = None,
        crash_after_begin: Callable[[Event], None] | None = None,
    ) -> SessionService:
        ledger = Ledger(db_path)
        row = ledger.get_session()
        if row is None:
            raise BusinessError(E_VERSION, "session metadata is missing", ["session"])
        config_raw = _decode_json(row["config_json"], "session.config_json")
        script_raw = _decode_json(row["script_json"], "session.script_json")
        config, config_errors = validate_session_config(
            config_raw, source_path="<ledger>"
        )
        if config_errors:
            raise BusinessError(
                E_VERSION, "stored session config is invalid", config_errors
            )
        script, script_errors = validate_peer_script(
            script_raw, config, source_path="<ledger>"
        )
        if script_errors:
            raise BusinessError(
                E_VERSION, "stored peer script is invalid", script_errors
            )
        if (
            config.digest != row["config_digest"]
            or script.digest != row["script_digest"]
        ):
            raise BusinessError(
                E_VERSION,
                "stored session digest mismatch",
                ["session.config_digest", "session.script_digest"],
            )
        if (
            config.config_version != row["config_version"]
            or script.script_version != row["script_version"]
        ):
            raise BusinessError(
                E_VERSION,
                "stored session version mismatch",
                ["session.config_version", "session.script_version"],
            )
        service = cls(
            ledger,
            config,
            script,
            clock_fn=clock_fn,
            logger=logger,
            crash_after_begin=crash_after_begin,
        )
        service._recover()
        return service

    def close(self) -> None:
        self.ledger.close()

    def _check_existing_session(self, row: Any) -> None:
        same_config_digest = self.config.digest == row["config_digest"]
        same_script_digest = self.script.digest == row["script_digest"]
        if not same_config_digest or not same_script_digest:
            raise version_error(
                ["session.config_digest", "session.script_digest"],
                "session content changed; use evolve to replace the whole conclusion set",
                current_config_version=int(row["config_version"]),
                current_script_version=int(row["script_version"]),
            )
        if self.config.config_version != int(
            row["config_version"]
        ) or self.script.script_version != int(row["script_version"]):
            raise version_error(
                ["session.config_version", "session.script_version"],
                "content is unchanged but the supplied version differs from the session ledger",
            )

    def _register_checked(self, kind: str, digest: str, version: int) -> None:
        existing = self.ledger.version_record(kind, digest)
        if existing is not None:
            if int(existing["version"]) != version:
                raise version_error(
                    [f"{kind}_version"],
                    "same content cannot be assigned a different version",
                    expected=int(existing["version"]),
                    actual=version,
                )
            return
        by_number = self.ledger.version_by_number(kind, version)
        if by_number is not None:
            raise version_error(
                [f"{kind}_version"],
                "version number is already assigned to different content",
                version=version,
            )
        if version <= self.ledger.max_version(kind):
            raise version_error(
                [f"{kind}_version"],
                "new content requires a strictly increasing version",
                version=version,
                current_max=self.ledger.max_version(kind),
            )
        self.ledger.register_version(kind, digest, version, self.clock_fn())

    def _recover(self) -> None:
        pending = self.ledger.mark_pending_unknown(
            "process interrupted before event commit"
        )
        events = self.ledger.events()
        try:
            recomputed = replay(
                events, self.config, self.script, capture_checkpoints=False
            )
        except BusinessError as exc:
            raise BusinessError(
                E_REPLAY,
                "complete ledger events cannot be replayed",
                exc.fields,
                exc.context,
            ) from exc
        persisted = self.ledger.conclusions()
        if persisted:
            differences = compare_items(persisted, recomputed.items)
            if differences:
                first = differences[0]
                basis = recomputed.items.get(first["field"].split(".", 1)[0])
                raise BusinessError(
                    E_REPLAY,
                    f"ledger conclusions disagree with replay at field {first['field']}",
                    [first["field"]],
                    {
                        "first_event_seq": 0
                        if basis is None
                        else basis.basis_event_seq,
                        "field": first["field"],
                        "expected": first["expected"],
                        "actual": first["actual"],
                    },
                )
        else:
            self.ledger.replace_conclusions(
                recomputed.items,
                reason="recovery_rebuild",
                mono_ms=self.clock_fn(),
                archive=False,
            )
        self.state = recomputed.state
        self.events = events
        self.items = recomputed.items
        unknown = tuple(sorted(set(pending) | set(self.ledger.unknown_event_seqs())))
        self.needs_rerun = bool(unknown)
        self.recovery_report = RecoveryReport(
            complete_event_seqs=tuple(event.event_seq for event in events),
            unknown_event_seqs=unknown,
            rerun_required=bool(unknown),
            exit_report_available=not bool(unknown),
            ledger_consistent=True,
            message=(
                "存在未完成事件，必须重跑；不能生成退出报告"
                if unknown
                else "仅依据完整账本事件恢复并通过逐项一致性校验"
            ),
        )
        if pending:
            self.logger.warning(
                "recovery_unknown_events",
                code="E_UNKNOWN_EVENT",
                event_seqs=list(unknown),
                rerun_required=True,
            )

    def ingest_event(self, event: Event) -> IngestionResult:
        if self.needs_rerun:
            raise BusinessError(
                E_STATE,
                "unknown events must be rerun before new events are accepted",
                ["rerun_required"],
                {
                    "unknown_event_seqs": list(
                        self.recovery_report.unknown_event_seqs
                        if self.recovery_report
                        else ()
                    )
                },
            )
        assigned = self.ledger.begin_event(
            event, self.config.config_version, self.script.script_version
        )
        if self.crash_after_begin is not None:
            self.crash_after_begin(assigned)
        return self._derive_pending_event(assigned)

    def _derive_pending_event(self, assigned: Event) -> IngestionResult:
        candidate_state = copy.deepcopy(self.state)
        try:
            affected = candidate_state.apply(assigned)
            prefix = self.events + (assigned,)
            candidate_items, materialized = apply_event_to_items(
                candidate_state,
                assigned,
                prefix,
                self.items,
                affected,
                self.config,
                self.script,
            )
        except BusinessError:
            self.ledger.mark_event_unknown(
                assigned.event_seq, "derivation rejected event"
            )
            raise
        self.ledger.commit_event(
            assigned,
            [candidate_items[key] for key in materialized],
            self.clock_fn(),
        )
        self.state = candidate_state
        self.events = prefix
        self.items = candidate_items
        result = IngestionResult(
            event=assigned,
            affected_items=materialized,
            item_revisions={
                key: item.revision for key, item in candidate_items.items()
            },
        )
        self.logger.info(
            "event_committed",
            event_seq=assigned.event_seq,
            event_type=assigned.event_type,
            affected_items=list(materialized),
            config_version=self.config.config_version,
            script_version=self.script.script_version,
        )
        return result

    def retry_unknown_events(self) -> tuple[IngestionResult, ...]:
        """Retry the exact, checksummed pending event payloads retained in the ledger."""

        if not self.needs_rerun:
            return ()
        results: list[IngestionResult] = []
        while True:
            unknown = self.ledger.unknown_event_seqs()
            if not unknown:
                break
            event_seq = unknown[0]
            expected = self.events[-1].event_seq + 1 if self.events else 1
            if event_seq != expected:
                raise BusinessError(
                    E_REPLAY,
                    "unknown events contain an unreplayable sequence gap",
                    ["event_seq"],
                    {"expected_event_seq": expected, "unknown_event_seq": event_seq},
                )
            assigned = self.ledger.unknown_to_pending(event_seq)
            results.append(self._derive_pending_event(assigned))
        self.needs_rerun = False
        self.recovery_report = RecoveryReport(
            complete_event_seqs=tuple(event.event_seq for event in self.events),
            unknown_event_seqs=(),
            rerun_required=False,
            exit_report_available=True,
            ledger_consistent=True,
            message="未知事件已按持久化载荷重新推导并通过提交",
        )
        self.logger.info(
            "unknown_events_retried",
            retried=[result.event.event_seq for result in results],
        )
        return tuple(results)

    def full_recompute(self):
        return replay(self.events, self.config, self.script, capture_checkpoints=True)

    def verify_incremental_equivalence(self) -> tuple[dict[str, Any], ...]:
        recomputed = self.full_recompute()
        differences = compare_items(recomputed.items, self.items)
        if differences:
            first = differences[0]
            raise BusinessError(
                E_REPLAY,
                f"incremental conclusions differ from full replay at {first['field']}",
                [first["field"]],
                first,
            )
        return tuple(differences)

    def evolve(
        self, new_config: SessionConfig, new_script: PeerScript
    ) -> ReplacementReport:
        old_items = dict(self.items)
        old_config_version = self.config.config_version
        old_script_version = self.script.script_version

        if new_config.digest == self.config.digest:
            record = self.ledger.version_record("config", new_config.digest)
            expected = None if record is None else int(record["version"])
            if expected is not None and new_config.config_version != expected:
                raise version_error(
                    ["config_version"], "unchanged config version must stay unchanged"
                )
        else:
            if new_config.config_version <= self.config.config_version:
                raise version_error(
                    ["config_version"],
                    "changed config requires a strictly increasing config_version",
                    current=self.config.config_version,
                    supplied=new_config.config_version,
                )
            existing = self.ledger.version_record("config", new_config.digest)
            if existing is not None:
                if int(existing["version"]) != new_config.config_version:
                    raise version_error(
                        ["config_version"],
                        "config content was already registered with another version",
                    )
            else:
                by_number = self.ledger.version_by_number(
                    "config", new_config.config_version
                )
                if by_number is not None:
                    raise version_error(
                        ["config_version"],
                        "config version number is already assigned to different content",
                        supplied=new_config.config_version,
                    )

        if new_script.digest == self.script.digest:
            record = self.ledger.version_record("script", new_script.digest)
            expected = None if record is None else int(record["version"])
            if expected is not None and new_script.script_version != expected:
                raise version_error(
                    ["script_version"],
                    "unchanged peer script version must stay unchanged",
                )
        else:
            if new_script.script_version <= self.script.script_version:
                raise version_error(
                    ["script_version"],
                    "changed peer script requires a strictly increasing script_version",
                    current=self.script.script_version,
                    supplied=new_script.script_version,
                )
            existing = self.ledger.version_record("script", new_script.digest)
            if existing is not None:
                if int(existing["version"]) != new_script.script_version:
                    raise version_error(
                        ["script_version"],
                        "script content was already registered with another version",
                    )
            else:
                by_number = self.ledger.version_by_number(
                    "script", new_script.script_version
                )
                if by_number is not None:
                    raise version_error(
                        ["script_version"],
                        "script version number is already assigned to different content",
                        supplied=new_script.script_version,
                    )

        recomputed = replay(
            self.events, new_config, new_script, capture_checkpoints=True
        )
        if new_config.digest != self.config.digest:
            self.ledger.register_version(
                "config", new_config.digest, new_config.config_version, self.clock_fn()
            )
        if new_script.digest != self.script.digest:
            self.ledger.register_version(
                "script", new_script.digest, new_script.script_version, self.clock_fn()
            )

        last_event_seq = self.events[-1].event_seq if self.events else 0
        self.ledger.replace_session(
            new_config,
            new_script,
            self.clock_fn(),
            current_event_seq=last_event_seq,
        )
        archived = self.ledger.replace_conclusions(
            recomputed.items,
            reason="version_evolution",
            mono_ms=self.clock_fn(),
            archive=True,
        )

        changes = _value_changes(old_items, recomputed.items)
        report = ReplacementReport(
            reason="version_evolution",
            affected_fields=tuple(changes),
            old_config_version=old_config_version,
            old_script_version=old_script_version,
            new_config_version=new_config.config_version,
            new_script_version=new_script.script_version,
            archived_fields=archived,
        )
        self.config = new_config
        self.script = new_script
        self.state = recomputed.state
        self.events = recomputed.events
        self.items = recomputed.items
        self.logger.info(
            "version_evolved",
            old_config_version=old_config_version,
            old_script_version=old_script_version,
            new_config_version=new_config.config_version,
            new_script_version=new_script.script_version,
            affected_fields=[change.field for change in changes],
        )
        return report

    def snapshot_payload(self, name: str) -> dict[str, Any]:
        recomputed = self.full_recompute()
        differences = compare_items(recomputed.items, self.items)
        if differences:
            raise BusinessError(
                E_REPLAY,
                "cannot snapshot inconsistent incremental conclusions",
                [differences[0]["field"]],
            )
        return {
            "name": name,
            "created_mono_ms": self.clock_fn(),
            "config_version": self.config.config_version,
            "script_version": self.script.script_version,
            "config_digest": self.config.digest,
            "script_digest": self.script.digest,
            "event_count": len(self.events),
            "state": state_snapshot(self.state),
            "conclusions": {key: self.items[key].as_dict() for key in ITEM_ORDER},
            "events": [event.as_dict() for event in self.events],
            "replay_points": list(recomputed.checkpoints),
            "config": self.config.raw,
            "script": self.script.raw,
        }

    def save_snapshot(self, name: str) -> int:
        snapshot_id = self.ledger.save_snapshot(self.snapshot_payload(name))
        self.logger.info(
            "snapshot_saved",
            snapshot_id=snapshot_id,
            name=name,
            event_count=len(self.events),
            config_version=self.config.config_version,
            script_version=self.script.script_version,
        )
        return snapshot_id

    def verify_snapshot(self, snapshot_id: int) -> dict[str, Any]:
        from .snapshot import verify_snapshot_row

        row = self.ledger.snapshot(snapshot_id)
        if row is None:
            raise version_error(
                ["snapshot"], "snapshot version is missing", snapshot_id=snapshot_id
            )
        result = verify_snapshot_row(row, self.ledger)
        result["matches_current"] = (
            result["config_version"] == self.config.config_version
            and result["script_version"] == self.script.script_version
            and row["config_digest"] == self.config.digest
            and row["script_digest"] == self.script.digest
        )
        result["current_version"] = (
            f"C{self.config.config_version}/S{self.script.script_version}"
        )
        result["whole_set_replacement_required"] = not result["matches_current"]
        self.logger.info("snapshot_verified", **result)
        return result

    def restore_snapshot(
        self, snapshot_id: int, *, inject_missing_version: bool = False
    ) -> ReplacementReport:
        from .snapshot import restore_snapshot_row

        row = self.ledger.snapshot(snapshot_id)
        if row is None:
            raise version_error(
                ["snapshot"], "snapshot version is missing", snapshot_id=snapshot_id
            )
        if inject_missing_version:
            raise version_error(
                ["snapshot.config_version", "snapshot.script_version"],
                "snapshot version is missing (injected parameter)",
                snapshot_id=snapshot_id,
            )
        return restore_snapshot_row(self, row)


def _decode_json(text: str, field: str) -> Any:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise version_error([field], f"stored JSON is invalid: {exc}") from exc


def _value_changes(
    old_items: dict[str, ItemResult], new_items: dict[str, ItemResult]
) -> list[FieldChange]:
    old_flat: dict[str, Any] = {}
    new_flat: dict[str, Any] = {}
    for key, item in old_items.items():
        for path, value in flatten_values(f"items.{key}", item.value):
            old_flat[path] = value
        old_flat[f"items.{key}.revision"] = item.revision
    for key, item in new_items.items():
        for path, value in flatten_values(f"items.{key}", item.value):
            new_flat[path] = value
        new_flat[f"items.{key}.revision"] = item.revision
    changes: list[FieldChange] = []
    for field in sorted(set(old_flat) | set(new_flat)):
        old_value = old_flat.get(field)
        new_value = new_flat.get(field)
        if canonical_json(old_value) != canonical_json(new_value):
            changes.append(FieldChange(field, old_value, new_value))
    return changes
