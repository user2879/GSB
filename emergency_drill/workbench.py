"""High-level operations: plan edits, runs, snapshots, recovery, and replay."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .engine import Engine
from .errors import (
    E_LEDGER,
    E_PLAN,
    E_REPLAY,
    E_SNAPSHOT,
    BusinessError,
    PlanValidationError,
    SimulatedCrash,
)
from .logging_utils import StructuredLogger
from .models import (
    ALL_VIEW_STATUSES,
    EventDefinition,
    EventRuntime,
    Plan,
    ReplayReport,
    RecoveryReport,
    SessionView,
    canonical_json,
    digest_json,
)
from .plan import (
    load_plan,
    validate_event_object,
    validate_plan_object,
    with_plan_version,
)
from .repository import Repository


SNAPSHOT_FORMAT = "emergency-drill-snapshot"
SNAPSHOT_FORMAT_VERSION = 1


class Workbench:
    def __init__(
        self,
        db_path: str | Path = "data/session.db",
        snapshots_dir: str | Path = "data/snapshots",
        log_path: str | Path = "data/logs/drill.jsonl",
        *,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or StructuredLogger(log_path)
        self.repository = Repository(self.db_path)

    @staticmethod
    def _plan_from_source(source: Plan | str | Path | dict[str, Any]) -> tuple[Plan, dict[str, Any]]:
        if isinstance(source, Plan):
            return source, source.to_dict()
        if isinstance(source, (str, Path)):
            path = Path(source)
            plan = load_plan(path)
            return plan, json.loads(path.read_text(encoding="utf-8"))
        if isinstance(source, dict):
            plan = validate_plan_object(source)
            return plan, source
        raise PlanValidationError(
            "预案来源必须是 Plan、JSON 对象或本地路径",
            code=E_PLAN,
            field_path="$source",
            errors=[{"message": "不支持的预案来源", "field_path": "$source", "stage_seq": None}],
        )

    def load_plan_file(self, path: str | Path) -> Plan:
        try:
            return load_plan(path)
        except PlanValidationError as exc:
            self.logger.error(exc, event="plan_rejected", path=str(path))
            raise

    def create_session(
        self,
        session_id: str,
        plan_source: Plan | str | Path | dict[str, Any],
        *,
        injected_events: list[EventDefinition | dict[str, Any]] | None = None,
        use_plan_events: bool = True,
    ) -> SessionView:
        try:
            plan, _ = self._plan_from_source(plan_source)
            event_items: list[tuple[EventDefinition, str]] = []
            if use_plan_events:
                event_items.extend((event, "plan") for event in plan.events)
            seen_ids = {event.event_id for event, _ in event_items}
            for raw_event in injected_events or []:
                event = (
                    raw_event
                    if isinstance(raw_event, EventDefinition)
                    else validate_event_object(raw_event, plan)
                )
                if event.stage_seq not in plan.stage_seqs:
                    raise PlanValidationError(
                        "注入事件作用阶段不存在",
                        code=E_PLAN,
                        stage_seq=event.stage_seq,
                        field_path="event.stage_seq",
                    )
                if event.event_id in seen_ids:
                    raise PlanValidationError(
                        f"event_id 重复: {event.event_id}",
                        code=E_PLAN,
                        stage_seq=event.stage_seq,
                        field_path="event.event_id",
                    )
                seen_ids.add(event.event_id)
                event_items.append((event, "manual"))
            self.repository.create_session(session_id, plan, event_items)
        except PlanValidationError as exc:
            self.logger.error(exc, event="session_rejected", session_id=session_id)
            raise
        self.logger.emit(
            "session_created",
            session_id=session_id,
            plan_version=plan.version,
            logical_clock_ms=0,
            event_count=len(event_items),
            plan_digest=plan.digest,
        )
        return self.repository.get_view(session_id)

    @staticmethod
    def _stage_changed_fields(old: Any, new: Any) -> list[str]:
        old_dict = old.to_dict()
        new_dict = new.to_dict()
        return [key for key in new_dict if old_dict.get(key) != new_dict.get(key)]

    def diff_plans(self, old_plan: Plan, new_plan: Plan) -> dict[str, Any]:
        old_stages = {stage.seq: stage for stage in old_plan.stages}
        new_stages = {stage.seq: stage for stage in new_plan.stages}
        added = sorted(set(new_stages) - set(old_stages))
        removed = sorted(set(old_stages) - set(new_stages))
        changed = []
        for seq in sorted(set(old_stages) & set(new_stages)):
            fields = self._stage_changed_fields(old_stages[seq], new_stages[seq])
            if fields:
                changed.append({"seq": seq, "fields": fields})
        old_events = {event.event_id: event for event in old_plan.events}
        new_events = {event.event_id: event for event in new_plan.events}
        event_added = sorted(set(new_events) - set(old_events))
        event_removed = sorted(set(old_events) - set(new_events))
        event_changed = []
        for event_id in sorted(set(old_events) & set(new_events)):
            fields = [
                key
                for key in ("type", "offset_ms", "stage_seq")
                if getattr(old_events[event_id], key) != getattr(new_events[event_id], key)
            ]
            if fields:
                event_changed.append(
                    {
                        "event_id": event_id,
                        "stage_seq": new_events[event_id].stage_seq,
                        "fields": fields,
                    }
                )
        plan_fields = [
            key
            for key in ("plan_id", "name", "failure_policy")
            if getattr(old_plan, key) != getattr(new_plan, key)
        ]
        return {
            "plan_fields": plan_fields,
            "added_stage_seqs": added,
            "removed_stage_seqs": removed,
            "changed_stages": changed,
            "added_event_ids": event_added,
            "removed_event_ids": event_removed,
            "changed_events": event_changed,
        }

    def publish_plan(
        self,
        session_id: str,
        plan_source: Plan | str | Path | dict[str, Any],
        *,
        auto_bump: bool = True,
    ) -> dict[str, Any]:
        session = self.repository.require_session(session_id)
        old_plan = self.repository.get_plan(session_id, session.plan_version)
        try:
            new_plan, raw = self._plan_from_source(plan_source)
            max_version = self.repository.max_plan_version(session_id)
            if auto_bump:
                target_version = max_version + 1
                if isinstance(plan_source, Plan):
                    bumped = new_plan.to_dict()
                    bumped["version"] = target_version
                    new_plan = validate_plan_object(bumped)
                else:
                    new_plan = validate_plan_object(with_plan_version(raw, target_version))
            elif new_plan.version <= max_version:
                raise PlanValidationError(
                    "预案版本必须单调递增",
                    code=E_PLAN,
                    field_path="version",
                    errors=[{"message": "预案版本必须单调递增", "field_path": "version", "stage_seq": None}],
                )
            diff = self.diff_plans(old_plan, new_plan)
            generation = self.repository.replace_plan_version(
                session_id,
                new_plan,
                diff=diff,
            )
        except PlanValidationError as exc:
            self.logger.error(
                exc,
                event="plan_refresh_rejected",
                session_id=session_id,
                logical_clock_ms=session.clock_ms,
            )
            raise
        self.logger.emit(
            "plan_refreshed",
            session_id=session_id,
            plan_version=new_plan.version,
            logical_clock_ms=0,
            old_plan_version=old_plan.version,
            stale_plan_version=old_plan.version,
            generation=generation,
            diff=diff,
        )
        return {
            "session_id": session_id,
            "old_plan_version": old_plan.version,
            "new_plan_version": new_plan.version,
            "generation": generation,
            "stale_plan_version": old_plan.version,
            "stale_stage_states_excluded": True,
            "diff": diff,
        }

    def inject_event(
        self,
        session_id: str,
        event_source: EventDefinition | dict[str, Any],
    ) -> EventRuntime:
        view = self.repository.get_view(session_id)
        try:
            event = (
                event_source
                if isinstance(event_source, EventDefinition)
                else validate_event_object(event_source, view.plan)
            )
            if event.stage_seq not in view.plan.stage_seqs:
                raise PlanValidationError(
                    "注入事件作用阶段不存在",
                    code=E_PLAN,
                    stage_seq=event.stage_seq,
                    field_path="event.stage_seq",
                )
            runtime = self.repository.inject_event(
                session_id,
                event,
                logical_ms=view.session.clock_ms,
            )
        except PlanValidationError as exc:
            self.logger.error(
                exc,
                event="event_injection_rejected",
                session_id=session_id,
                logical_clock_ms=view.session.clock_ms,
            )
            raise
        self.logger.emit(
            "event_injected",
            session_id=session_id,
            plan_version=view.plan.version,
            stage_seq=event.stage_seq,
            logical_clock_ms=view.session.clock_ms,
            event_id=event.event_id,
            event_type=event.type,
            offset_ms=event.offset_ms,
            injection_seq=runtime.injection_seq,
        )
        return runtime

    def run_session(
        self,
        session_id: str,
        *,
        abort_after_stage: int | None = None,
        crash_after_stage: int | None = None,
        hard_crash: bool = False,
        abort_check=None,
        stage_callback=None,
        engine: Engine | None = None,
    ) -> dict[str, Any]:
        runner = engine or Engine(self.repository, session_id, self.logger)
        return runner.run(
            abort_after_stage=abort_after_stage,
            crash_after_stage=crash_after_stage,
            hard_crash=hard_crash,
            abort_check=abort_check,
            stage_callback=stage_callback,
        )

    def new_engine(self, session_id: str) -> Engine:
        return Engine(self.repository, session_id, self.logger)

    def _ledger_summary(self, view: SessionView) -> dict[str, Any]:
        entries = self.repository.ledger_rows(
            self.session_id_safe(view), view.session.generation
        )
        transitions = [
            entry
            for entry in entries
            if entry.is_complete and entry.kind == "stage_transition"
        ]
        counts: dict[int, int] = {}
        for entry in transitions:
            if entry.stage_seq is not None:
                counts[entry.stage_seq] = counts.get(entry.stage_seq, 0) + 1
        stage_summaries = []
        for stage in view.stages:
            stage_summaries.append(
                {
                    "seq": stage.seq,
                    "status": stage.status,
                    "skip_reason": stage.skip_reason,
                    "business_code": stage.business_code,
                    "logical_ms": stage.updated_ms,
                    "migration_count": counts.get(stage.seq, 0),
                }
            )
        return {
            "migration_count": len(transitions),
            "entry_count": len([entry for entry in entries if entry.is_complete]),
            "stages": stage_summaries,
        }

    @staticmethod
    def session_id_safe(view: SessionView) -> str:
        return view.session.session_id

    def save_snapshot(self, session_id: str) -> dict[str, Any]:
        view = self.repository.get_view(session_id)
        entries = [
            entry.to_dict()
            for entry in self.repository.ledger_rows(session_id, view.session.generation)
            if entry.is_complete
        ]
        summary = self._ledger_summary(view)
        n = self.repository.next_snapshot_number()
        snapshot_id = f"snap-{n:04d}"
        payload_without_digest: dict[str, Any] = {
            "format": SNAPSHOT_FORMAT,
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "plan_version": view.plan.version,
            "plan_digest": view.plan.digest,
            "plan": view.plan.to_dict(),
            "events": [event.to_dict() for event in view.events],
            "phase_states": [stage.to_dict() for stage in view.stages],
            "completed_stages": [
                stage.seq for stage in view.stages if stage.status == "succeeded"
            ],
            "logical_clock_ms": view.session.clock_ms,
            "ledger_trace": entries,
            "ledger_summary": summary,
        }
        content_digest = digest_json(payload_without_digest)
        payload = dict(payload_without_digest)
        payload["content_digest"] = content_digest
        safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "session"
        destination = self.snapshots_dir / f"{safe_session}-{snapshot_id}.json"
        while destination.exists():
            n += 1
            snapshot_id = f"snap-{n:04d}"
            payload["snapshot_id"] = snapshot_id
            payload_without_digest["snapshot_id"] = snapshot_id
            content_digest = digest_json(payload_without_digest)
            payload["content_digest"] = content_digest
            destination = self.snapshots_dir / f"{safe_session}-{snapshot_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        self.repository.register_snapshot(
            snapshot_id=snapshot_id,
            session_id=session_id,
            file_path=str(destination.resolve()),
            plan_version=view.plan.version,
            content_digest=content_digest,
            created_ms=view.session.clock_ms,
            generation=view.session.generation,
        )
        self.logger.emit(
            "snapshot_saved",
            session_id=session_id,
            plan_version=view.plan.version,
            logical_clock_ms=view.session.clock_ms,
            snapshot_id=snapshot_id,
            content_digest=content_digest,
            path=str(destination.resolve()),
        )
        return {
            "snapshot_id": snapshot_id,
            "file_path": str(destination.resolve()),
            "plan_version": view.plan.version,
            "content_digest": content_digest,
            "logical_clock_ms": view.session.clock_ms,
            "completed_stages": payload["completed_stages"],
            "ledger_summary": summary,
        }

    @staticmethod
    def _snapshot_stage_locator(raw: Any) -> int | None:
        if not isinstance(raw, dict):
            return None
        phase_states = raw.get("phase_states")
        if isinstance(phase_states, list):
            seqs = [
                item.get("seq")
                for item in phase_states
                if isinstance(item, dict) and type(item.get("seq")) is int
            ]
            if seqs:
                return int(min(seqs))
        events = raw.get("events")
        if isinstance(events, list):
            seqs = [
                item.get("stage_seq")
                for item in events
                if isinstance(item, dict) and type(item.get("stage_seq")) is int
            ]
            if seqs:
                return int(min(seqs))
        return None

    def _snapshot_error(
        self,
        message: str,
        *,
        field_path: str,
        raw: Any = None,
        logical_ms: int = 0,
    ) -> BusinessError:
        error = BusinessError(
            E_SNAPSHOT,
            message,
            stage_seq=self._snapshot_stage_locator(raw),
            field_path=field_path,
        )
        self.logger.error(error, event="snapshot_rejected", logical_clock_ms=logical_ms)
        return error

    def load_snapshot(self, path: str | Path) -> dict[str, Any]:
        snapshot_path = Path(path)
        try:
            raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise self._snapshot_error(
                f"快照无法读取或解析: {exc}", field_path="$file"
            ) from exc
        if not isinstance(raw, dict):
            raise self._snapshot_error("快照根节点必须是 JSON 对象", field_path="$", raw=raw)
        required = [
            "format",
            "format_version",
            "snapshot_id",
            "session_id",
            "plan_version",
            "plan_digest",
            "plan",
            "events",
            "phase_states",
            "completed_stages",
            "logical_clock_ms",
            "ledger_trace",
            "ledger_summary",
            "content_digest",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise self._snapshot_error(
                "快照字段不完整: " + ", ".join(missing),
                field_path=missing[0],
                raw=raw,
            )
        if raw["format"] != SNAPSHOT_FORMAT or raw["format_version"] != SNAPSHOT_FORMAT_VERSION:
            raise self._snapshot_error(
                "不支持的快照格式或版本", field_path="format_version", raw=raw
            )
        digest_payload = {key: value for key, value in raw.items() if key != "content_digest"}
        calculated = digest_json(digest_payload)
        if not isinstance(raw["content_digest"], str) or calculated != raw["content_digest"]:
            error = BusinessError(
                E_SNAPSHOT,
                "快照内容摘要不一致，拒绝打开",
                stage_seq=self._snapshot_stage_locator(raw),
                field_path="content_digest",
                details={"expected": raw["content_digest"], "calculated": calculated},
            )
            self.logger.error(
                error,
                event="snapshot_digest_mismatch",
                logical_clock_ms=int(raw.get("logical_clock_ms", 0))
                if type(raw.get("logical_clock_ms")) is int
                else 0,
                snapshot_id=raw.get("snapshot_id"),
            )
            raise error
        if (
            not isinstance(raw["snapshot_id"], str)
            or not isinstance(raw["session_id"], str)
            or type(raw["plan_version"]) is not int
            or not isinstance(raw["plan_digest"], str)
            or type(raw["logical_clock_ms"]) is not int
            or raw["logical_clock_ms"] < 0
        ):
            raise self._snapshot_error(
                "快照标识、版本或逻辑时钟字段类型错误",
                field_path="logical_clock_ms",
                raw=raw,
            )
        try:
            plan = validate_plan_object(raw["plan"])
        except PlanValidationError as exc:
            raise self._snapshot_error(
                f"快照内预案不合法: {exc.message}",
                field_path=f"plan.{exc.field_path or '$'}",
                raw=raw,
            ) from exc
        if plan.version != raw["plan_version"] or plan.digest != raw["plan_digest"]:
            raise self._snapshot_error(
                "快照预案版本或摘要与内容不一致",
                field_path="plan_digest",
                raw=raw,
            )
        if not isinstance(raw["phase_states"], list) or not isinstance(raw["events"], list):
            raise self._snapshot_error(
                "快照阶段状态或事件序列类型错误", field_path="phase_states", raw=raw
            )
        seen_stage_seqs: set[int] = set()
        for index, stage in enumerate(raw["phase_states"]):
            if (
                not isinstance(stage, dict)
                or stage.get("status") not in ALL_VIEW_STATUSES
                or type(stage.get("seq")) is not int
                or stage.get("seq") not in plan.stage_seqs
                or stage.get("seq") in seen_stage_seqs
            ):
                raise self._snapshot_error(
                    "快照阶段状态字段不合法",
                    field_path=f"phase_states[{index}].status",
                    raw=raw,
                )
            seen_stage_seqs.add(stage["seq"])
        if seen_stage_seqs != set(plan.stage_seqs):
            raise self._snapshot_error(
                "快照阶段状态集合与预案不一致",
                field_path="phase_states",
                raw=raw,
            )
        seen_injection_seqs: set[int] = set()
        for index, event in enumerate(raw["events"]):
            if (
                not isinstance(event, dict)
                or type(event.get("injection_seq")) is not int
                or event.get("injection_seq") in seen_injection_seqs
            ):
                raise self._snapshot_error(
                    "快照事件注入序号不合法或重复",
                    field_path=f"events[{index}].injection_seq",
                    raw=raw,
                )
            seen_injection_seqs.add(event["injection_seq"])
            try:
                validate_event_object(event, plan)
            except PlanValidationError as exc:
                raise self._snapshot_error(
                    f"快照事件序列不合法: {exc.message}",
                    field_path=f"events[{index}].{exc.field_path or '$'}",
                    raw=raw,
                ) from exc
        completed = raw["completed_stages"]
        if (
            not isinstance(completed, list)
            or any(type(seq) is not int or seq not in plan.stage_seqs for seq in completed)
            or len(completed) != len(set(completed))
        ):
            raise self._snapshot_error(
                "已完成阶段集合不合法",
                field_path="completed_stages",
                raw=raw,
            )
        if not isinstance(raw["ledger_trace"], list):
            raise self._snapshot_error(
                "账本轨迹结构不合法",
                field_path="ledger_trace",
                raw=raw,
            )
        seen_ledger_seqs: set[int] = set()
        for index, entry in enumerate(raw["ledger_trace"]):
            if (
                not isinstance(entry, dict)
                or type(entry.get("seq")) is not int
                or entry.get("seq") in seen_ledger_seqs
                or not isinstance(entry.get("kind"), str)
                or type(entry.get("logical_ms")) is not int
                or entry.get("logical_ms") < 0
                or type(entry.get("plan_version")) is not int
                or not isinstance(entry.get("details", {}), dict)
                or not isinstance(entry.get("is_complete", True), bool)
                or (
                    entry.get("stage_seq") is not None
                    and type(entry.get("stage_seq")) is not int
                )
                or (
                    entry.get("from_status") is not None
                    and not isinstance(entry.get("from_status"), str)
                )
                or (
                    entry.get("to_status") is not None
                    and not isinstance(entry.get("to_status"), str)
                )
            ):
                raise self._snapshot_error(
                    "账本轨迹记录字段不合法",
                    field_path=f"ledger_trace[{index}]",
                    raw=raw,
                )
            seen_ledger_seqs.add(entry["seq"])
        summary = raw["ledger_summary"]
        if (
            not isinstance(summary, dict)
            or not isinstance(summary.get("stages"), list)
            or type(summary.get("migration_count")) is not int
        ):
            raise self._snapshot_error(
                "账本摘要结构不合法",
                field_path="ledger_summary",
                raw=raw,
            )
        for index, stage_summary in enumerate(summary["stages"]):
            if (
                not isinstance(stage_summary, dict)
                or type(stage_summary.get("seq")) is not int
                or stage_summary.get("seq") not in plan.stage_seqs
                or type(stage_summary.get("migration_count")) is not int
                or stage_summary.get("migration_count") < 0
            ):
                raise self._snapshot_error(
                    "账本逐阶段摘要不合法",
                    field_path=f"ledger_summary.stages[{index}]",
                    raw=raw,
                )
        return raw

    def restore_snapshot(self, session_id: str, snapshot_path: str | Path) -> dict[str, Any]:
        current = self.repository.get_view(session_id)
        snapshot = self.load_snapshot(snapshot_path)
        if snapshot["session_id"] != session_id:
            raise self._snapshot_error(
                "快照不属于当前会话",
                field_path="session_id",
                raw=snapshot,
                logical_ms=int(snapshot["logical_clock_ms"]),
            )
        version_matches = current.plan.version == snapshot["plan_version"]
        digest_matches = current.plan.digest == snapshot["plan_digest"]
        generation = self.repository.replace_from_snapshot(
            session_id,
            snapshot,
            previous_plan_version=current.plan.version,
            previous_plan_digest=current.plan.digest,
        )
        self.repository.mark_snapshots_read_only(session_id)
        restored = self.repository.get_view(session_id)
        snapshots = self.list_snapshots(session_id)
        self.logger.emit(
            "snapshot_restored",
            session_id=session_id,
            plan_version=restored.plan.version,
            logical_clock_ms=restored.session.clock_ms,
            snapshot_id=snapshot["snapshot_id"],
            previous_plan_version=current.plan.version,
            plan_version_matches=version_matches,
            plan_digest_matches=digest_matches,
            whole_replacement=True,
            generation=generation,
        )
        return {
            "session_id": session_id,
            "snapshot_id": snapshot["snapshot_id"],
            "previous_plan_version": current.plan.version,
            "restored_plan_version": restored.plan.version,
            "plan_version_matches": version_matches,
            "plan_digest_matches": digest_matches,
            "whole_replacement": True,
            "incremental_merge": False,
            "restored_stage_count": len(restored.stages),
            "restored_event_count": len(restored.events),
            "generation": generation,
            "snapshots": snapshots,
        }

    def _replay_error(
        self,
        message: str,
        *,
        session_id: str,
        stage_seq: int | None,
        field_path: str,
        logical_ms: int,
        details: dict[str, Any] | None = None,
    ) -> BusinessError:
        error = BusinessError(
            E_REPLAY,
            message,
            stage_seq=stage_seq,
            field_path=field_path,
            details=details,
        )
        self.logger.error(
            error,
            event="replay_mismatch",
            session_id=session_id,
            logical_clock_ms=logical_ms,
        )
        return error

    @staticmethod
    def _event_comparison_fields(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "type": event.get("type"),
            "offset_ms": event.get("offset_ms"),
            "stage_seq": event.get("stage_seq"),
        }

    @staticmethod
    def _first_plan_difference_stage(expected_plan: Plan, actual_plan: Plan) -> int:
        expected = {stage.seq: stage.to_dict() for stage in expected_plan.stages}
        actual = {stage.seq: stage.to_dict() for stage in actual_plan.stages}
        differences = set(expected) ^ set(actual)
        for seq in sorted(set(expected) & set(actual)):
            if expected[seq] != actual[seq]:
                differences.add(seq)
        if differences:
            return min(differences)
        all_stages = sorted(expected) or sorted(actual) or [1]
        return all_stages[0]

    def replay_snapshot(
        self,
        session_id: str,
        snapshot_path: str | Path,
    ) -> ReplayReport:
        current = self.repository.get_view(session_id)
        snapshot = self.load_snapshot(snapshot_path)
        if snapshot["session_id"] != session_id:
            raise self._snapshot_error(
                "快照不属于当前会话",
                field_path="session_id",
                raw=snapshot,
                logical_ms=int(snapshot["logical_clock_ms"]),
            )
        plan = validate_plan_object(snapshot["plan"])
        if current.plan.version != snapshot["plan_version"]:
            first_stage = self._first_plan_difference_stage(plan, current.plan)
            raise self._replay_error(
                "预案版本不一致，无法复现",
                session_id=session_id,
                stage_seq=first_stage,
                field_path="plan_version",
                logical_ms=current.session.clock_ms,
                details={
                    "expected": snapshot["plan_version"],
                    "actual": current.plan.version,
                },
            )
        if current.plan.digest != snapshot["plan_digest"]:
            first_stage = self._first_plan_difference_stage(plan, current.plan)
            raise self._replay_error(
                "预案内容摘要不一致，无法复现",
                session_id=session_id,
                stage_seq=first_stage,
                field_path="plan_digest",
                logical_ms=current.session.clock_ms,
                details={
                    "expected": snapshot["plan_digest"],
                    "actual": current.plan.digest,
                },
            )

        current_events = sorted(current.events, key=lambda event: event.injection_seq)
        saved_events = sorted(
            snapshot["events"], key=lambda event: int(event["injection_seq"])
        )
        current_specs = [event.spec_dict() for event in current_events]
        saved_specs = [self._event_comparison_fields(event) for event in saved_events]
        if len(current_specs) != len(saved_specs):
            common_length = min(len(current_specs), len(saved_specs))
            if len(current_specs) > len(saved_specs):
                stage_locator = int(current_specs[common_length].get("stage_seq", 1))
            else:
                stage_locator = int(saved_specs[common_length].get("stage_seq", 1))
            raise self._replay_error(
                "注入事件序列长度不一致，无法复现",
                session_id=session_id,
                stage_seq=stage_locator,
                field_path="events.length",
                logical_ms=current.session.clock_ms,
                details={"expected": len(saved_specs), "actual": len(current_specs)},
            )
        for index, (actual, expected) in enumerate(zip(current_specs, saved_specs)):
            for field in ("event_id", "type", "offset_ms", "stage_seq"):
                if actual.get(field) != expected.get(field):
                    stage_locator = expected.get("stage_seq") if type(expected.get("stage_seq")) is int else 1
                    raise self._replay_error(
                        "注入事件序列不一致，无法复现",
                        session_id=session_id,
                        stage_seq=stage_locator,
                        field_path=f"events[{index}].{field}",
                        logical_ms=current.session.clock_ms,
                        details={"expected": expected.get(field), "actual": actual.get(field)},
                    )

        current_stage_map = {stage.seq: stage for stage in current.stages}
        saved_stage_map = {
            int(stage["seq"]): stage for stage in snapshot["phase_states"]
        }
        for stage_seq in sorted(stage.seq for stage in plan.stages):
            actual_stage = current_stage_map.get(stage_seq)
            expected_stage = saved_stage_map.get(stage_seq)
            if actual_stage is None or expected_stage is None:
                raise self._replay_error(
                    "上游阶段状态集合不一致，无法复现",
                    session_id=session_id,
                    stage_seq=stage_seq,
                    field_path=f"phase_states.{stage_seq}",
                    logical_ms=current.session.clock_ms,
                )
            for field, expected_value, actual_value in (
                ("status", expected_stage.get("status"), actual_stage.status),
                (
                    "skip_reason",
                    expected_stage.get("skip_reason"),
                    actual_stage.skip_reason,
                ),
                (
                    "business_code",
                    expected_stage.get("business_code"),
                    actual_stage.business_code,
                ),
                (
                    "skip_from_stage_seq",
                    expected_stage.get("skip_from_stage_seq"),
                    actual_stage.skip_from_stage_seq,
                ),
                (
                    "unknown_reason",
                    expected_stage.get("unknown_reason"),
                    actual_stage.unknown_reason,
                ),
                (
                    "updated_ms",
                    expected_stage.get("updated_ms"),
                    actual_stage.updated_ms,
                ),
            ):
                if expected_value != actual_value:
                    raise self._replay_error(
                        "上游阶段状态不一致，无法复现",
                        session_id=session_id,
                        stage_seq=stage_seq,
                        field_path=f"phase_states.{stage_seq}.{field}",
                        logical_ms=current.session.clock_ms,
                        details={"expected": expected_value, "actual": actual_value},
                    )

        with tempfile.TemporaryDirectory(prefix="drill-replay-") as temp_dir:
            replay_repository = Repository(Path(temp_dir) / "replay.db")
            replay_events = [validate_event_object(event, plan) for event in saved_events]
            replay_repository.create_session(
                "replay",
                plan,
                [(event, "snapshot") for event in replay_events],
            )
            replay_engine = Engine(
                replay_repository,
                "replay",
                self.logger,
            )
            replay_summary = replay_engine.run()

        replay_stage_map = {stage["seq"]: stage for stage in replay_summary["stages"]}
        saved_summary_stages = {
            int(stage["seq"]): stage
            for stage in snapshot["ledger_summary"]["stages"]
        }
        comparisons: list[dict[str, Any]] = []
        first_mismatch: dict[str, Any] | None = None
        for stage_seq in sorted(stage.seq for stage in plan.stages):
            expected_state = saved_stage_map[stage_seq]
            actual_state = replay_stage_map[stage_seq]
            expected_summary = saved_summary_stages[stage_seq]
            for field, expected, actual in (
                ("status", expected_state.get("status"), actual_state.get("status")),
                (
                    "skip_reason",
                    expected_state.get("skip_reason"),
                    actual_state.get("skip_reason"),
                ),
                (
                    "migration_count",
                    int(expected_summary.get("migration_count", 0)),
                    int(actual_state.get("migration_count", 0)),
                ),
            ):
                comparison = {
                    "stage_seq": stage_seq,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                    "equal": expected == actual,
                }
                comparisons.append(comparison)
                if expected != actual and first_mismatch is None:
                    first_mismatch = comparison
        if first_mismatch is not None:
            stage_seq = int(first_mismatch["stage_seq"])
            field = str(first_mismatch["field"])
            raise self._replay_error(
                "历史演练不可复现，首个不一致字段已定位",
                session_id=session_id,
                stage_seq=stage_seq,
                field_path=f"stage_states.{stage_seq}.{field}",
                logical_ms=current.session.clock_ms,
                details={
                    "first_mismatch": first_mismatch,
                    "comparisons": comparisons,
                },
            )
        report = ReplayReport(
            session_id=session_id,
            snapshot_id=snapshot["snapshot_id"],
            reproducible=True,
            compared_stage_count=len(plan.stages),
            comparisons=comparisons,
        )
        self.logger.emit(
            "replay_verified",
            session_id=session_id,
            plan_version=plan.version,
            logical_clock_ms=current.session.clock_ms,
            snapshot_id=snapshot["snapshot_id"],
            compared_stage_count=len(plan.stages),
            reproducible=True,
        )
        return report

    def recover(self, session_id: str) -> RecoveryReport:
        view = self.repository.get_view(session_id)
        entries = self.repository.ledger_rows(session_id, view.session.generation)
        tail_entries: list[Any] = []
        index = len(entries) - 1
        while index >= 0 and not entries[index].is_complete:
            tail_entries.insert(0, entries[index])
            index -= 1
        if any(not entry.is_complete for entry in entries[: index + 1]):
            raise BusinessError(
                E_LEDGER,
                "账本中部存在不完整记录，超出尾部恢复规则",
                field_path="ledger.is_complete",
            )
        tail_seqs = [entry.seq for entry in tail_entries]
        tail_action = "未发现不完整尾记录"
        if tail_seqs:
            tail_action = (
                f"隔离并丢弃不完整尾记录 {tail_seqs}；不计入迁移、不参与状态恢复"
            )
            self.repository.discard_incomplete_tail(
                session_id,
                view.session.generation,
                tail_seqs=tail_seqs,
                logical_ms=view.session.clock_ms,
                reason=tail_action,
            )
            self.logger.emit(
                "incomplete_tail_discarded",
                session_id=session_id,
                plan_version=view.plan.version,
                logical_clock_ms=view.session.clock_ms,
                business_code=E_LEDGER,
                tail_seqs=tail_seqs,
                reason=tail_action,
            )
            entries = self.repository.ledger_rows(
                session_id, view.session.generation
            )

        reconstructed: dict[int, dict[str, Any]] = {
            stage.seq: {
                "status": "pending",
                "reason": None,
                "code": None,
                "skip_from_stage_seq": None,
            }
            for stage in view.stages
        }
        for entry in entries:
            if entry.kind != "stage_transition" or entry.stage_seq is None:
                continue
            if entry.stage_seq not in reconstructed:
                raise BusinessError(
                    E_LEDGER,
                    "账本引用了不存在的阶段",
                    stage_seq=entry.stage_seq,
                    field_path="ledger.stage_seq",
                )
            target = reconstructed[entry.stage_seq]
            if entry.from_status != target["status"]:
                raise BusinessError(
                    E_LEDGER,
                    "账本状态链断裂",
                    stage_seq=entry.stage_seq,
                    field_path="ledger.from_status",
                    details={
                        "expected": target["status"],
                        "actual": entry.from_status,
                        "seq": entry.seq,
                    },
                )
            target["status"] = entry.to_status
            target["reason"] = entry.reason
            target["code"] = entry.code
            target["skip_from_stage_seq"] = entry.details.get("skip_from_stage_seq")

        current_stage_map = {stage.seq: stage for stage in view.stages}
        unknown: list[int] = []
        for seq, expected in reconstructed.items():
            actual = current_stage_map[seq]
            if expected["status"] == "running":
                if actual.status not in {"running", "unknown"}:
                    raise BusinessError(
                        E_LEDGER,
                        "账本重建状态与阶段表不一致",
                        stage_seq=seq,
                        field_path="phase_states.status",
                        details={"ledger": "running", "actual": actual.status},
                    )
                if actual.status == "running":
                    self.repository.mark_unknown(
                        session_id,
                        view.session.generation,
                        seq,
                        reason="强制终止后 running 阶段无终态落盘，标注未知并要求重跑",
                        logical_ms=view.session.clock_ms,
                    )
                unknown.append(seq)
            elif actual.status != expected["status"]:
                raise BusinessError(
                    E_LEDGER,
                    "账本重建状态与阶段表不一致",
                    stage_seq=seq,
                    field_path="phase_states.status",
                    details={"ledger": expected["status"], "actual": actual.status},
                )

        current_stage_map = {
            stage.seq: stage
            for stage in self.repository.get_view(session_id).stages
        }
        terminal_statuses = {"succeeded", "failed", "skipped"}
        recovered_seqs = sorted(
            seq
            for seq, stage in current_stage_map.items()
            if stage.status in terminal_statuses
        )
        pending_seqs = sorted(
            seq for seq, stage in current_stage_map.items() if stage.status == "pending"
        )
        self.repository.set_session_status(
            session_id,
            "recovered_interrupted",
            kind="recovery_completed",
            reason="依据完整账本迁移恢复；未知阶段必须重跑",
            logical_ms=view.session.clock_ms,
            code=E_LEDGER,
            details={"unknown_stage_seqs": unknown, "incomplete_tail_seqs": tail_seqs},
        )
        report = RecoveryReport(
            session_id=session_id,
            recovered_stage_count=len(recovered_seqs),
            recovered_stage_seqs=recovered_seqs,
            unknown_stage_seqs=sorted(unknown),
            pending_stage_seqs=pending_seqs,
            incomplete_tail_seqs=tail_seqs,
            tail_action=tail_action,
        )
        self.logger.emit(
            "recovery_completed",
            session_id=session_id,
            plan_version=view.plan.version,
            logical_clock_ms=view.session.clock_ms,
            recovered_stage_count=report.recovered_stage_count,
            unknown_stage_seqs=report.unknown_stage_seqs,
            incomplete_tail_seqs=tail_seqs,
            tail_action=tail_action,
            exit_report_available=False,
        )
        return report

    def resume(self, session_id: str) -> dict[str, Any]:
        view = self.repository.get_view(session_id)
        recovery: dict[str, Any] | None = None
        if view.session.status in {"running", "recovered_interrupted"} or any(
            stage.status == "running" for stage in view.stages
        ):
            recovery = self.recover(session_id).to_dict()
        summary = self.run_session(session_id)
        return {"recovery": recovery, "run": summary}

    def list_snapshots(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return self.repository.list_snapshots(session_id)

    def render_model(self, session_id: str) -> dict[str, Any]:
        view = self.repository.get_view(session_id)
        snapshots = self.list_snapshots(session_id)
        return {
            "session_id": session_id,
            "plan_id": view.plan.plan_id,
            "plan_name": view.plan.name,
            "plan_version": view.plan.version,
            "plan_digest": view.plan.digest,
            "generation": view.session.generation,
            "status": view.session.status,
            "logical_clock_ms": view.session.clock_ms,
            "clock_source": "logical",
            "unknown_stage_seqs": [
                stage.seq for stage in view.stages if stage.status == "unknown"
            ],
            "stages": [
                {
                    **stage.to_dict(),
                    "name": view.plan.stage(stage.seq).name,
                    "depends_on": list(view.plan.stage(stage.seq).depends_on),
                    "unknown": stage.status == "unknown",
                }
                for stage in sorted(view.stages, key=lambda item: item.seq)
            ],
            "events": [
                event.to_dict()
                for event in sorted(
                    view.events,
                    key=lambda item: (item.offset_ms, item.injection_seq),
                )
            ],
            "snapshots": [
                {
                    **snapshot,
                    "display_mode": "只读历史" if snapshot["read_only"] else "当前写入",
                }
                for snapshot in sorted(snapshots, key=lambda item: item["snapshot_id"])
            ],
        }

    def log_error(self, error: BusinessError, *, event: str = "business_error", **fields: Any) -> None:
        self.logger.error(error, event=event, **fields)
