"""Deterministic stage orchestrator driven only by the logical clock."""

from __future__ import annotations

import os
from typing import Any, Callable

from .errors import E_ABORT, E_EVENT, E_LEDGER, E_STAGE, SimulatedCrash
from .logging_utils import StructuredLogger
from .models import SessionView
from .repository import Repository


EVENT_MITIGATIONS = {
    "node_lost": "isolate_node",
    "switchover_failed": "force_switchover",
    "validation_timeout": "retry_validation",
}


class Engine:
    """Run one session. All decisions use integer logical milliseconds."""

    def __init__(
        self,
        repository: Repository,
        session_id: str,
        logger: StructuredLogger,
    ) -> None:
        self.repository = repository
        self.session_id = session_id
        self.logger = logger
        self.abort_requested = False

    def request_abort(self) -> None:
        self.abort_requested = True

    def abort(self) -> dict[str, Any]:
        """Abort a session from the UI or a controller thread."""

        view = self.repository.get_view(self.session_id)
        return self._abort(view, view.session.clock_ms)

    def _descendants(self, view: SessionView, root_seq: int) -> list[int]:
        result: set[int] = set()
        changed = True
        while changed:
            changed = False
            for stage in view.plan.stages:
                if stage.seq in result:
                    continue
                if root_seq in stage.depends_on or any(dep in result for dep in stage.depends_on):
                    result.add(stage.seq)
                    changed = True
        return sorted(result)

    def _cascade_skip(
        self,
        view: SessionView,
        failed_stage_seq: int,
        *,
        clock_ms: int,
    ) -> list[int]:
        failed_stage = view.plan.stage(failed_stage_seq)
        if failed_stage.failure_policy != "skip_downstream":
            self.logger.emit(
                "failure_policy_continue",
                stage_seq=failed_stage_seq,
                logical_clock_ms=clock_ms,
                session_id=self.session_id,
                plan_version=view.plan.version,
                reason="失败策略为 continue_downstream，下游不自动跳过",
            )
            return []
        skipped: list[int] = []
        descendants = self._descendants(view, failed_stage_seq)
        for seq in descendants:
            current = self.repository.get_stage_state(
                self.session_id, view.session.generation, seq
            )
            if current.status != "pending":
                continue
            reason = (
                f"上游阶段 {failed_stage_seq} 失败，按 {failed_stage.failure_policy} 跳过"
            )
            entry = self.repository.transition_stage(
                self.session_id,
                view.session.generation,
                seq,
                "skipped",
                reason=reason,
                logical_ms=clock_ms,
                code=E_STAGE,
                skip_from_stage_seq=failed_stage_seq,
            )
            self.repository.cancel_events_for_stage(
                self.session_id,
                view.session.generation,
                seq,
                reason=f"作用阶段 {seq} 已跳过，事件取消",
                logical_ms=clock_ms,
            )
            skipped.append(seq)
            self.logger.emit(
                "stage_transition",
                stage_seq=seq,
                logical_clock_ms=clock_ms,
                session_id=self.session_id,
                plan_version=view.plan.version,
                from_status=entry.from_status,
                to_status="skipped",
                reason=reason,
                business_code=E_STAGE,
                skip_from_stage_seq=failed_stage_seq,
                ledger_seq=entry.seq,
            )
        return skipped

    def _apply_pre_existing_skips(self, view: SessionView) -> int:
        applied = 0
        failed = [
            stage
            for stage in view.plan.stages
            if self.repository.get_stage_state(
                self.session_id, view.session.generation, stage.seq
            ).status
            == "failed"
            and stage.failure_policy == "skip_downstream"
        ]
        for stage in failed:
            state = self.repository.get_stage_state(
                self.session_id, view.session.generation, stage.seq
            )
            applied += len(
                self._cascade_skip(
                    view,
                    stage.seq,
                    clock_ms=state.updated_ms,
                )
            )
        return applied

    def _trigger_due_late_events(
        self, view: SessionView, *, clock_ms: int
    ) -> None:
        latest = self.repository.get_view(self.session_id)
        stage_status = {item.seq: item.status for item in latest.stages}
        for event in latest.events:
            if event.status != "pending" or event.offset_ms > clock_ms:
                continue
            target_status = stage_status.get(event.stage_seq)
            if target_status in {"succeeded", "failed", "skipped"}:
                reason = "事件到期时作用阶段已结束，未改变阶段状态"
                self.repository.trigger_event(
                    self.session_id,
                    latest.session.generation,
                    event.injection_seq,
                    status="ignored",
                    reason=reason,
                    logical_ms=clock_ms,
                    code="W_EVENT_LATE",
                    details={"scheduled_offset_ms": event.offset_ms},
                )
                self.logger.emit(
                    "event_ignored_late",
                    stage_seq=event.stage_seq,
                    logical_clock_ms=clock_ms,
                    session_id=self.session_id,
                    plan_version=latest.plan.version,
                    business_code="W_EVENT_LATE",
                    event_id=event.event_id,
                    event_type=event.type,
                    reason=reason,
                )

    def _summary(self, status: str | None = None) -> dict[str, Any]:
        view = self.repository.get_view(self.session_id)
        engines = self.repository.ledger_rows(
            self.session_id, view.session.generation
        )
        transitions = [item for item in engines if item.kind == "stage_transition"]
        migration_counts: dict[str, int] = {}
        for entry in transitions:
            if entry.stage_seq is not None:
                key = str(entry.stage_seq)
                migration_counts[key] = migration_counts.get(key, 0) + 1
        return {
            "session_id": self.session_id,
            "plan_id": view.plan.plan_id,
            "plan_version": view.plan.version,
            "generation": view.session.generation,
            "status": status or view.session.status,
            "logical_clock_ms": view.session.clock_ms,
            "clock_source": "logical",
            "stages": [
                {
                    "seq": item.seq,
                    "name": view.plan.stage(item.seq).name,
                    "depends_on": list(view.plan.stage(item.seq).depends_on),
                    "status": item.status,
                    "skip_reason": item.skip_reason,
                    "business_code": item.business_code,
                    "skip_from_stage_seq": item.skip_from_stage_seq,
                    "unknown_reason": item.unknown_reason,
                    "logical_ms": item.updated_ms,
                    "migration_count": migration_counts.get(str(item.seq), 0),
                }
                for item in view.stages
            ],
            "events": [item.to_dict() for item in view.events],
            "ledger_entry_count": len(engines),
            "ledger_transition_count": len(transitions),
        }

    def _abort(self, view: SessionView, clock_ms: int) -> dict[str, Any]:
        latest = self.repository.get_view(self.session_id)
        for stage in latest.stages:
            if stage.status != "pending":
                continue
            reason = "演练中止，待执行阶段跳过"
            entry = self.repository.transition_stage(
                self.session_id,
                latest.session.generation,
                stage.seq,
                "skipped",
                reason=reason,
                logical_ms=clock_ms,
                code=E_ABORT,
            )
            self.repository.cancel_events_for_stage(
                self.session_id,
                latest.session.generation,
                stage.seq,
                reason="演练中止，事件取消",
                logical_ms=clock_ms,
            )
            self.logger.emit(
                "stage_transition",
                stage_seq=stage.seq,
                logical_clock_ms=clock_ms,
                session_id=self.session_id,
                plan_version=latest.plan.version,
                from_status=entry.from_status,
                to_status="skipped",
                reason=reason,
                business_code=E_ABORT,
                ledger_seq=entry.seq,
            )
        self.repository.set_session_status(
            self.session_id,
            "aborted",
            kind="drill_aborted",
            reason="用户中止演练",
            logical_ms=clock_ms,
            code=E_ABORT,
        )
        return self._summary("aborted")

    def _crash_after(
        self,
        *,
        next_stage_seq: int | None,
        generation: int,
        plan_version: int,
        clock_ms: int,
        hard_crash: bool,
    ) -> None:
        if next_stage_seq is None:
            return
        current = self.repository.get_stage_state(
            self.session_id, generation, next_stage_seq
        )
        if current.status != "pending":
            return
        entry = self.repository.transition_stage(
            self.session_id,
            generation,
            next_stage_seq,
            "running",
            reason="阶段启动；随后模拟进程强制终止",
            logical_ms=clock_ms,
        )
        tail_seq = self.repository.append_incomplete_tail(
            self.session_id,
            generation,
            stage_seq=next_stage_seq,
            logical_ms=clock_ms,
        )
        self.logger.emit(
            "simulated_forced_termination",
            stage_seq=next_stage_seq,
            logical_clock_ms=clock_ms,
            session_id=self.session_id,
            plan_version=plan_version,
            from_status=entry.from_status,
            to_status="running",
            ledger_seq=entry.seq,
            incomplete_tail_seq=tail_seq,
            reason="进程在 running 状态提交后被强制终止，不生成退出报告",
        )
        if hard_crash:
            os._exit(137)
        raise SimulatedCrash(self.session_id, next_stage_seq, tail_seq)

    def run(
        self,
        *,
        abort_after_stage: int | None = None,
        crash_after_stage: int | None = None,
        hard_crash: bool = False,
        abort_check: Callable[[], bool] | None = None,
        stage_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        view = self.repository.get_view(self.session_id)
        if all(item.status in {"succeeded", "failed", "skipped"} for item in view.stages):
            return self._summary(view.session.status)

        self.repository.set_session_status(
            self.session_id,
            "running",
            kind="drill_started",
            reason="逻辑时钟从当前值开始推进",
            logical_ms=view.session.clock_ms,
            details={"clock_source": "logical", "starts_at_ms": view.session.clock_ms},
        )
        view = self.repository.get_view(self.session_id)
        self._apply_pre_existing_skips(view)
        clock = view.session.clock_ms
        processed_this_run = 0

        if crash_after_stage == 0:
            first = next(
                (item for item in view.stages if item.status == "pending"), None
            )
            self._crash_after(
                next_stage_seq=first.seq if first else None,
                generation=view.session.generation,
                plan_version=view.plan.version,
                clock_ms=clock,
                hard_crash=hard_crash,
            )

        for definition in view.plan.stages:
            state = self.repository.get_stage_state(
                self.session_id, view.session.generation, definition.seq
            )
            if state.status in {"succeeded", "failed", "skipped"}:
                continue
            if state.status == "running":
                self.repository.mark_unknown(
                    self.session_id,
                    view.session.generation,
                    definition.seq,
                    reason="发现未完成 running 阶段，标注未知并要求重跑",
                    logical_ms=clock,
                )
                state.status = "unknown"
            if state.status == "unknown":
                reason = "恢复后重跑未知阶段"
                transition_kind = "recovery_rerun"
            elif state.status == "pending":
                reason = "阶段开始"
                transition_kind = "stage_transition"
            else:
                continue
            entry = self.repository.transition_stage(
                self.session_id,
                view.session.generation,
                definition.seq,
                "running",
                reason=reason,
                logical_ms=clock,
                kind=transition_kind,
            )
            self.logger.emit(
                "stage_transition",
                stage_seq=definition.seq,
                logical_clock_ms=clock,
                session_id=self.session_id,
                plan_version=view.plan.version,
                from_status=entry.from_status,
                to_status="running",
                reason=reason,
                ledger_seq=entry.seq,
            )
            stage_start = clock
            stage_end = stage_start + definition.estimated_ms
            failed = False

            latest = self.repository.get_view(self.session_id)
            pending_events = sorted(
                (
                    event
                    for event in latest.events
                    if event.status == "pending"
                    and event.stage_seq == definition.seq
                ),
                key=lambda event: (event.offset_ms, event.injection_seq),
            )
            for event in pending_events:
                if event.offset_ms > stage_end:
                    continue
                event_clock = max(stage_start, event.offset_ms)
                mitigation = EVENT_MITIGATIONS[event.type]
                mitigated = mitigation in definition.actions
                if mitigated:
                    event_reason = (
                        f"演练事件 {event.event_id} ({event.type}) 由动作 {mitigation} 缓解"
                    )
                    self.repository.trigger_event(
                        self.session_id,
                        view.session.generation,
                        event.injection_seq,
                        status="triggered",
                        reason=event_reason,
                        logical_ms=event_clock,
                        details={
                            "effective_clock_ms": event_clock,
                            "mitigation_action": mitigation,
                        },
                    )
                    clock = max(clock, event_clock)
                    self.logger.emit(
                        "event_triggered",
                        stage_seq=definition.seq,
                        logical_clock_ms=event_clock,
                        session_id=self.session_id,
                        plan_version=view.plan.version,
                        event_id=event.event_id,
                        event_type=event.type,
                        reason=event_reason,
                        mitigation_action=mitigation,
                    )
                else:
                    event_reason = (
                        f"演练事件 {event.event_id} ({event.type}) 触发，"
                        f"阶段缺少缓解动作 {mitigation}"
                    )
                    self.repository.trigger_event(
                        self.session_id,
                        view.session.generation,
                        event.injection_seq,
                        status="triggered",
                        reason=event_reason,
                        logical_ms=event_clock,
                        code=E_EVENT,
                        details={
                            "effective_clock_ms": event_clock,
                            "missing_mitigation_action": mitigation,
                        },
                    )
                    self.logger.emit(
                        "event_triggered",
                        stage_seq=definition.seq,
                        logical_clock_ms=event_clock,
                        session_id=self.session_id,
                        plan_version=view.plan.version,
                        business_code=E_EVENT,
                        event_id=event.event_id,
                        event_type=event.type,
                        reason=event_reason,
                        missing_mitigation_action=mitigation,
                    )
                    clock = event_clock
                    failure_reason = (
                        f"事件 {event.event_id} ({event.type}) 导致阶段 {definition.seq} 失败"
                    )
                    failure = self.repository.transition_stage(
                        self.session_id,
                        view.session.generation,
                        definition.seq,
                        "failed",
                        reason=failure_reason,
                        logical_ms=clock,
                        code=E_EVENT,
                        details={"event_id": event.event_id, "event_type": event.type},
                    )
                    self.logger.emit(
                        "stage_transition",
                        stage_seq=definition.seq,
                        logical_clock_ms=clock,
                        session_id=self.session_id,
                        plan_version=view.plan.version,
                        from_status=failure.from_status,
                        to_status="failed",
                        reason=failure_reason,
                        business_code=E_EVENT,
                        ledger_seq=failure.seq,
                    )
                    failed = True
                    break

            if failed:
                self._cascade_skip(
                    view,
                    definition.seq,
                    clock_ms=clock,
                )
            else:
                clock = max(clock, stage_end)
                reason = "阶段预计耗时完成，成功判据满足"
                success = self.repository.transition_stage(
                    self.session_id,
                    view.session.generation,
                    definition.seq,
                    "succeeded",
                    reason=reason,
                    logical_ms=clock,
                )
                self.logger.emit(
                    "stage_transition",
                    stage_seq=definition.seq,
                    logical_clock_ms=clock,
                    session_id=self.session_id,
                    plan_version=view.plan.version,
                    from_status=success.from_status,
                    to_status="succeeded",
                    reason=reason,
                    ledger_seq=success.seq,
                )
            self._trigger_due_late_events(view, clock_ms=clock)
            processed_this_run += 1
            if stage_callback is not None:
                stage_callback(self._summary())

            should_abort = self.abort_requested or (
                abort_check is not None and abort_check()
            )
            if abort_after_stage is not None and processed_this_run >= abort_after_stage:
                should_abort = True
            if should_abort:
                return self._abort(view, clock)

            if (
                crash_after_stage is not None
                and processed_this_run >= crash_after_stage
            ):
                remaining = self.repository.get_view(self.session_id)
                next_stage = next(
                    (item for item in remaining.stages if item.status == "pending"),
                    None,
                )
                self._crash_after(
                    next_stage_seq=next_stage.seq if next_stage else None,
                    generation=view.session.generation,
                    plan_version=view.plan.version,
                    clock_ms=clock,
                    hard_crash=hard_crash,
                )

        self.repository.set_session_status(
            self.session_id,
            "completed",
            kind="drill_completed",
            reason="全部阶段进入终态，逻辑时钟推进结束",
            logical_ms=clock,
        )
        return self._summary("completed")
