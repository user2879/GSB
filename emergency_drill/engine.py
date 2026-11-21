"""演练阶段状态机与确定性模拟引擎。

逻辑时钟只由注入参数与阶段预计耗时推导，全程不调用 sleep、不依赖系统时间，
满足“不得以线程等待代替逻辑推进”的约束。
"""

from dataclasses import dataclass

from .graph import stable_skipped_path, topological_order
from .models import apply_criterion

# 显式状态迁移许可表。
ALLOWED_TRANSITIONS = {
    "pending": {"running", "skipped"},
    "running": {"succeeded", "failed"},
    "succeeded": set(),
    "failed": set(),
    "skipped": set(),
}


class LogicalClock:
    def __init__(self):
        self.now_ms = 0

    def advance(self, clock_ms):
        if clock_ms < self.now_ms:
            raise ValueError(f"逻辑时钟回退: {clock_ms} < {self.now_ms}")
        self.now_ms = clock_ms
        return self.now_ms


@dataclass
class SimulationResult:
    outcomes: dict
    transitions: list
    event_conclusions: list
    stats: dict

    def ordered_outcomes(self, topo):
        return [self.outcomes[sid] for sid in topo]


def _base_payload(stage, plan_version):
    return {
        "stage_id": stage.stage_id,
        "state": "pending",
        "plan_version": plan_version,
        "start_clock_ms": 0,
        "end_clock_ms": 0,
        "reason": None,
        "source_event_id": None,
        "criterion": dict(stage.criterion),
        "criterion_passed": None,
        "failed_actions": [],
        "skip_trace": [],
        "upstream_failure": None,
    }


def _stage_events(events, stage_id):
    matched = [event for event in events if event.target_stage == stage_id]
    return sorted(matched, key=lambda event: (event.offset_ms, event.event_id))


def _emit(transitions, clock, stage_id, from_state, to_state, reason,
          plan_version, clock_ms, source_event_id=None):
    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        raise ValueError(f"非法状态迁移 {stage_id}: {from_state} -> {to_state}")
    clock_value = clock.advance(clock_ms)
    transitions.append(
        {
            "stage_id": stage_id,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "clock_ms": clock_value,
            "plan_version": plan_version,
            "source_event_id": source_event_id,
        }
    )


def simulate(plan, events, fixed_outcomes=None, invalid_stage_ids=None):
    """对预案执行一次确定性模拟。

    fixed_outcomes / invalid_stage_ids 同时提供时执行增量重算：
    固定阶段逐字节沿用历史结论，仅对失效子集重新推进状态机。
    """
    stages = plan.stages
    topo = topological_order(stages)
    mapping = plan.stage_map
    clock = LogicalClock()
    transitions = []
    outcomes = {}
    event_conclusions = []
    applied_events = set()
    failed_stage = None
    skipped_set = set()

    fixed_outcomes = fixed_outcomes or {}
    invalid_stage_ids = set(invalid_stage_ids or ())
    partial = bool(fixed_outcomes)

    def end_of(sid):
        if sid in outcomes:
            return outcomes[sid]["end_clock_ms"]
        if sid in fixed_outcomes:
            return fixed_outcomes[sid]["end_clock_ms"]
        return 0

    def start_of(sid):
        if sid in outcomes:
            return outcomes[sid]["start_clock_ms"]
        if sid in fixed_outcomes:
            return fixed_outcomes[sid]["start_clock_ms"]
        return 0

    for stage_id in topo:
        stage = mapping[stage_id]
        if partial and stage_id in fixed_outcomes:
            payload = dict(fixed_outcomes[stage_id])
            # 冻结阶段逐字节保留业务结论，但归属当前预案版本。
            payload["plan_version"] = plan.version
            outcomes[stage_id] = payload
            if payload["state"] == "failed" and failed_stage is None:
                failed_stage = stage_id
            if payload["state"] == "skipped":
                skipped_set.add(stage_id)
            for event in _stage_events(events.events, stage_id):
                event_conclusions.append(
                    {
                        "event_id": event.event_id,
                        "target_stage": stage_id,
                        "applied": False,
                        "clock_ms": None,
                        "effect": "frozen_stage_not_replayed",
                    }
                )
                applied_events.add(event.event_id)
            continue

        payload = _base_payload(stage, plan.version)
        prereq_end = max((end_of(prereq) for prereq in stage.prerequisites), default=0)

        upstream_failed = next(
            (
                prereq
                for prereq in stage.prerequisites
                if (outcomes.get(prereq) or fixed_outcomes.get(prereq) or {}).get("state")
                in ("failed", "skipped")
            ),
            None,
        )
        if upstream_failed is not None:
            fault_source = outcomes.get(upstream_failed) or fixed_outcomes.get(upstream_failed)
            root_failure = (fault_source or {}).get("upstream_failure") or upstream_failed
            skip_clock = max(prereq_end, clock.now_ms)
            _emit(transitions, clock, stage_id, "pending", "skipped",
                  "upstream_failed", plan.version, skip_clock)
            payload.update(
                {
                    "state": "skipped",
                    "start_clock_ms": skip_clock,
                    "end_clock_ms": skip_clock,
                    "reason": "upstream_failed",
                    "upstream_failure": root_failure,
                    "skip_trace": stable_skipped_path(stages, root_failure, stage_id),
                }
            )
            outcomes[stage_id] = payload
            skipped_set.add(stage_id)
            for event in _stage_events(events.events, stage_id):
                event_conclusions.append(
                    {
                        "event_id": event.event_id,
                        "target_stage": stage_id,
                        "applied": False,
                        "clock_ms": None,
                        "effect": "stage_skipped",
                    }
                )
                applied_events.add(event.event_id)
            continue

        start_clock = max(prereq_end, clock.now_ms)
        end_clock = start_clock + stage.duration_ms
        stage_events = _stage_events(events.events, stage_id)
        active_events = [event for event in stage_events if event.offset_ms <= end_clock]
        _emit(transitions, clock, stage_id, "pending", "running", "stage_start",
              plan.version, start_clock)

        node_event = next((event for event in active_events if event.kind == "node_lost"), None)
        timeout_event = next((event for event in active_events if event.kind == "check_timeout"), None)
        switch_event = next((event for event in active_events if event.kind == "switch_fail"), None)

        if node_event is not None:
            fault_clock = max(start_clock, node_event.offset_ms)
            _emit(transitions, clock, stage_id, "running", "failed", "node_lost",
                  plan.version, fault_clock, node_event.event_id)
            payload.update(
                {
                    "state": "failed",
                    "start_clock_ms": start_clock,
                    "end_clock_ms": fault_clock,
                    "reason": "node_lost",
                    "source_event_id": node_event.event_id,
                    "criterion_passed": False,
                }
            )
            outcomes[stage_id] = payload
            failed_stage = stage_id
            for event in stage_events:
                applied = event.event_id == node_event.event_id
                applied_events.add(event.event_id)
                event_conclusions.append(
                    {
                        "event_id": event.event_id,
                        "target_stage": stage_id,
                        "applied": applied,
                        "clock_ms": fault_clock if applied else None,
                        "effect": "node_lost_failure" if applied else "superseded",
                    }
                )
            continue

        failed_actions = []
        if switch_event is not None and stage.actions:
            failed_actions.append(stage.actions[0]["action_id"])

        criterion_passed = apply_criterion(
            stage.criterion, [action["action_id"] for action in stage.actions], failed_actions
        )
        if timeout_event is not None:
            criterion_passed = False

        if not criterion_passed:
            reason = "check_timeout" if timeout_event is not None else "switch_fail"
            source = timeout_event or switch_event
            _emit(transitions, clock, stage_id, "running", "failed", reason,
                  plan.version, end_clock, source.event_id if source else None)
            payload.update(
                {
                    "state": "failed",
                    "start_clock_ms": start_clock,
                    "end_clock_ms": end_clock,
                    "reason": reason,
                    "source_event_id": source.event_id if source else None,
                    "criterion_passed": False,
                    "failed_actions": sorted(failed_actions),
                }
            )
            outcomes[stage_id] = payload
            failed_stage = stage_id
            for event in stage_events:
                applied = source is not None and event.event_id == source.event_id
                applied_events.add(event.event_id)
                event_conclusions.append(
                    {
                        "event_id": event.event_id,
                        "target_stage": stage_id,
                        "applied": applied,
                        "clock_ms": end_clock if applied else None,
                        "effect": f"{reason}_failure" if applied else "superseded",
                    }
                )
            continue

        _emit(transitions, clock, stage_id, "running", "succeeded", "criterion_passed",
              plan.version, end_clock, switch_event.event_id if switch_event else None)
        payload.update(
            {
                "state": "succeeded",
                "start_clock_ms": start_clock,
                "end_clock_ms": end_clock,
                "reason": "criterion_passed",
                "criterion_passed": True,
                "failed_actions": sorted(failed_actions),
                "source_event_id": switch_event.event_id if switch_event else None,
            }
        )
        outcomes[stage_id] = payload
        for event in stage_events:
            applied_events.add(event.event_id)
            if event.kind == "switch_fail":
                effect = "action_failed_but_criterion_passed"
            elif event.offset_ms > end_clock:
                effect = "offset_after_stage_window"
            else:
                effect = "observed_without_failure"
            event_conclusions.append(
                {
                    "event_id": event.event_id,
                    "target_stage": stage_id,
                    "applied": False,
                    "clock_ms": max(start_clock, min(event.offset_ms, end_clock)),
                    "effect": effect,
                }
            )

    # 未被处理的事件（目标阶段不存在的情形已由加载校验拦截，这里兜底记录）。
    known = {entry["event_id"] for entry in event_conclusions}
    for event in sorted(events.events, key=lambda item: (item.offset_ms, item.event_id)):
        if event.event_id not in known:
            event_conclusions.append(
                {
                    "event_id": event.event_id,
                    "target_stage": event.target_stage,
                    "applied": False,
                    "clock_ms": None,
                    "effect": "not_reached",
                }
            )

    stats = {
        "total": len(topo),
        "succeeded": sum(1 for item in outcomes.values() if item["state"] == "succeeded"),
        "failed": sum(1 for item in outcomes.values() if item["state"] == "failed"),
        "skipped": sum(1 for item in outcomes.values() if item["state"] == "skipped"),
        "unknown": 0,
    }
    return SimulationResult(
        outcomes=outcomes,
        transitions=transitions,
        event_conclusions=event_conclusions,
        stats=stats,
    )


def verify_recompute(plan, events, partial_result, fixed_stage_ids):
    """整体重跑并与增量结论逐项比对，首个差异抛 E_INCR。"""
    from .errors import IncrementalMismatchError
    from .models import canonical_json

    full_result = simulate(plan, events)
    topo = topological_order(plan.stages)
    fixed_set = set(fixed_stage_ids)
    compare_fields = [
        "state",
        "start_clock_ms",
        "end_clock_ms",
        "reason",
        "source_event_id",
        "criterion_passed",
        "failed_actions",
        "skip_trace",
        "upstream_failure",
        "criterion",
    ]
    for stage_id in topo:
        left = dict(partial_result.outcomes[stage_id])
        right = dict(full_result.outcomes[stage_id])
        left.pop("plan_version", None)
        right.pop("plan_version", None)
        for field in compare_fields:
            if canonical_json(left.get(field)) != canonical_json(right.get(field)):
                raise IncrementalMismatchError(
                    "增量重算与整体重跑结论不一致",
                    location=f"stage={stage_id}.field={field}",
                    stage_id=stage_id,
                    field=field,
                    incremental=left.get(field),
                    full_rerun=right.get(field),
                )
    return {
        "matched": True,
        "frozen_stage_ids": sorted(fixed_set),
        "checked_stage_ids": topo,
    }
