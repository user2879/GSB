"""演练会话编排：版本演进、失效重算、快照与重放校验。"""

import json
import os

from .editing import apply_changeset, assert_frozen_identical
from .engine import simulate, verify_recompute
from .errors import ReplayError, SnapshotError
from .graph import topological_order
from .models import digest_of


class DrillSession:
    def __init__(self, ledger):
        self.ledger = ledger

    # ---- 首轮演练 -------------------------------------------------------
    def start_round(self, plan, events):
        existing = self.ledger.list_versions()
        if plan.version in existing:
            plan = plan.with_version(plan.stages, max(existing) + 1)
        self.ledger.save_plan(plan, events)
        result = simulate(plan, events)
        self.ledger.append_simulation(plan, result)
        return {
            "plan": plan,
            "events": events,
            "simulation": result,
            "version": plan.version,
            "replaced": None,
        }

    def new_round(self, plan, events):
        """新一轮演练整体替换当前阶段、事件与结论，并稳定列出被剔除标识。"""
        previous_version = self.ledger.latest_version()
        replaced = None
        if previous_version is not None:
            old_plan, old_events = self.ledger.load_plan_events(previous_version)
            new_version = previous_version + 1
            plan = plan.with_version(plan.stages, new_version)
            replaced = {
                "stale_stage_ids": topological_order(old_plan.stages),
                "stale_event_ids": sorted(event.event_id for event in old_events.events),
                "old_version": previous_version,
                "new_version": new_version,
            }
        self.ledger.save_plan(plan, events)
        result = simulate(plan, events)
        self.ledger.append_simulation(plan, result, mark_stale_versions={previous_version} if previous_version else set())
        return {
            "plan": plan,
            "events": events,
            "simulation": result,
            "version": plan.version,
            "replaced": replaced,
        }

    # ---- 增量编辑与仅失效子集重算 ---------------------------------------
    def edit_and_recompute(self, plan, events, changeset):
        old_version = plan.version
        old_outcomes = {
            payload["stage_id"]: payload
            for payload in self.ledger.active_outcomes(old_version)
        }
        edit = apply_changeset(plan, changeset)
        new_plan = edit["plan"]
        invalid_ids = edit["invalid_stage_ids"]
        frozen_ids = [sid for sid in topological_order(new_plan.stages) if sid not in set(invalid_ids)]

        fixed_outcomes = {sid: old_outcomes[sid] for sid in frozen_ids}
        self.ledger.save_plan(new_plan, events)
        partial = simulate(
            new_plan,
            events,
            fixed_outcomes=fixed_outcomes,
            invalid_stage_ids=invalid_ids,
        )
        assert_frozen_identical(old_outcomes, frozen_ids, partial.outcomes)
        self.ledger.append_simulation(
            new_plan, partial, mark_stale_versions={old_version}
        )
        verification = verify_recompute(new_plan, events, partial, frozen_ids)
        return {
            "plan": new_plan,
            "simulation": partial,
            "version": new_plan.version,
            "invalid_stage_ids": invalid_ids,
            "frozen_stage_ids": frozen_ids,
            "propagation": edit["propagation"],
            "deleted_stage_ids": edit["deleted_stage_ids"],
            "verification": verification,
        }

    # ---- 快照 -----------------------------------------------------------
    def create_snapshot(self, version, path):
        plan, events = self.ledger.load_plan_events(version)
        outcomes = self.ledger.active_outcomes(version)
        progress = self.ledger.progress(version)
        completed = sorted(
            payload["stage_id"]
            for payload in outcomes
            if payload["state"] in ("succeeded", "failed", "skipped")
        )
        body = {
            "snapshot_format": 1,
            "plan_version": version,
            "plan_digest": plan.digest(),
            "event_digest": events.digest(),
            "event_sequence": [event.event_id for event in events.events],
            "state_machine": {
                payload["stage_id"]: {
                    "state": payload["state"],
                    "plan_version": payload["plan_version"],
                }
                for payload in outcomes
            },
            "completed_stage_ids": completed,
            "ledger_digest": self.ledger.ledger_digest(),
            "progress_cursor": progress,
        }
        snapshot = dict(body)
        snapshot["content_digest"] = digest_of(body)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, indent=2)
        return snapshot

    def open_snapshot(self, current_version, path):
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        stored_digest = snapshot.get("content_digest")
        body = {key: value for key, value in snapshot.items() if key != "content_digest"}
        if not stored_digest or digest_of(body) != stored_digest:
            raise SnapshotError(
                "快照内容摘要不符，拒绝恢复",
                location=path,
                expected=stored_digest,
                actual=digest_of(body),
            )
        plan, events = self.ledger.load_plan_events(current_version)
        diff = {
            "version_mismatch": snapshot["plan_version"] != current_version,
            "plan_digest_mismatch": snapshot["plan_digest"] != plan.digest(),
            "event_digest_mismatch": snapshot["event_digest"] != events.digest(),
            "snapshot_version": snapshot["plan_version"],
            "current_version": current_version,
        }
        diff["needs_full_replace"] = any(
            (diff["version_mismatch"], diff["plan_digest_mismatch"], diff["event_digest_mismatch"])
        )
        diff["removed_stage_ids"] = sorted(
            set(snapshot["state_machine"]) - set(plan.stage_map)
        )
        diff["added_stage_ids"] = sorted(
            set(plan.stage_map) - set(snapshot["state_machine"])
        )
        diff["removed_event_ids"] = sorted(
            set(snapshot["event_sequence"]) - set(event.event_id for event in events.events)
        )
        diff["added_event_ids"] = sorted(
            set(event.event_id for event in events.events) - set(snapshot["event_sequence"])
        )
        return snapshot, diff

    def replay_snapshot(self, version, path):
        snapshot, diff = self.open_snapshot(version, path)
        if diff["needs_full_replace"]:
            raise SnapshotError(
                "快照与当前会话版本或内容摘要不一致，须整体替换而非重放",
                location=path,
                **diff,
            )
        plan, events = self.ledger.load_plan_events(version)
        result = simulate(plan, events)
        topo = topological_order(plan.stages)
        replay_fields = ["state", "reason", "source_event_id", "start_clock_ms",
                         "end_clock_ms", "criterion_passed", "skip_trace",
                         "upstream_failure", "failed_actions"]
        for stage_id in topo:
            saved = snapshot["state_machine"].get(stage_id)
            if saved is None:
                raise ReplayError(
                    "快照缺少阶段状态",
                    location=f"stage={stage_id}",
                    stage_id=stage_id,
                    field="state",
                )
            actual = result.outcomes[stage_id]
            for field in replay_fields:
                if field not in saved and field in ("reason", "source_event_id",
                                                    "start_clock_ms", "end_clock_ms",
                                                    "criterion_passed", "skip_trace",
                                                    "upstream_failure", "failed_actions"):
                    continue
                if field == "state":
                    if saved["state"] != actual["state"]:
                        raise ReplayError(
                            "快照重放状态不一致",
                            location=f"stage={stage_id}.field=state",
                            stage_id=stage_id,
                            field="state",
                            snapshot=saved["state"],
                            replay=actual["state"],
                        )
            if saved.get("plan_version") != actual["plan_version"]:
                raise ReplayError(
                    "快照重放预案版本不一致",
                    location=f"stage={stage_id}.field=plan_version",
                    stage_id=stage_id,
                    field="plan_version",
                    snapshot=saved.get("plan_version"),
                    replay=actual["plan_version"],
                )
        expected_completed = set(snapshot["completed_stage_ids"])
        actual_completed = {
            sid for sid in topo if result.outcomes[sid]["state"] in ("succeeded", "failed", "skipped")
        }
        if expected_completed != actual_completed:
            first = next(sid for sid in topo if (sid in expected_completed) != (sid in actual_completed))
            raise ReplayError(
                "快照已完成阶段集合与重放结果不一致",
                location=f"stage={first}.field=completed",
                stage_id=first,
                field="completed",
                snapshot=first in expected_completed,
                replay=first in actual_completed,
            )
        return {"matched": True, "plan_version": version, "checked_stage_ids": topo}


def tamper_snapshot_digest(path, output_path):
    """生成内容摘要不符的快照变体，供 E_SNAPSHOT 异常注入验证。"""
    with open(path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    snapshot["content_digest"] = "0" * 64
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return output_path
