"""Incremental dependency engine with deterministic concurrent scoring."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from .codes import Code, failure
from .commands import apply_command
from .diagnostics import build_error_propagation, compare_turn_reports, to_jsonable
from .fixedpoint import FixedProfile
from .hashing import fingerprint
from .model import RuntimeState
from .recall import build_recall_cache_key, compute_turn_recall
from .scoring import attribute_evidence, build_scoring_cache_key, score_turn
from .topics import build_segmentation_cache_key, segment_session


STAGE_CACHE_NAMES = ("recall", "dedup", "topic", "attribution", "contribution", "scoring")


class IncrementalEngine:
    def __init__(
        self,
        state: RuntimeState,
        base_failures: list[dict[str, Any]] | None = None,
        *,
        cache: dict[str, dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
        pending: list[dict[str, Any]] | None = None,
        store=None,
        recovery: dict[str, Any] | None = None,
        command_cursor: int = 0,
    ):
        self.state = state
        self.base_failures = list(base_failures or [])
        self.cache = cache or {name: {} for name in STAGE_CACHE_NAMES}
        for name in STAGE_CACHE_NAMES:
            self.cache.setdefault(name, {})
        self.history = history or []
        self.pending = pending or []
        self.store = store
        self.recovery = recovery or {"mode": "fresh"}
        self.command_cursor = command_cursor
        self.manifest: dict[str, Any] | None = None
        self.stop_event = threading.Event()
        self.command_failures: list[dict[str, Any]] = []
        self.last_evaluation: dict[str, Any] | None = None

    def request_stop(self) -> None:
        self.stop_event.set()

    def failure_list(self) -> list[dict[str, Any]]:
        return to_jsonable(self.base_failures + self.command_failures)

    def run_commands(self, commands: list[dict[str, Any]], *, commit: bool = True) -> dict[str, Any]:
        if self.command_cursor == 0 and not self.history:
            self.record_baseline(commit=commit)
        final_step = None
        for index in range(self.command_cursor, len(commands)):
            command = commands[index]
            command = dict(command)
            command.setdefault("index", index)
            step = self.apply_and_run(command, commit=commit)
            final_step = step
            if self.stop_event.is_set():
                break
        if final_step is None and self.history:
            final_step = self.history[-1]
        report = self.build_report(final_step)
        return report

    def record_baseline(self, *, commit: bool = True) -> dict[str, Any]:
        evaluation = self.evaluate()
        invalidation = self._build_invalidation(self.state, self.state, evaluation, {"changes": {}})
        step = {
            "kind": "baseline",
            "step": "baseline",
            "command_index": None,
            "op": "baseline",
            "applied": True,
            "failure_list": self.failure_list(),
            "invalidation": invalidation,
            "budget": evaluation["budget"],
            "pending": evaluation["pending"],
            "turn_results": evaluation["turn_results"],
            "recall_report": {"turns": evaluation["recall_turns"]},
            "topic_report": evaluation["topic_report"],
            "evidence_attribution": evaluation["attribution_list"],
            "error_propagation": evaluation["error_propagation"],
        }
        step = to_jsonable(step)
        self.history.append(step)
        if commit and self.store is not None:
            self.manifest = self.store.commit(
                state=self.state,
                cache=self.cache,
                history=self.history,
                pending=self.pending,
                command_cursor=0,
                base_failures=self.failure_list(),
                recovery=self.recovery,
            )
        self.last_evaluation = evaluation
        return step

    def apply_and_run(self, command: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        old_state = copy.deepcopy(self.state)
        command_result = apply_command(self.state, command)
        for item in command_result["failures"]:
            item.setdefault("command_index", command.get("index"))
            self.command_failures.append(item)

        budget_override = command_result.get("batch_budget_points")
        evaluation = self.evaluate(budget_override=budget_override)
        invalidation = self._build_invalidation(old_state, self.state, evaluation, command_result)
        previous_step = self.history[-1] if self.history else None
        step = {
            "step": command.get("index", len(self.history)),
            "command": to_jsonable(command),
            "command_result": to_jsonable(command_result),
            "applied": command_result["applied"],
            "failure_list": self.failure_list(),
            "invalidation": invalidation,
            "budget": evaluation["budget"],
            "pending": evaluation["pending"],
            "turn_results": evaluation["turn_results"],
            "recall_report": {"turns": evaluation["recall_turns"]},
            "topic_report": evaluation["topic_report"],
            "evidence_attribution": evaluation["attribution_list"],
            "error_propagation": evaluation["error_propagation"],
            "flip_attribution": compare_turn_reports(previous_step, {
                "turn_results": evaluation["turn_results"],
                "recall_report": {"turns": evaluation["recall_turns"]},
            }) if previous_step is not None else None,
        }
        step = to_jsonable(step)
        self.history.append(step)
        self.command_cursor = int(command.get("index", self.command_cursor)) + 1
        if commit and self.store is not None:
            self.manifest = self.store.commit(
                state=self.state,
                cache=self.cache,
                history=self.history,
                pending=self.pending,
                command_cursor=self.command_cursor,
                base_failures=self.failure_list(),
                recovery=self.recovery,
            )
        self.last_evaluation = evaluation
        return step

    def evaluate(self, *, budget_override: int | None = None) -> dict[str, Any]:
        profile = FixedProfile(self.state.fixed_width, self.state.quant_gear, self.state.rounding)
        turn_keys = self.state.all_turn_keys()
        recall_results: dict[tuple[str, str], dict[str, Any]] = {}
        recall_turns: list[dict[str, Any]] = []
        cache_decisions: list[dict[str, Any]] = []
        recompute_turns: list[dict[str, Any]] = []
        reused_turns: list[dict[str, Any]] = []

        for session_id, seq, turn_id in turn_keys:
            stable_key = [session_id, seq, turn_id]
            recall_key = build_recall_cache_key(self.state, session_id, turn_id, profile)
            decision: dict[str, Any] = {
                "session_id": session_id,
                "turn_id": turn_id,
                "seq": seq,
                "stable_sort_key": stable_key,
            }
            if recall_key in self.cache["recall"]:
                recall_native = self.cache["recall"][recall_key]
                recall_status = "reused"
                recall_basis = "recall cache key matched turn/declaration/channel-weight/quant fingerprints"
            else:
                recall_internal = compute_turn_recall(self.state, session_id, turn_id, profile)
                recall_native = to_jsonable(recall_internal)
                self.cache["recall"][recall_key] = recall_native
                self.cache["dedup"][recall_internal["dedup_cache_key"]] = recall_native
                recall_status = "recomputed"
                recall_basis = "recall cache key absent or changed"
            recall_results[(session_id, turn_id)] = recall_native
            recall_turns.append(recall_native)

            dedup_key = recall_native["dedup_cache_key"]
            if dedup_key in self.cache["dedup"]:
                dedup_status = "reused"
                dedup_basis = "dedup cache key matched recall key, cap and quant profile"
            else:
                self.cache["dedup"][dedup_key] = recall_native
                dedup_status = "recomputed"
                dedup_basis = "dedup cache key absent or recall key changed"

            decision["recall"] = {"status": recall_status, "cache_key": recall_key, "hit_basis": recall_basis}
            decision["dedup"] = {"status": dedup_status, "cache_key": dedup_key, "hit_basis": dedup_basis}
            if recall_status == "recomputed" or dedup_status == "recomputed":
                recompute_turns.append({"session_id": session_id, "turn_id": turn_id, "stages": [stage for stage, status in (("recall", recall_status), ("dedup", dedup_status)) if status == "recomputed"]})
            else:
                reused_turns.append({"session_id": session_id, "turn_id": turn_id, "stage": "recall_dedup"})
            cache_decisions.append(decision)

        topic_results: dict[str, dict[str, Any]] = {}
        topic_sessions: list[dict[str, Any]] = []
        recompute_topics: list[dict[str, Any]] = []
        reused_topics: list[dict[str, Any]] = []
        for session_id in sorted(self.state.session_order):
            segmentation_key = build_segmentation_cache_key(self.state, session_id)
            if segmentation_key in self.cache["topic"]:
                topic_result = self.cache["topic"][segmentation_key]
                segmentation_status = "reused"
            else:
                topic_result = to_jsonable(segment_session(self.state, session_id))
                self.cache["topic"][segmentation_key] = topic_result
                segmentation_status = "recomputed"
            topic_results[session_id] = topic_result
            topic_sessions.append(topic_result)
            for topic in topic_result["topics"]:
                member_key = "topic-member:" + topic["member_fingerprint"]
                if member_key in self.cache["topic"]:
                    reused_topics.append({"session_id": session_id, "topic_id": topic["topic_id"], "member_fingerprint": topic["member_fingerprint"], "hit_basis": "ordered member turn fingerprints unchanged"})
                else:
                    self.cache["topic"][member_key] = topic
                    recompute_topics.append({"session_id": session_id, "topic_id": topic["topic_id"], "member_fingerprint": topic["member_fingerprint"], "segmentation_status": segmentation_status})

        attribution_internal = attribute_evidence(self.state, recall_results, topic_results)
        attribution_list: list[dict[str, Any]] = []
        recompute_evidence: list[dict[str, Any]] = []
        reused_evidence: list[dict[str, Any]] = []
        for evidence_id in sorted(self.state.evidence):
            info = attribution_internal[evidence_id]
            attribution_key = info["cache_key"]
            if attribution_key in self.cache["attribution"]:
                info_native = self.cache["attribution"][attribution_key]
                evidence_status = "reused"
                reused_evidence.append({"evidence_id": evidence_id, "cache_key": attribution_key, "hit_basis": "citation turn/topic fingerprints unchanged"})
            else:
                info_native = to_jsonable(info)
                self.cache["attribution"][attribution_key] = info_native
                evidence_status = "recomputed"
                recompute_evidence.append({"evidence_id": evidence_id, "cache_key": attribution_key, "hit_basis": "citation or topic fingerprints changed"})
            attribution_internal[evidence_id] = info_native
            attribution_list.append(info_native)

        planned_tasks: list[dict[str, Any]] = []
        turn_results: dict[tuple[str, str], dict[str, Any]] = {}
        for session_id, seq, turn_id in turn_keys:
            scoring_key, points = build_scoring_cache_key(
                self.state, session_id, turn_id, recall_results, topic_results, attribution_internal, profile
            )
            decision = next(item for item in cache_decisions if item["session_id"] == session_id and item["turn_id"] == turn_id)
            dependencies = self._task_dependencies(session_id, turn_id, recall_results, topic_results, attribution_internal)
            decision["scoring"] = {"cache_key": scoring_key, "budget_points": points}
            decision["dependencies"] = dependencies
            if scoring_key in self.cache["scoring"]:
                cached = self.cache["scoring"][scoring_key]
                turn_results[(session_id, turn_id)] = cached
                decision["scoring"].update({"status": "reused", "hit_basis": "target/chain/evidence/topic/parameter/quant fingerprints matched"})
            else:
                planned_tasks.append(
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "seq": seq,
                        "stable_sort_key": [session_id, seq, turn_id],
                        "scoring_key": scoring_key,
                        "points": points,
                        "decision": decision,
                    }
                )

        pending_turn_ids = {(item.get("session_id"), item.get("turn_id")) for item in self.pending}
        execution_order = sorted(
            planned_tasks,
            key=lambda task: (0 if (task["session_id"], task["turn_id"]) in pending_turn_ids else 1, tuple(task["stable_sort_key"])),
        )
        budget = self.state.budget_points if budget_override is None else budget_override
        selected: list[dict[str, Any]] = []
        budget_pending: list[dict[str, Any]] = []
        used_points = 0
        if budget is None:
            selected = execution_order
            used_points = sum(task["points"] for task in selected)
        else:
            for order_index, task in enumerate(execution_order):
                if used_points + task["points"] <= int(budget):
                    selected.append(task)
                    used_points += task["points"]
                else:
                    budget_pending.extend(execution_order[order_index:])
                    break
            # Preserve stable order in the persisted todo.
            budget_pending.sort(key=lambda task: tuple(task["stable_sort_key"]))

        self._active_recall = recall_results
        self._active_topic = topic_results
        self._active_attribution = attribution_internal
        stop_pending: list[dict[str, Any]] = []
        selected_results = self._execute_scoring(selected, profile, turn_results, stop_pending)
        contribution_recomputed_turns: dict[str, list[dict[str, str]]] = {}
        for key, result in selected_results.items():
            turn_results[key] = result
            matching = next((task for task in selected if (task["session_id"], task["turn_id"]) == key), None)
            if matching is not None:
                self.cache["scoring"][matching["scoring_key"]] = result
            for detail in result.get("chain_details", []):
                contribution_status = detail.pop("cache_status", None)
                if contribution_status == "recomputed":
                    descriptor = {"session_id": key[0], "turn_id": key[1]}
                    turns = contribution_recomputed_turns.setdefault(detail["evidence_id"], [])
                    if descriptor not in turns:
                        turns.append(descriptor)
                    self.cache["contribution"][detail["contribution_cache_key"]] = {
                        "contribution_int": detail["contribution_int"],
                        "weight_int": detail["factors"]["evidence_weight"]["fixed_int"],
                        "decay_int": detail["factors"]["turn_distance_decay"]["fixed_int"],
                        "penalty_int": detail["factors"]["cross_topic_penalty"]["fixed_int"],
                    }
        # Tasks not submitted because stop was observed are pending ahead of future work.
        submitted_keys = set(selected_results.keys())
        for task in selected:
            key = (task["session_id"], task["turn_id"])
            if key not in submitted_keys and not any(item["turn_id"] == task["turn_id"] and item["session_id"] == task["session_id"] for item in stop_pending):
                stop_pending.append(task)

        pending = []
        for task in budget_pending:
            pending.append(self._pending_descriptor(task, Code.BUDGET_PENDING, "evidence forensic point budget exhausted"))
        for task in stop_pending:
            pending.append(self._pending_descriptor(task, Code.STOPPING, "stop requested after in-flight turn"))
        pending.sort(key=lambda item: tuple(item["stable_sort_key"]))
        self.pending = pending

        decision_by_key = {(item["session_id"], item["turn_id"]): item for item in cache_decisions}
        for task in selected:
            key = (task["session_id"], task["turn_id"])
            if key in selected_results:
                task["decision"]["scoring"].update(
                    {"status": "recomputed", "hit_basis": "scoring cache key absent or an input fingerprint changed"}
                )
        for item in pending:
            decision = decision_by_key[(item["session_id"], item["turn_id"])]
            decision["scoring"].update(
                {"status": "pending", "business_code": item["code"], "hit_basis": item["reason"]}
            )

        recompute_turns = []
        reused_turns = []
        for decision in cache_decisions:
            stages = []
            if decision["recall"]["status"] == "recomputed":
                stages.append("recall")
            if decision["dedup"]["status"] == "recomputed":
                stages.append("dedup")
            scoring_status = decision.get("scoring", {}).get("status")
            if scoring_status == "recomputed":
                stages.append("confidence_synthesis")
            if stages:
                recompute_turns.append(
                    {"session_id": decision["session_id"], "turn_id": decision["turn_id"], "stages": stages}
                )
            elif scoring_status == "reused":
                reused_turns.append(
                    {
                        "session_id": decision["session_id"],
                        "turn_id": decision["turn_id"],
                        "stages": ["recall", "dedup", "confidence_synthesis"],
                        "hit_basis": "all stage cache keys matched",
                    }
                )

        ordered_results = [turn_results[(sid, tid)] for sid, _seq, tid in turn_keys if (sid, tid) in turn_results]
        error_propagation = build_error_propagation(turn_keys, recall_results, turn_results)
        budget_report = {
            "limit": budget,
            "used_points": used_points if budget is not None else sum(task["points"] for task in selected),
            "remaining_points": None if budget is None else max(0, int(budget) - used_points),
            "point_unit": "one source-turn/evidence chain entry",
            "processed_tasks": len(selected_results),
            "pending_tasks": len(pending),
            "business_codes": sorted({item["code"] for item in pending}),
        }
        evaluation = {
            "profile": profile.fingerprint,
            "turn_keys": [list(key) for key in turn_keys],
            "cache_decisions": cache_decisions,
            "recompute_turns": recompute_turns,
            "reused_turns": reused_turns,
            "recompute_topics": recompute_topics,
            "reused_topics": reused_topics,
            "recompute_evidence": recompute_evidence,
            "reused_evidence": reused_evidence,
            "turn_results": to_jsonable(ordered_results),
            "recall_turns": [to_jsonable(item) for item in recall_turns],
            "topic_report": {"sessions": to_jsonable(topic_sessions)},
            "attribution_list": attribution_list,
            "error_propagation": error_propagation,
            "budget": budget_report,
            "pending": pending,
            "cache": self.cache,
            "contribution_recomputed_turns": contribution_recomputed_turns,
        }
        self.last_evaluation = evaluation
        return evaluation

    def _execute_scoring(self, tasks, profile, existing_results, stop_pending):
        results: dict[tuple[str, str], dict[str, Any]] = {}
        if not tasks:
            return results

        def run(task):
            result = score_turn(
                self.state,
                task["session_id"],
                task["turn_id"],
                self._recall_view(),
                self._topic_view(),
                self._attribution_view(),
                profile,
                self.cache["contribution"],
            )
            return result

        if self.state.workers <= 1 or len(tasks) == 1:
            for task in sorted(tasks, key=lambda item: tuple(item["stable_sort_key"])):
                if self.stop_event.is_set():
                    stop_pending.append(task)
                    continue
                native = run(task)
                results[(task["session_id"], task["turn_id"])] = to_jsonable(native)
            return results

        max_workers = min(self.state.workers, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="forensics") as executor:
            task_iter = iter(sorted(tasks, key=lambda item: tuple(item["stable_sort_key"])))
            active = {}

            def submit_next():
                if self.stop_event.is_set():
                    return False
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(run, task)
                active[future] = task
                return True

            for _ in range(max_workers):
                if not submit_next():
                    break
            while active:
                done, _not_done = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        native = future.result()
                        results[(task["session_id"], task["turn_id"])] = to_jsonable(native)
                    except Exception as exc:  # deterministic business failure instead of crash
                        failure_item = failure(
                            "E_SYNTHESIS_FAILED",
                            f"{type(exc).__name__}: {exc}",
                            session_id=task["session_id"],
                            turn_id=task["turn_id"],
                            field="confidence_synthesis",
                        )
                        self.base_failures.append(failure_item)
                while len(active) < max_workers and submit_next():
                    pass
        return results

    def _recall_view(self):
        # evaluate() stores the active view on self during concurrent execution.
        return getattr(self, "_active_recall", {})

    def _topic_view(self):
        return getattr(self, "_active_topic", {})

    def _attribution_view(self):
        return getattr(self, "_active_attribution", {})

    def _task_dependencies(self, session_id, turn_id, recall_results, topic_results, attribution):
        target = self.state.find_turn(session_id, turn_id)
        source_turns = []
        evidence_ids = []
        if target is not None:
            for source in self.state.sorted_turns(session_id):
                if source.seq > target.seq:
                    break
                recall = recall_results.get((session_id, source.turn_id))
                if not recall:
                    continue
                source_turns.append(source.turn_id)
                for item in sorted(recall["evidence"], key=lambda evidence: evidence["evidence_id"]):
                    evidence_ids.append(item["evidence_id"])
        return {
            "turn_chain": source_turns,
            "evidence_ids": evidence_ids,
            "topic_id": topic_results[session_id]["turn_topic_map"].get(turn_id),
            "evidence_topic_refs": {eid: attribution[eid]["global_topic_keys"] for eid in sorted(set(evidence_ids)) if eid in attribution},
        }

    def _pending_descriptor(self, task, code, reason):
        return {
            "code": code,
            "reason": reason,
            "session_id": task["session_id"],
            "turn_id": task["turn_id"],
            "stable_sort_key": task["stable_sort_key"],
            "scoring_cache_key": task["scoring_key"],
            "points": task["points"],
        }

    def _build_invalidation(self, old_state: RuntimeState, new_state: RuntimeState, evaluation: dict, command_result: dict) -> dict:
        old_turns = {(sid, tid): turn for sid in old_state.session_order for turn in old_state.turns.get(sid, []) for tid in [turn.turn_id]}
        new_turns = {(sid, turn.turn_id): turn for sid in new_state.session_order for turn in new_state.turns.get(sid, [])}
        changed_turns = []
        for key in sorted(set(old_turns) | set(new_turns)):
            old = old_turns.get(key)
            new = new_turns.get(key)
            if old is None:
                changed_turns.append({"key": list(key), "change": "added", "fingerprint": new.content_fingerprint})
            elif new is None:
                changed_turns.append({"key": list(key), "change": "deleted", "fingerprint": old.content_fingerprint})
            elif old.content_fingerprint != new.content_fingerprint:
                changed_turns.append({"key": list(key), "change": "content_or_seq_changed", "old_fingerprint": old.content_fingerprint, "new_fingerprint": new.content_fingerprint})
        changed_evidence = []
        for eid in sorted(set(old_state.evidence) | set(new_state.evidence)):
            old = old_state.evidence.get(eid)
            new = new_state.evidence.get(eid)
            if old is None or new is None:
                changed_evidence.append({"evidence_id": eid, "change": "deleted" if new is None else "added"})
            elif old.content_fingerprint != new.content_fingerprint or old.weight_fingerprint != new.weight_fingerprint:
                changed_evidence.append({"evidence_id": eid, "change": "content_or_weight_changed", "old_weight_fingerprint": old.weight_fingerprint, "new_weight_fingerprint": new.weight_fingerprint})
        changed_channels = [
            cid for cid in sorted(set(old_state.channels) | set(new_state.channels))
            if cid in old_state.channels and cid in new_state.channels and old_state.channels[cid].weight_fingerprint != new_state.channels[cid].weight_fingerprint
        ]
        quant_changed = (old_state.fixed_width, old_state.quant_gear) != (new_state.fixed_width, new_state.quant_gear)
        topic_params_changed = (old_state.topic_threshold, old_state.equal_rule) != (new_state.topic_threshold, new_state.equal_rule)

        attribution_recompute = {item["evidence_id"]: item for item in evaluation["recompute_evidence"]}
        forensic_turns = {
            evidence_id: [dict(item) for item in turns]
            for evidence_id, turns in evaluation.get("contribution_recomputed_turns", {}).items()
        }
        changed_evidence_ids = {item["evidence_id"] for item in changed_evidence}
        evidence_decisions = []
        recompute_evidence = []
        reused_evidence = []
        for evidence_id in sorted(set(old_state.evidence) | set(new_state.evidence)):
            stages = []
            if evidence_id in attribution_recompute:
                stages.append("topic_attribution")
            affected_turns = sorted(
                forensic_turns.get(evidence_id, []), key=lambda item: (item["session_id"], item["turn_id"])
            )
            if evidence_id in changed_evidence_ids or affected_turns:
                stages.append("forensic_weighting")
            decision = {
                "evidence_id": evidence_id,
                "stages": stages,
                "attribution_status": "recomputed" if evidence_id in attribution_recompute else "reused",
                "content_or_weight_changed": evidence_id in changed_evidence_ids,
                "forensic_affected_turns": affected_turns,
            }
            evidence_decisions.append(decision)
            if stages:
                recompute_evidence.append(decision)
            else:
                reused_evidence.append(decision)

        return {
            "command_changes": command_result.get("changes", {}),
            "changed_turns": changed_turns,
            "changed_evidence": changed_evidence,
            "changed_channels": changed_channels,
            "topic_parameters_changed": topic_params_changed,
            "quant_profile_changed": quant_changed,
            "recompute_turns": evaluation["recompute_turns"],
            "reused_turns": evaluation["reused_turns"],
            "recompute_topics": evaluation["recompute_topics"],
            "reused_topics": evaluation["reused_topics"],
            "evidence_decisions": evidence_decisions,
            "recompute_evidence": recompute_evidence,
            "reused_evidence": reused_evidence,
            "cache_decisions": evaluation["cache_decisions"],
        }

    def build_report(self, final_step: dict[str, Any] | None = None) -> dict[str, Any]:
        completed = [step for step in self.history if not step.get("pending")]
        baseline_step = next((step for step in self.history if step.get("kind") == "baseline"), None)
        command_steps = [step for step in self.history if step.get("kind") != "baseline"]
        current_turn_results = final_step["turn_results"] if final_step else (self.last_evaluation or {}).get("turn_results", [])
        current_error = final_step["error_propagation"] if final_step else (self.last_evaluation or {}).get("error_propagation", {})
        current_invalidation = final_step["invalidation"] if final_step else None
        current_budget = final_step["budget"] if final_step else (self.last_evaluation or {}).get("budget", {})
        current_pending = final_step["pending"] if final_step else (self.last_evaluation or {}).get("pending", self.pending)
        report = {
            "schema_version": "diagnostic-report/1",
            "seed": self.state.seed,
            "profile": FixedProfile(self.state.fixed_width, self.state.quant_gear, self.state.rounding).fingerprint,
            "failure_list": self.failure_list(),
            "turn_results": current_turn_results,
            "error_propagation": current_error,
            "flip_attribution": (final_step or {}).get("flip_attribution"),
            "invalidation": current_invalidation,
            "budget": current_budget,
            "pending": current_pending,
            "baseline": baseline_step,
            "steps": command_steps,
            "command_cursor": self.command_cursor,
            "recovery": self.recovery,
            "snapshot": self._snapshot_summary(),
        }
        return to_jsonable(report)

    def _snapshot_summary(self):
        if not self.manifest:
            return None
        return {
            "global_version": self.manifest.get("global_version"),
            "command_cursor": self.manifest.get("command_cursor"),
            "config_hash": self.manifest.get("config_hash"),
            "evidence_fingerprint": self.manifest.get("evidence_fingerprint"),
            "stage_versions": self.manifest.get("stage_versions"),
            "stage_changed": self.manifest.get("stage_changed"),
        }

    def commit_now(self):
        if self.store is None:
            return None
        self.manifest = self.store.commit(
            state=self.state,
            cache=self.cache,
            history=self.history,
            pending=self.pending,
            command_cursor=self.command_cursor,
            base_failures=self.failure_list(),
            recovery=self.recovery,
        )
        return self.manifest
