from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from .budget import BudgetPlan, build_budget_plan
from .canonical import canonical_json, digest_json
from .errors import DiagnosticError, TaskExecutionError
from .ledger import SCHEMA_VERSION, Ledger
from .loader import dataset_from_dict
from .model import Dataset
from .scheduler import QuotaScheduler, TaskTrace
from .visibility import VisibilityDecision, compute_visibility, decisions_digest


@dataclass(frozen=True)
class RunOptions:
    crash_in_round: int | None = None
    crash_after_submit: int = 0


def _sorted_decisions(decisions: Iterable[VisibilityDecision]) -> list[VisibilityDecision]:
    return sorted(decisions, key=lambda item: (item.read_point_id, item.row_key, item.query_id))


def _decision_bytes(decision: VisibilityDecision | dict[str, Any] | None) -> str:
    if decision is None:
        return "null"
    if isinstance(decision, VisibilityDecision):
        return canonical_json(decision.as_dict())
    return canonical_json(decision)


def _first_query_for_refs(dataset: Dataset, *, row_keys: set[str] | None = None, read_points: set[str] | None = None) -> Any:
    for query in dataset.sorted_queries():
        if row_keys is not None and query.row_key in row_keys:
            return query
        if read_points is not None and query.read_point_id in read_points:
            return query
    queries = dataset.sorted_queries()
    return queries[0] if queries else None


def _location_for_query(query: Any, field: str, detail: str) -> dict[str, Any]:
    if query is None:
        return {
            "read_point_id": "<none>",
            "row_key": "<none>",
            "field": field,
            "detail": detail,
        }
    return {
        "read_point_id": query.read_point_id,
        "row_key": query.row_key,
        "query_id": query.query_id,
        "field": field,
        "detail": detail,
    }


def _txn_changes(old: Dataset, new: Dataset) -> dict[str, Any]:
    old_ids = set(old.txns_by_id)
    new_ids = set(new.txns_by_id)
    changed_ids: set[str] = set()
    changed: list[dict[str, Any]] = []
    for txn_id in sorted(old_ids & new_ids):
        before = old.txns_by_id[txn_id].as_dict()
        after = new.txns_by_id[txn_id].as_dict()
        if before != after:
            changed_ids.add(txn_id)
            changed.append({"txn_id": txn_id, "before": before, "after": after})
    for txn_id in sorted(new_ids - old_ids):
        changed_ids.add(txn_id)
        changed.append({"txn_id": txn_id, "before": None, "after": new.txns_by_id[txn_id].as_dict()})
    for txn_id in sorted(old_ids - new_ids):
        changed_ids.add(txn_id)
        changed.append({"txn_id": txn_id, "before": old.txns_by_id[txn_id].as_dict(), "after": None})
    return {"changed_ids": changed_ids, "items": changed}


def _row_digest(dataset: Dataset, row_key: str) -> str | None:
    if row_key not in dataset.chains:
        return None
    return digest_json([item.as_dict() for item in dataset.chains[row_key]])


def _row_changes(old: Dataset, new: Dataset) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    all_rows = sorted(set(old.chains) | set(new.chains))
    for row_key in all_rows:
        before = _row_digest(old, row_key)
        after = _row_digest(new, row_key)
        if before != after:
            before_chain = old.chains.get(row_key, ())
            after_chain = new.chains.get(row_key, ())
            changed.append(
                {
                    "row_key": row_key,
                    "before_version_count": len(before_chain),
                    "after_version_count": len(after_chain),
                    "appended_versions": [
                        item.version_no
                        for item in after_chain
                        if item.version_no not in {version.version_no for version in before_chain}
                    ],
                }
            )
    return {"changed_row_keys": {item["row_key"] for item in changed}, "items": changed}


def _rp_changes(old: Dataset, new: Dataset) -> dict[str, Any]:
    changed: set[str] = set()
    items: list[dict[str, Any]] = []
    all_ids = sorted(set(old.read_points_by_id) | set(new.read_points_by_id))
    for read_point_id in all_ids:
        before = old.read_points_by_id.get(read_point_id)
        after = new.read_points_by_id.get(read_point_id)
        before_dict = before.as_dict() if before else None
        after_dict = after.as_dict() if after else None
        if before_dict != after_dict:
            changed.add(read_point_id)
            items.append({"read_point_id": read_point_id, "before": before_dict, "after": after_dict})
    return {"changed_read_point_ids": changed, "items": items}


def _query_changes(old: Dataset, new: Dataset) -> set[str]:
    changed: set[str] = set()
    all_ids = sorted(set(old.queries_by_id) | set(new.queries_by_id))
    for query_id in all_ids:
        before = old.queries_by_id.get(query_id)
        after = new.queries_by_id.get(query_id)
        if (before.as_dict() if before else None) != (after.as_dict() if after else None):
            changed.add(query_id)
    return changed


def compute_affected(
    old: Dataset,
    new: Dataset,
    old_decisions: dict[str, dict[str, Any]],
    new_decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    txns = _txn_changes(old, new)
    rows = _row_changes(old, new)
    read_points = _rp_changes(old, new)
    changed_query_ids = _query_changes(old, new)
    changed_txn_ids: set[str] = txns["changed_ids"]
    changed_row_keys: set[str] = set(rows["changed_row_keys"])

    reference_pairs: set[tuple[str, str]] = set()
    for txn_id in changed_txn_ids:
        reference_pairs.update(old.query_refs_for_txn(txn_id))
        reference_pairs.update(new.query_refs_for_txn(txn_id))
    for row_key in changed_row_keys:
        for query in new.sorted_queries():
            if query.row_key == row_key:
                reference_pairs.add((query.read_point_id, query.row_key))
        for query in old.sorted_queries():
            if query.row_key == row_key:
                reference_pairs.add((query.read_point_id, query.row_key))

    affected_query_ids: set[str] = set(changed_query_ids)
    for query in list(old.queries) + list(new.queries):
        if (query.read_point_id, query.row_key) in reference_pairs:
            affected_query_ids.add(query.query_id)
        if query.read_point_id in read_points["changed_read_point_ids"]:
            affected_query_ids.add(query.query_id)

    affected_read_points = sorted(
        {
            query.read_point_id
            for query in list(old.queries) + list(new.queries)
            if query.query_id in affected_query_ids
        }
        | read_points["changed_read_point_ids"]
    )

    all_query_ids = sorted(set(old.queries_by_id) | set(new.queries_by_id))
    changed_items: list[dict[str, Any]] = []
    unchanged_checks: list[dict[str, Any]] = []
    for query_id in all_query_ids:
        query = new.queries_by_id.get(query_id) or old.queries_by_id[query_id]
        before = old_decisions.get(query_id)
        after = new_decisions.get(query_id)
        before_bytes = _decision_bytes(before)
        after_bytes = _decision_bytes(after)
        item = {
            "query_id": query_id,
            "read_point_id": query.read_point_id,
            "row_key": query.row_key,
            "before_visible_version_no": before.get("visible_version_no") if before else None,
            "after_visible_version_no": after.get("visible_version_no") if after else None,
            "before_reason_digest": before.get("reason_digest") if before else None,
            "after_reason_digest": after.get("reason_digest") if after else None,
            "before_outcome": before.get("outcome") if before else None,
            "after_outcome": after.get("outcome") if after else None,
            "changed": before_bytes != after_bytes,
        }
        if query_id in affected_query_ids:
            changed_items.append(item)
        else:
            unchanged_checks.append(
                {
                    **item,
                    "byte_equal": before_bytes == after_bytes,
                    "before_decision_bytes": before_bytes,
                    "after_decision_bytes": after_bytes,
                }
            )
    return {
        "affected_read_point_ids": affected_read_points,
        "affected_query_ids": sorted(affected_query_ids),
        "changed_queries": changed_items,
        "unaffected_checks": unchanged_checks,
        "unchanged_checks": unchanged_checks,
        "transaction_changes": txns["items"],
        "row_changes": rows["items"],
        "read_point_changes": read_points["items"],
        "changed_row_keys": sorted(changed_row_keys),
        "changed_txn_ids": sorted(changed_txn_ids),
    }


def first_replay_mismatch(
    old_dataset: Dataset,
    new_dataset: Dataset,
    affected: dict[str, Any] | None = None,
    conclusion_mismatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    old_meta = old_dataset.metadata_digest()
    new_meta = new_dataset.metadata_digest()
    checks = (
        ("txn_set_version", ("manifest", "txn_set_version")),
        ("dataset_id", ("manifest", "dataset_id")),
        ("txn_digest", ("txn_digest",)),
        ("chain_version", ("manifest", "chain_version")),
        ("chain_digest", ("chain_digest",)),
        ("budget_version", ("manifest", "budget_version")),
        ("budget_digest", ("budget_digest",)),
        ("read_point_set_version", ("manifest", "read_point_set_version")),
        ("read_point_digest", ("read_point_digest",)),
        ("query_digest", ("query_digest",)),
    )
    for field_name, path in checks:
        before: Any = old_meta
        after: Any = new_meta
        for part in path:
            before = before.get(part) if isinstance(before, dict) else None
            after = after.get(part) if isinstance(after, dict) else None
        if before != after:
            if field_name.startswith("txn"):
                query = _first_query_for_refs(
                    new_dataset,
                    row_keys={
                        query.row_key
                        for txn_id in (affected or {}).get("changed_txn_ids", [])
                        for query in new_dataset.sorted_queries()
                        if any(version.txn_id == txn_id for version in new_dataset.chains[query.row_key])
                    },
                )
            elif field_name.startswith("chain"):
                query = _first_query_for_refs(
                    new_dataset,
                    row_keys=set((affected or {}).get("changed_row_keys", [])),
                )
            elif field_name.startswith("read_point") or field_name == "query_digest":
                query = new_dataset.sorted_queries()[0] if new_dataset.queries else None
            else:
                query = new_dataset.sorted_queries()[0] if new_dataset.queries else None
            return _location_for_query(query, field_name, "dataset declaration differs from snapshot")
    if conclusion_mismatch is not None:
        return conclusion_mismatch
    return {
        "read_point_id": "<none>",
        "row_key": "<none>",
        "field": "unknown",
        "detail": "snapshot comparison failed",
    }


def compare_conclusions(
    expected: list[VisibilityDecision],
    stored: list[dict[str, Any]],
) -> dict[str, Any] | None:
    expected_dict = {
        item.query_id: item.as_dict()
        for item in _sorted_decisions(expected)
    }
    stored_dict = {item.get("query_id"): item for item in stored}
    all_ids = sorted(set(expected_dict) | set(stored_dict))
    for query_id in all_ids:
        before = stored_dict.get(query_id)
        after = expected_dict.get(query_id)
        if canonical_json(before) == canonical_json(after):
            continue
        base = after or before or {}
        read_point_id = base.get("read_point_id", "<none>")
        row_key = base.get("row_key", "<none>")
        field = "query_presence"
        if before is not None and after is not None:
            for candidate in (
                "visible_version_no",
                "visible_txn_id",
                "outcome",
                "predicate_matched",
                "reason_digest",
                "cost_units",
            ):
                if before.get(candidate) != after.get(candidate):
                    field = candidate
                    break
        return {
            "read_point_id": read_point_id,
            "row_key": row_key,
            "query_id": query_id,
            "field": field,
            "detail": "stored conclusion differs from deterministic replay",
        }
    return None


class Workbench:
    def __init__(self, dataset: Dataset, ledger: Ledger):
        self.dataset = dataset
        self.ledger = ledger
        self.last_report: dict[str, Any] | None = None

    def _snapshot_conclusions(self, snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            item["query_id"]: item
            for item in snapshot.get("conclusions", [])
        }

    def _pure_decisions(self, dataset: Dataset, plan: BudgetPlan) -> list[VisibilityDecision]:
        return [compute_visibility(dataset, dataset.queries_by_id[query_id]) for query_id in plan.allocated_query_ids()]

    def _replacement_assessment(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
        old_dataset = dataset_from_dict(snapshot["declarations"])
        plan = build_budget_plan(self.dataset)
        new_decisions = self._pure_decisions(self.dataset, plan)
        generation = int(snapshot["generation"])
        completed_rounds = self.ledger.completed_rounds(generation)
        completed_query_ids = {
            allocation.query_id
            for budget_round in plan.rounds
            if budget_round.round_no <= len(completed_rounds)
            for allocation in budget_round.allocations
        }
        old_decisions = {
            item["query_id"]: item for item in snapshot.get("conclusions", [])
        }
        new_decisions_by_id = {item.query_id: item.as_dict() for item in new_decisions}
        metadata_equal = old_dataset.metadata_digest() == self.dataset.metadata_digest()
        conclusion_mismatch = None
        if metadata_equal:
            conclusion_mismatch = compare_conclusions(
                [item for item in new_decisions if item.query_id in completed_query_ids],
                snapshot.get("conclusions", []),
            )
        if metadata_equal and conclusion_mismatch is None:
            return True, None, None
        affected = compute_affected(old_dataset, self.dataset, old_decisions, new_decisions_by_id)
        location = first_replay_mismatch(
            old_dataset,
            self.dataset,
            affected=affected,
            conclusion_mismatch=conclusion_mismatch,
        )
        event = {
            "ok": True,
            "code": "E_REPLAY",
            "message": "snapshot declaration or conclusion mismatch; current view was replaced wholesale",
            "location": location,
            "old_snapshot_content_digest": snapshot.get("content_digest"),
            "new_dataset_digest": digest_json(self.dataset.as_declaration_dict()),
        }
        return False, event, affected

    def execute(self, options: RunOptions | None = None) -> dict[str, Any]:
        options = options or RunOptions()
        unknown_at_start = list(self.ledger.unknown_tasks())
        snapshot = self.ledger.latest_snapshot()
        replay_event: dict[str, Any] | None = None
        affected: dict[str, Any] | None = None
        status = "COLD_START_COMPLETE"
        replacement = False

        if snapshot is None:
            generation = 1
        else:
            matches, replay_event, affected = self._replacement_assessment(snapshot)
            if matches:
                generation = int(snapshot["generation"])
                completed = self.ledger.completed_rounds(generation)
                plan = build_budget_plan(self.dataset)
                complete_count = len(completed)
                if complete_count >= len(plan.rounds):
                    expected = self._pure_decisions(self.dataset, plan)
                    mismatch = compare_conclusions(expected, snapshot.get("conclusions", []))
                    if mismatch is not None:
                        replay_event = {
                            "ok": True,
                            "code": "E_REPLAY",
                            "message": "stored conclusions failed deterministic replay; current view was replaced wholesale",
                            "location": mismatch,
                            "old_snapshot_content_digest": snapshot.get("content_digest"),
                            "new_dataset_digest": digest_json(self.dataset.as_declaration_dict()),
                        }
                        affected = {
                            "affected_read_point_ids": [mismatch["read_point_id"]],
                            "affected_query_ids": [mismatch["query_id"]],
                            "changed_queries": [],
                            "unaffected_checks": [],
                            "unchanged_checks": [],
                            "transaction_changes": [],
                            "row_changes": [],
                            "read_point_changes": [],
                            "changed_row_keys": [],
                            "changed_txn_ids": [],
                        }
                        replacement = True
                        generation += 1
                        status = "EVOLUTION_REPLACED"
                        self.ledger.record_reconciliation(
                            generation=generation,
                            old_snapshot_digest=snapshot.get("content_digest"),
                            new_dataset_digest=digest_json(self.dataset.as_declaration_dict()),
                            location=replay_event["location"],
                            affected=affected,
                            changes={"status": status, "changed_queries": []},
                        )
                    else:
                        latest_reconciliation = self.ledger.latest_reconciliation()
                        if (
                            latest_reconciliation is not None
                            and int(latest_reconciliation["generation"]) == generation
                        ):
                            affected = latest_reconciliation["affected"]
                        report = self._build_report(
                            status="REPLAY_OK",
                            generation=generation,
                            snapshot=snapshot,
                            plan=plan,
                            decisions=expected,
                            rounds=completed,
                            traces=self.ledger.all_schedule_traces(generation),
                            replay_event=None,
                            affected=affected,
                            unknown_tasks=unknown_at_start,
                        )
                        self.last_report = report
                        return report
                else:
                    status = "RESUMED_INCOMPLETE_ROUNDS"
                    replacement = False
            else:
                generation = int(snapshot["generation"]) + 1
                replacement = True
                status = "EVOLUTION_REPLACED"
                self.ledger.record_reconciliation(
                    generation=generation,
                    old_snapshot_digest=snapshot.get("content_digest"),
                    new_dataset_digest=digest_json(self.dataset.as_declaration_dict()),
                    location=replay_event["location"],
                    affected=affected or {},
                    changes={
                        "status": status,
                        "changed_queries": (affected or {}).get("changed_queries", []),
                    },
                )

        plan = build_budget_plan(self.dataset)
        completed_rounds = self.ledger.completed_rounds(generation)
        if replacement:
            completed_rounds = []
        if completed_rounds:
            status = "RESUMED_COMPLETE" if status == "RESUMED_INCOMPLETE_ROUNDS" else status
        all_decisions: list[VisibilityDecision] = []
        for round_row in completed_rounds:
            for payload in round_row["decisions"]:
                all_decisions.append(self._decision_from_payload(payload))

        traces: list[dict[str, Any]] = list(self.ledger.all_schedule_traces(generation)) if completed_rounds else []
        committed_rounds: list[dict[str, Any]] = list(completed_rounds)
        for budget_round in plan.rounds:
            if budget_round.round_no <= len(completed_rounds):
                continue
            self.ledger.insert_inflight(generation, budget_round.round_no, budget_round.allocations)

            def on_submit(count: int, allocation: Any) -> None:
                if (
                    options.crash_in_round == budget_round.round_no
                    and options.crash_after_submit > 0
                    and count >= options.crash_after_submit
                ):
                    os._exit(87)

            def operation(allocation: Any, deadline_ns: int) -> VisibilityDecision:
                query = self.dataset.queries_by_id[allocation.query_id]
                return compute_visibility(self.dataset, query, deadline_ns)

            scheduler = QuotaScheduler(self.dataset.budget)
            round_traces = scheduler.run(
                budget_round.allocations,
                operation,
                on_submit=on_submit,
            )
            failed = [trace for trace in round_traces if trace.status != "OK"]
            if failed:
                self.ledger.mark_running_unknown()
                raise TaskExecutionError(
                    "visibility task failed or timed out; the round was not committed: "
                    + ", ".join(trace.query_id for trace in failed)
                )
            round_decisions = [
                trace.decision for trace in round_traces if trace.decision is not None
            ]
            for trace, allocation in zip(round_traces, budget_round.allocations):
                if trace.decision is None:
                    continue
                if trace.decision.cost_units != allocation.estimated_cost:
                    raise TaskExecutionError(
                        f"cost estimate drift for {allocation.query_id}: "
                        f"reserved={allocation.estimated_cost} actual={trace.decision.cost_units}"
                    )
            all_decisions.extend(round_decisions)
            all_decisions = _sorted_decisions(all_decisions)
            commit = self.ledger.commit_round(
                dataset=self.dataset,
                generation=generation,
                round_no=budget_round.round_no,
                budget_round=budget_round,
                budget_plan=plan,
                traces=round_traces,
                decisions=sorted(round_decisions, key=lambda item: item.query_id),
                all_decisions=all_decisions,
            )
            committed_rounds.append(
                {
                    "ledger_cursor": commit["ledger_cursor"],
                    "generation": generation,
                    "round_no": budget_round.round_no,
                    "allocation": budget_round.as_dict(),
                    "schedule": [trace.as_dict() for trace in round_traces],
                    "decisions": [item.as_dict() for item in sorted(round_decisions, key=lambda item: item.query_id)],
                    "content_digest": commit["round_content_digest"],
                    "committed_at": commit["committed_at"],
                }
            )
            traces.extend(trace.as_dict() for trace in round_traces)

        final_snapshot = self.ledger.latest_snapshot(generation)
        if final_snapshot is None:
            status = "NO_ALLOCATED_ROUNDS"
        else:
            expected = self._pure_decisions(self.dataset, plan)
            mismatch = compare_conclusions(expected, final_snapshot.get("conclusions", []))
            if mismatch is not None:
                raise DiagnosticError(
                    "E_REPLAY",
                    mismatch,
                    "post-run replay verification failed",
                )
            all_decisions = expected
        report = self._build_report(
            status=status,
            generation=generation,
            snapshot=final_snapshot,
            plan=plan,
            decisions=all_decisions,
            rounds=committed_rounds,
            traces=traces,
            replay_event=replay_event,
            affected=affected,
            unknown_tasks=self.ledger.unknown_tasks(),
        )
        self.last_report = report
        return report

    def _decision_from_payload(self, payload: dict[str, Any]) -> VisibilityDecision:
        return VisibilityDecision(
            query_id=payload["query_id"],
            read_point_id=payload["read_point_id"],
            row_key=payload["row_key"],
            snapshot_seq=payload["snapshot_seq"],
            reader_txn_id=payload["reader_txn_id"],
            visible_version_no=payload["visible_version_no"],
            visible_txn_id=payload["visible_txn_id"],
            visible_value_digest=payload["visible_value_digest"],
            visible_value_summary=payload["visible_value_summary"],
            outcome=payload["outcome"],
            predicate_matched=payload["predicate_matched"],
            predicate_detail=payload["predicate_detail"],
            reason_chain=tuple(payload["reason_chain"]),
            reason_digest=payload["reason_digest"],
            cost_units=payload["cost_units"],
            steps_examined=payload["steps_examined"],
            predicate_columns=payload["predicate_columns"],
            visibility_policy=payload["visibility_policy"],
        )

    def _build_report(
        self,
        *,
        status: str,
        generation: int,
        snapshot: dict[str, Any] | None,
        plan: BudgetPlan,
        decisions: list[VisibilityDecision],
        rounds: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        replay_event: dict[str, Any] | None,
        affected: dict[str, Any] | None,
        unknown_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ordered_decisions = [item.as_dict() for item in _sorted_decisions(decisions)]
        round_payload = [
            {
                "generation": row["generation"],
                "round_no": row["round_no"],
                "ledger_cursor": row["ledger_cursor"],
                "allocation": row["allocation"],
                "schedule": row["schedule"],
                "decisions": row["decisions"],
                "content_digest": row["content_digest"],
                "committed_at": row["committed_at"],
            }
            for row in rounds
        ]
        snapshot_summary = None
        if snapshot is not None:
            snapshot_summary = {
                "generation": snapshot.get("generation"),
                "ledger_cursor": snapshot.get("ledger_cursor"),
                "content_digest": snapshot.get("content_digest"),
                "conclusion_digest": snapshot.get("conclusion_digest"),
                "manifest": snapshot.get("manifest"),
                "metadata": snapshot.get("metadata"),
                "cursor": snapshot.get("cursor"),
            }
        return {
            "ok": True,
            "status": status,
            "code": replay_event["code"] if replay_event else "OK",
            "diagnostic": replay_event,
            "generation": generation,
            "ledger_version": {
                "schema_version": SCHEMA_VERSION,
                "ledger_cursor": snapshot.get("ledger_cursor") if snapshot else self.ledger.ledger_seq(),
                "generation": generation,
            },
            "snapshot": snapshot_summary,
            "budget_plan": plan.as_dict(),
            "deferred_queries": [item.as_dict() for item in plan.deferred],
            "decisions": ordered_decisions,
            "conclusion_digest": decisions_digest(decisions),
            "rounds": round_payload,
            "schedule": traces,
            "affected": affected,
            "recovery": {
                "unknown_tasks": unknown_tasks,
                "unknown_count": len(unknown_tasks),
                "report_available": snapshot is not None and not unknown_tasks,
                "next_action": "RERUN_UNKNOWN_TASKS" if unknown_tasks else "NONE",
            },
        }

    def refresh_view(self) -> dict[str, Any]:
        snapshot = self.ledger.latest_snapshot()
        generation = int(snapshot["generation"]) if snapshot else (self.ledger.current_generation() or 0)
        plan = build_budget_plan(self.dataset)
        decisions: list[VisibilityDecision] = []
        if snapshot:
            decisions = [
                self._decision_from_payload(item)
                for item in snapshot.get("conclusions", [])
            ]
        rounds = self.ledger.completed_rounds(generation) if generation else []
        report = self._build_report(
            status="VIEW_ONLY",
            generation=generation,
            snapshot=snapshot,
            plan=plan,
            decisions=decisions,
            rounds=rounds,
            traces=self.ledger.all_schedule_traces(generation) if generation else [],
            replay_event=None,
            affected=self.ledger.latest_reconciliation()["affected"] if self.ledger.latest_reconciliation() else None,
            unknown_tasks=self.ledger.unknown_tasks(),
        )
        self.last_report = report
        return report


def parse_dataset_json(text: str) -> Dataset:
    return dataset_from_dict(json.loads(text))
