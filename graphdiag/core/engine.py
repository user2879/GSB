"""Session engine: versioning, staleness guard, snapshots and replay checks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .errors import ReplayError, StateError
from .fixedpoint import Quantizer
from .loader import load_declaration
from .models import (
    GraphDeclaration,
    canonical_json,
    content_digest,
)
from .persistence import Store
from .scheduler import ScheduleReport, TraversalBudget, schedule_all
from .traversal import (
    CONFIRMED_OVER,
    ERROR_CROSS,
    UNREACHABLE,
    WITHIN,
    TraversalResult,
)


def result_to_dict(result: TraversalResult) -> dict[str, Any]:
    return {
        "query_id": result.query_id,
        "source": result.source,
        "cap_fp": result.cap_fp,
        "error_mode": result.error_mode,
        "budget_visited": result.budget_visited,
        "budget_exhausted": result.budget_exhausted,
        "saturated_events": result.saturated_events,
        "visited_sequence": list(result.visited_sequence),
        "pruned": [
            {
                "source": p.source,
                "target": p.target,
                "weight_micro": p.weight_micro,
                "dist_before_fp": p.dist_before_fp,
                "candidate_fp": p.candidate_fp,
                "cap_fp": p.cap_fp,
                "reason": p.reason,
            }
            for p in result.pruned
        ],
        "targets": {
            str(target): {
                "target": tr.target,
                "reached": tr.reached,
                "path_nodes": list(tr.path_nodes),
                "dist_fp": tr.dist_fp,
                "dist_micro": tr.dist_micro,
                "exact_bound_micro": tr.exact_bound_micro,
                "conservative_bound_ulp": tr.conservative_bound_ulp,
                "conservative_bound_micro": tr.conservative_bound_micro,
                "dominant_edge": list(tr.dominant_edge) if tr.dominant_edge else None,
                "dominant_remainder_micro": tr.dominant_remainder_micro,
                "cap_micro": tr.cap_micro,
                "classification": tr.classification,
                "classification_exact": tr.classification_exact,
                "classification_conservative": tr.classification_conservative,
                "error_affected": tr.error_affected,
                "terms": [term.__dict__ for term in tr.terms],
            }
            for target, tr in sorted(result.targets.items())
        },
    }


def report_to_dict(report: ScheduleReport) -> dict[str, Any]:
    return {
        "order": list(report.order),
        "schedules": {
            qid: {
                "query_id": sched.query_id,
                "enqueue_monotonic_ms": sched.enqueue_monotonic_ms,
                "wait_start_monotonic_ms": sched.wait_start_monotonic_ms,
                "wait_end_monotonic_ms": sched.wait_end_monotonic_ms,
                "wait_duration_ms": sched.wait_duration_ms,
                "start_monotonic_ms": sched.start_monotonic_ms,
                "end_monotonic_ms": sched.end_monotonic_ms,
                "visited_nodes": sched.visited_nodes,
                "pruned_branches": sched.pruned_branches,
                "budget_consumed": sched.budget_consumed,
                "budget_exhausted": sched.budget_exhausted,
                "result": result_to_dict(sched.result) if sched.result else None,
            }
            for qid, sched in sorted(report.schedules.items())
        },
    }


def _rebuild_graph(graph_payload: dict[str, Any]):
    from .loader import _validate_graph
    return _validate_graph(graph_payload)


def _rebuild_queries(queries_payload: list[dict[str, Any]], graph):
    from .loader import _validate_queries
    return _validate_queries({"queries": queries_payload}, graph)


@dataclass
class ConclusionView:
    """A complete, replace-as-a-whole view of the current conclusions."""

    graph_version: int
    query_version: int
    scale_exponent: int
    graph_digest: str
    queries_digest: str
    report: ScheduleReport
    created_wall_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ChangeEntry:
    query_id: str
    target: int
    old_dist_micro: int | None
    new_dist_micro: int | None
    distance_delta_micro: int | None
    old_classification: str
    new_classification: str


class Engine:
    """Owns session state, version counters, ledger and snapshots."""

    def __init__(self, db_path: str, error_mode: str = "exact") -> None:
        self.store = Store(db_path)
        self.error_mode = error_mode
        self.declaration: GraphDeclaration | None = None
        self.graph_version = 0
        self.query_version = 0
        self.scale_exponent: int | None = None
        self._last_graph_digest: str | None = None
        self._last_queries_digest: str | None = None
        self._last_scale: int | None = None
        self.current_view: ConclusionView | None = None
        self.previous_view: ConclusionView | None = None  # kept for diffing

    # -- loading / versioning ----------------------------------------------
    def load(self, declaration_path: str) -> dict[str, Any]:
        """Load a declaration, bumping monotonically on any content change.

        Versions are content-addressed: graph / query / scale versions only
        advance when the corresponding digest changes.
        """
        declaration = load_declaration(declaration_path)
        graph_digest = content_digest(declaration.graph_payload())
        queries_digest = content_digest(declaration.queries_payload())
        scale = declaration.scale_exponent

        graph_changed = graph_digest != self._last_graph_digest
        query_changed = queries_digest != self._last_queries_digest
        scale_changed = scale != self._last_scale

        if graph_changed:
            self.graph_version += 1
        if query_changed:
            self.query_version += 1
        if scale_changed and self._last_scale is not None:
            # A scale change is a fixed-point caliber change: track it through
            # the graph-version axis as a new (graph, scale) combination.
            self.graph_version += 1
        if self._last_scale is None and self._last_graph_digest is None:
            # First load still receives explicit version 1.
            self.graph_version = 1
            self.query_version = 1

        self.declaration = declaration
        self.scale_exponent = scale
        self._last_graph_digest = graph_digest
        self._last_queries_digest = queries_digest
        self._last_scale = scale
        # Any prior conclusions are now stale and must not be displayed.
        self.previous_view = self.current_view
        self.current_view = None
        return {
            "graph_version": self.graph_version,
            "query_version": self.query_version,
            "scale_exponent": scale,
            "graph_digest": graph_digest,
            "queries_digest": queries_digest,
            "graph_changed": graph_changed,
            "query_changed": query_changed,
            "scale_changed": scale_changed,
        }

    # -- execution ----------------------------------------------------------
    def _quantizer(self) -> Quantizer:
        if self.declaration is None:
            raise StateError("no declaration loaded")
        quantizer = Quantizer(self.declaration.scale_exponent)
        max_weight = max(
            (edge.weight_micro for edge in self.declaration.graph.edges),
            default=0,
        )
        max_cap = max(query.cap_micro for query in self.declaration.queries)
        quantizer.check_domain(max_weight, max_cap)
        return quantizer

    def run(
        self,
        budget_limit: int,
        max_workers: int = 2,
        wait_timeout: float | None = None,
    ) -> ConclusionView:
        if self.declaration is None:
            raise StateError("no declaration loaded")
        quantizer = self._quantizer()
        budget = TraversalBudget(budget_limit)
        report = schedule_all(
            self.declaration.graph,
            self.declaration.queries,
            quantizer,
            budget,
            max_workers=max_workers,
            error_mode=self.error_mode,
            wait_timeout=wait_timeout,
        )
        now_monotonic = time.monotonic_ns() // 1_000_000
        for schedule in report.ordered():
            if schedule.result is not None:
                self.store.append_ledger(
                    schedule.query_id,
                    self.graph_version,
                    self.query_version,
                    self.scale_exponent,
                    result_to_dict(schedule.result),
                    now_monotonic,
                )
        view = ConclusionView(
            graph_version=self.graph_version,
            query_version=self.query_version,
            scale_exponent=self.scale_exponent,
            graph_digest=self._last_graph_digest,
            queries_digest=self._last_queries_digest,
            report=report,
        )
        # Whole-view replacement, never incremental merging.  ``load`` already
        # moved the prior conclusions into ``previous_view``; keep them there
        # so the post-run diff can compare the two whole views.
        if self.previous_view is None and self.current_view is not None:
            self.previous_view = self.current_view
        self.current_view = view
        return view

    def diff_views(self) -> list[ChangeEntry]:
        """Stable change list between the previous and current whole views."""
        if self.previous_view is None or self.current_view is None:
            return []
        changes: list[ChangeEntry] = []
        old = self.previous_view.report.schedules
        new = self.current_view.report.schedules
        for query_id in self.current_view.report.order:
            if query_id not in old or query_id not in new:
                continue
            old_targets = old[query_id].result.targets if old[query_id].result else {}
            new_targets = new[query_id].result.targets if new[query_id].result else {}
            for target in sorted(new_targets):
                old_tr = old_targets.get(target)
                new_tr = new_targets[target]
                old_dist = old_tr.dist_micro if old_tr else None
                new_dist = new_tr.dist_micro
                old_cls = old_tr.classification if old_tr else UNREACHABLE
                if old_dist != new_dist or old_cls != new_tr.classification:
                    delta = None
                    if old_dist is not None and new_dist is not None:
                        delta = new_dist - old_dist
                    changes.append(
                        ChangeEntry(
                            query_id=query_id,
                            target=target,
                            old_dist_micro=old_dist,
                            new_dist_micro=new_dist,
                            distance_delta_micro=delta,
                            old_classification=old_cls,
                            new_classification=new_tr.classification,
                        )
                    )
        return changes

    # -- snapshots ----------------------------------------------------------
    def build_snapshot_payload(self) -> dict[str, Any]:
        if self.declaration is None or self.current_view is None:
            raise StateError("cannot snapshot before loading and running a session")
        return {
            "graph_version": self.graph_version,
            "query_version": self.query_version,
            "scale_exponent": self.scale_exponent,
            "graph_digest": self._last_graph_digest,
            "queries_digest": self._last_queries_digest,
            "error_mode": self.error_mode,
            "graph": self.declaration.graph_payload(),
            "queries": self.declaration.queries_payload(),
            "ledger": self.store.read_ledger(),
            "conclusions": report_to_dict(self.current_view.report),
        }

    def save_snapshot(self, snapshot_id: str) -> str:
        payload = self.build_snapshot_payload()
        digest = self.store.save_snapshot(snapshot_id, payload)
        self.store.record_event(snapshot_id, "saved", {"digest": digest})
        return digest

    def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore whole state from a snapshot, then verify via replay."""
        payload = self.store.load_snapshot(snapshot_id)
        if payload["integer_format_version"] != 1:
            raise StateError(
                f"unsupported integer format version: "
                f"{payload['integer_format_version']}"
            )
        payload_body = {
            k: v for k, v in payload.items() if k != "integer_format_version"
        }
        digest_now = content_digest(payload_body)

        # Whole replacement: rebuild a declaration object from the snapshot.
        graph = _rebuild_graph(payload["graph"])
        queries = _rebuild_queries(payload["queries"], graph)
        rebuilt = GraphDeclaration(
            graph=graph,
            queries=queries,
            scale_exponent=payload["scale_exponent"],
        )
        mismatch = self._replay_compare(rebuilt, payload)
        if mismatch is not None:
            self.store.record_event(snapshot_id, "replay_failed", mismatch)
            raise ReplayError(
                "replay diverges from snapshot",
                detail=mismatch,
            )
        # Only adopt the snapshot state after replay succeeds.
        self.declaration = rebuilt
        self.graph_version = payload["graph_version"]
        self.query_version = payload["query_version"]
        self.scale_exponent = payload["scale_exponent"]
        self._last_graph_digest = payload["graph_digest"]
        self._last_queries_digest = payload["queries_digest"]
        self._last_scale = payload["scale_exponent"]
        self.error_mode = payload["error_mode"]
        self.current_view = None
        self.previous_view = None
        self.store.record_event(snapshot_id, "restored", {"status": "replay_ok"})
        return {"snapshot_id": snapshot_id, "replay": "ok", "digest": digest_now}

    def _replay_compare(
        self, declaration: GraphDeclaration, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Re-execute every query and locate the first diverging target."""
        quantizer = Quantizer(declaration.scale_exponent)
        max_weight = max(
            (edge.weight_micro for edge in declaration.graph.edges), default=0
        )
        max_cap = max(query.cap_micro for query in declaration.queries)
        quantizer.check_domain(max_weight, max_cap)
        budget = TraversalBudget(10 ** 9)
        replay_report = schedule_all(
            declaration.graph,
            declaration.queries,
            quantizer,
            budget,
            max_workers=1,  # replay is single-threaded for strict determinism
            error_mode=payload["error_mode"],
        )
        snap_conclusions = payload["conclusions"]["schedules"]
        for query in declaration.queries:
            snap_schedule = snap_conclusions[query.query_id]
            replay_result = replay_report.schedules[query.query_id].result
            snap_result = snap_schedule["result"]
            mismatch = self._first_target_mismatch(
                query.query_id, snap_result, result_to_dict(replay_result)
            )
            if mismatch is not None:
                return mismatch
        return None

    @staticmethod
    def _first_target_mismatch(
        query_id: str, snap_result: dict[str, Any], replay_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        snap_targets = snap_result["targets"]
        replay_targets = replay_result["targets"]
        for target_key in sorted(snap_targets, key=lambda k: int(k)):
            snap_tr = snap_targets[target_key]
            replay_tr = replay_targets.get(target_key)
            if replay_tr is None:
                return {
                    "query_id": query_id,
                    "target": int(target_key),
                    "field": "target_missing",
                    "snapshot": snap_tr,
                    "replay": None,
                }
            for field_name in (
                "dist_fp",
                "exact_bound_micro",
                "conservative_bound_ulp",
                "classification",
                "classification_exact",
                "classification_conservative",
                "dominant_edge",
                "path_nodes",
            ):
                if snap_tr[field_name] != replay_tr[field_name]:
                    return {
                        "query_id": query_id,
                        "target": int(target_key),
                        "field": field_name,
                        "snapshot": snap_tr[field_name],
                        "replay": replay_tr[field_name],
                    }
        return None
