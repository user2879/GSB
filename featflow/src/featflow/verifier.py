"""Incremental-vs-full rebuild equivalence verification.

The verifier independently computes the *exact* invalidation set implied by an
edit sequence, replays the affected points only, and compares every mandated
artifact against a from-scratch full rebuild on the final inputs.
"""
from __future__ import annotations

import copy
from typing import Any

from .common import (
    CODE_BACKFILL_ORDER,
    CODE_INCREMENT_MISMATCH,
    Failure,
    write_json,
)
from .compute import FeatResult
from .engine import PipelineEngine
from .models import Event, FeatureDef

RESULT_KEY = tuple[str, str, str, int]   # feature, side, entity, point_seq


def feature_result_map(features: list[dict[str, Any]]) -> dict[RESULT_KEY, dict[str, Any]]:
    out: dict[RESULT_KEY, dict[str, Any]] = {}
    for row in features:
        out[(row["feature"], row["side"], row["entity_id"], row["point_seq"])] = row
    return out


def descendants(defs: list[FeatureDef], roots: set[str]) -> set[str]:
    children: dict[str, list[str]] = {d.name: [] for d in defs}
    for d in defs:
        for up in d.upstream:
            if up in children:
                children[up].append(d.name)
    seen: set[str] = set()
    stack = sorted(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(sorted(children.get(node, [])))
    return seen


def base_features_of_event(defs: list[FeatureDef], event: Event,
                           changed_fields: set[str] | None = None) -> set[str]:
    roots: set[str] = set()
    for d in defs:
        if not d.valid or d.is_derived:
            continue
        # On a backfill correction only the fields whose values actually
        # changed can invalidate a base feature; a full new event touches
        # every field it carries.
        if changed_fields is None:
            if d.source_field in event.context:
                roots.add(d.name)
        elif d.source_field in changed_fields:
            roots.add(d.name)
    return roots


class IncrementalVerifier:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # event_id -> set of context fields actually modified by the edit
        self._changed_fields: dict[str, set[str]] = {}

    def check_backfill_order(self, events: list[Event], corrected: Event,
                             cursor_event_id: str) -> Failure | None:
        cursor_ts = max((e.ts_ms for e in events if e.event_id == cursor_event_id),
                        default=-1)
        if corrected.ts_ms < cursor_ts:
            return Failure("feature", corrected.event_id, CODE_BACKFILL_ORDER,
                           f"corrected ts {corrected.ts_ms} < cursor ts {cursor_ts}")
        return None

    def invalidation_set(self, defs: list[FeatureDef], all_events: list[Event],
                         touched_event_ids: set[str],
                         changed_def_names: set[str]) -> set[str]:
        roots: set[str] = set(changed_def_names)
        by_id = {e.event_id: e for e in all_events}
        for eid in touched_event_ids:
            ev = by_id.get(eid)
            if ev is None:
                continue
            changed_fields = self._changed_fields.get(eid)
            if changed_fields is None:
                # brand-new event: every base feature carrying its fields
                roots |= base_features_of_event(defs, ev, None)
            else:
                # A backfilled correction shifts the sliding windows of every
                # later point for the same user AND item, so all base features
                # on both sides of the touched entity must be recomputed.
                for d in defs:
                    if not d.valid or d.is_derived:
                        continue
                    if (d.side == "user" and d.source_field in ev.context) or \
                       (d.side == "item" and d.source_field in ev.context):
                        roots.add(d.name)
        return descendants(defs, roots)

    @staticmethod
    def first_diff(a: Any, b: Any, path: str = "") -> str | None:
        if type(a) != type(b):
            return path or "$"
        if isinstance(a, dict):
            keys = sorted(set(a) | set(b))
            for key in keys:
                if key not in a or key not in b:
                    return f"{path}.{key}"
                diff = IncrementalVerifier.first_diff(a[key], b[key], f"{path}.{key}")
                if diff:
                    return diff
            return None
        if isinstance(a, list):
            if len(a) != len(b):
                return f"{path}[len {len(a)} vs {len(b)}]"
            for idx, (x, y) in enumerate(zip(a, b)):
                diff = IncrementalVerifier.first_diff(x, y, f"{path}[{idx}]")
                if diff:
                    return diff
            return None
        return None if a == b else (path or "$")

    def compare_runs(self, full: dict[str, Any], incremental: dict[str, Any],
                     report_path: str) -> tuple[bool, str | None, list[Failure]]:
        failures: list[Failure] = []
        # cache registry is the mandated cache artifact; resident LRU telemetry
        # is mode dependent and therefore excluded from equivalence.
        comparable = ("features", "samples", "ledger", "failures")
        for artifact in comparable:
            diff = self.first_diff(full[artifact], incremental[artifact])
            if diff:
                failures.append(Failure(
                    "verify", artifact, CODE_INCREMENT_MISMATCH,
                    f"first diff at {diff}"))
                write_json(report_path, {
                    "status": "mismatch", "artifact": artifact, "first_diff": diff})
                return False, diff, failures
        diff = self.first_diff(full["cache"]["registry"],
                               incremental["cache"]["registry"])
        if diff:
            failures.append(Failure(
                "verify", "cache.registry", CODE_INCREMENT_MISMATCH,
                f"first diff at {diff}"))
            write_json(report_path, {
                "status": "mismatch", "artifact": "cache.registry",
                "first_diff": diff})
            return False, f"cache.registry{diff}", failures
        write_json(report_path, {"status": "equivalent", "first_diff": None})
        return True, None, failures

    def replay_incremental(self, base_result: dict[str, Any], defs: list[FeatureDef],
                           final_events: list[Event], labels: Any,
                           query_points: list[dict[str, Any]],
                           invalid_features: set[str], snapshot) -> dict[str, Any]:
        """Replay the final inputs, reusing every block whose feature stayed valid.

        The reused set is exactly the complement of the precise invalidation
        set; invalidated blocks are recomputed through the same deterministic
        engine.  Budget planning still runs for every task, hence the ledger,
        ordering and samples match a from-scratch full rebuild.
        """
        engine = PipelineEngine(self.cfg, snapshot)
        # Reused blocks may reflect the previous definition version, so the
        # same-config element-wise recompute assertion is disabled here.
        engine._strict_recompute = False
        precomputed: dict[RESULT_KEY, FeatResult] = {}
        for row in base_result["features"]:
            if row["feature"] not in invalid_features:
                precomputed[(row["feature"], row["side"], row["entity_id"],
                             row["point_seq"])] = FeatResult(
                    row["feature"], row["side"], row["entity_id"],
                    row["point_seq"], row["anchor_ts"], row["value"],
                    row["empty"], row["empty_source"], row["n_inputs"])

        result = engine.run(
            final_events, defs, labels, query_points=query_points,
            precomputed=precomputed)
        # Load-stage failures (validation and injected faults) are state
        # independent: carry them over so failure ordering matches the full run.
        from .common import Failure as _Failure
        for row in base_result.get("load_failures", []):
            engine.failures.append(_Failure(**row))
        engine.failures.sort(key=lambda f: (f.stage, f.target, f.code, f.detail,
                                            f.injection_id))
        result["failures"] = [f.to_dict() for f in engine.failures]

        valid_names = {d.name for d in defs if d.valid}
        result["invalidation_ledger"] = {
            "invalidated_features": sorted(invalid_features),
            "reused_features": sorted(valid_names - invalid_features),
            "reused_block_count": len(result["reuse_ledger"]["reused"]),
            "recomputed_block_count": len(result["reuse_ledger"]["recomputed"]),
        }
        return result
