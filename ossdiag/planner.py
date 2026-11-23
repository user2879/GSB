from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .content import LEAF_SIZE
from .models import ObjectSpec


PATHS = ("direct", "sliding", "hierarchical")
TIE_PRIORITY = ("sliding", "hierarchical", "direct")


@dataclass(frozen=True)
class PathEstimate:
    path: str
    score: int
    leaf_count: int
    batch_count: int
    shard_count: int
    memory_class: str
    work_components: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "leaf_count": self.leaf_count,
            "batch_count": self.batch_count,
            "shard_count": self.shard_count,
            "memory_class": self.memory_class,
            "work_components": dict(self.work_components),
        }


@dataclass(frozen=True)
class PathSelection:
    path: str
    reason: str
    boundary_case: bool
    tie_threshold_work: int
    estimates: tuple[PathEstimate, ...]
    best_score: int
    runner_up_score: int
    score_gap: int
    tie_break_rule: str
    total_bytes: int
    shard_count: int
    batch_count: int
    leaf_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "boundary_case": self.boundary_case,
            "tie_threshold_work": self.tie_threshold_work,
            "best_score": self.best_score,
            "runner_up_score": self.runner_up_score,
            "score_gap": self.score_gap,
            "tie_break_rule": self.tie_break_rule,
            "total_bytes": self.total_bytes,
            "shard_count": self.shard_count,
            "batch_count": self.batch_count,
            "leaf_count": self.leaf_count,
            "estimates": [estimate.as_dict() for estimate in self.estimates],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scale_parameters(spec: ObjectSpec) -> tuple[int, int, int]:
    shard_count = spec.shard_count
    batch_size = min(spec.batch_size, spec.shard_size)
    batch_count = 0
    leaf_count = 0
    for shard_index in range(shard_count):
        length = spec.shard_length(shard_index)
        batch_count += math.ceil(length / batch_size)
        leaf_count += math.ceil(length / LEAF_SIZE)
    return leaf_count, batch_count, shard_count


def estimate_paths(spec: ObjectSpec) -> tuple[PathEstimate, ...]:
    leaves, batches, shards = scale_parameters(spec)

    # Integer work model, documented in docs/DESIGN.md.  It intentionally
    # captures different costs: direct repeated per-shard batch folding,
    # bounded sliding-window churn, and hierarchical materialization overhead.
    direct_components = {
        "frontier": 2 * leaves,
        "shard_setup": 10 * shards,
        "batch_fold": (batches * batches) // max(1, shards),
        "fixed": 16,
    }
    sliding_components = {
        "frontier": 2 * leaves,
        "shard_setup": 4 * shards,
        "batch_fold": 2 * batches,
        "window_churn": leaves // 32,
        "fixed": 80,
    }
    hierarchical_components = {
        "frontier": (3 * leaves) // 2,
        "shard_setup": 4 * shards,
        "batch_fold": batches,
        "level_materialization": leaves // 4,
        "fixed": 256,
    }
    return (
        PathEstimate(
            path="direct",
            score=sum(direct_components.values()),
            leaf_count=leaves,
            batch_count=batches,
            shard_count=shards,
            memory_class="O(shards + log(leaves))",
            work_components=direct_components,
        ),
        PathEstimate(
            path="sliding",
            score=sum(sliding_components.values()),
            leaf_count=leaves,
            batch_count=batches,
            shard_count=shards,
            memory_class="O(window + log(leaves))",
            work_components=sliding_components,
        ),
        PathEstimate(
            path="hierarchical",
            score=sum(hierarchical_components.values()),
            leaf_count=leaves,
            batch_count=batches,
            shard_count=shards,
            memory_class="O(leaves)",
            work_components=hierarchical_components,
        ),
    )


def choose_path(spec: ObjectSpec, tie_threshold_work: int) -> PathSelection:
    estimates = estimate_paths(spec)
    ranked = sorted(estimates, key=lambda item: (item.score, item.path))
    best_score = ranked[0].score
    runner_up_score = ranked[1].score
    gap = runner_up_score - best_score
    boundary_case = gap < tie_threshold_work

    near_best = [item for item in estimates if item.score - best_score < tie_threshold_work]
    priority = {name: index for index, name in enumerate(TIE_PRIORITY)}
    selected = sorted(near_best, key=lambda item: (priority[item.path], item.score, item.path))[0]

    if boundary_case:
        tie_names = ",".join(item.path for item in sorted(near_best, key=lambda item: (item.score, item.path)))
        reason = (
            f"boundary: gap={gap} < threshold={tie_threshold_work}; "
            f"candidates={tie_names}; stable priority={selected.path}"
        )
    else:
        reason = (
            f"minimum estimated work: {selected.path} score={selected.score}; "
            f"runner-up gap={gap} >= threshold={tie_threshold_work}"
        )

    return PathSelection(
        path=selected.path,
        reason=reason,
        boundary_case=boundary_case,
        tie_threshold_work=tie_threshold_work,
        estimates=estimates,
        best_score=best_score,
        runner_up_score=runner_up_score,
        score_gap=gap,
        tie_break_rule="sliding > hierarchical > direct among candidates within threshold",
        total_bytes=spec.total_bytes,
        shard_count=spec.shard_count,
        batch_count=scale_parameters(spec)[1],
        leaf_count=scale_parameters(spec)[0],
    )


def selection_from_stored(
    spec: ObjectSpec,
    stored_path: str,
    stored_reason: str | None,
    stored_scores: str | None,
    stored_boundary: bool | None,
    tie_threshold_work: int,
) -> PathSelection:
    """Restore the recorded choice while retaining fresh estimates for display."""

    fresh = choose_path(spec, tie_threshold_work)
    if stored_path not in PATHS:
        return fresh
    estimates = fresh.estimates
    if stored_scores:
        try:
            parsed = json.loads(stored_scores)
            if parsed:
                by_path = {item.path: item for item in estimates}
                estimates = tuple(by_path[item["path"]] for item in parsed if item.get("path") in by_path)
        except (ValueError, KeyError, TypeError):
            pass
    best = min(estimate.score for estimate in estimates)
    ranked = sorted(item.score for item in estimates)
    runner = ranked[1] if len(ranked) > 1 else best
    return PathSelection(
        path=stored_path,
        reason=stored_reason or f"recovered recorded path {stored_path}",
        boundary_case=bool(stored_boundary),
        tie_threshold_work=tie_threshold_work,
        estimates=estimates,
        best_score=best,
        runner_up_score=runner,
        score_gap=runner - best,
        tie_break_rule="recorded in ledger; fresh estimates remain diagnostic only",
        total_bytes=spec.total_bytes,
        shard_count=spec.shard_count,
        batch_count=scale_parameters(spec)[1],
        leaf_count=scale_parameters(spec)[0],
    )
