"""Deterministic, fixed-point integer group aggregation using NumPy.

The simulation servers run this exact code, and the platform reuses it for
local recomputation during authority rebuild and replay so both sides always
derive byte-identical summaries.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .errors import canonical_json
from .model import Revision, Task

STRATEGY_FACTOR = {"canary": 1, "rolling": 2, "full": 3}
CALIBERS = (1, 2, 3)
GROUP_COUNT = 4
VALUE_SPAN = 1000


def effective_values(chain: list[Revision], keys: tuple[str, ...]) -> dict[str, int]:
    """Fold every revision from the root so missing keys fall back to zero."""
    merged: dict[str, int] = {key: 0 for key in keys}
    for rev in chain:
        for key in keys:
            if key in rev.values:
                merged[key] = int(rev.values[key])
    return merged


def sample_rows(seed: int, row_count: int) -> np.ndarray:
    """Reproducible integer sample rows drawn from a per-cluster fixed seed."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, VALUE_SPAN, size=(row_count, 2), dtype=np.int64)


def _group_totals(rows: np.ndarray) -> list[int]:
    groups = rows[:, 0] % GROUP_COUNT
    totals = np.zeros(GROUP_COUNT, dtype=np.int64)
    np.add.at(totals, groups, rows[:, 1])
    return [int(value) for value in totals]


def _group_counts(rows: np.ndarray) -> list[int]:
    groups = rows[:, 0] % GROUP_COUNT
    counts = np.zeros(GROUP_COUNT, dtype=np.int64)
    np.add.at(counts, groups, 1)
    return [int(value) for value in counts]


def aggregate(seed: int, row_count: int, values: dict[str, int],
              strategy: str, caliber: int,
              derives: tuple[int, ...] | list[int] | None = None) -> dict:
    """Return the stable aggregation summary for one dispatch request.

    Derived upstream references enter the fixed-point contract as a stable
    integer term so a changed derivation visibly changes the node content and
    propagates invalidation through the revision DAG.
    """
    rows = sample_rows(seed, row_count)
    totals = _group_totals(rows)
    counts = _group_counts(rows)
    factor = STRATEGY_FACTOR[strategy]
    scalar = sum(int(v) for v in values.values())
    derive_list = sorted(int(d) for d in (derives or ()))
    derive_term = sum(derive_list)
    if caliber == 1:
        groups = [factor * total + derive_term for total in totals]
        kind = "group_sum"
    elif caliber == 2:
        groups = [factor * (total // count if count else 0) + derive_term
                  for total, count in zip(totals, counts)]
        kind = "group_floor_mean"
    elif caliber == 3:
        weighted = [factor * total * (1 + scalar) + derive_term
                    for total in totals]
        groups = weighted
        kind = "group_weighted_sum"
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"unsupported caliber {caliber}")
    return {
        "kind": kind,
        "caliber": caliber,
        "strategy": strategy,
        "factor": factor,
        "derives": derive_list,
        "derive_term": derive_term,
        "groups": groups,
        "total": int(sum(groups)),
    }


def build_request(cluster, task: Task, version: int,
                  values: dict[str, int], strategy: str,
                  derives: tuple[int, ...] | list[int]) -> dict:
    return {
        "type": "dispatch",
        "cluster_id": cluster.cluster_id,
        "task_id": task.task_id,
        "version": version,
        "caliber": task.caliber,
        "strategy": strategy,
        "derives": [int(d) for d in derives],
        "seed": cluster.seed,
        "sample_rows": cluster.sample_rows,
        "values": {key: int(values[key]) for key in sorted(values)},
    }


def summarize_request(request: dict) -> dict:
    """Server-side entry point: aggregate a JSON-line dispatch request."""
    return aggregate(
        seed=int(request["seed"]),
        row_count=int(request["sample_rows"]),
        values=request["values"],
        strategy=request["strategy"],
        caliber=int(request["caliber"]),
        derives=request.get("derives", []),
    )


def node_content(task_id: str, cluster_id: str, version: int,
                 parent: int | None, summary: dict) -> dict:
    """Canonical node payload whose JSON text feeds the content digest."""
    return {
        "task_id": task_id,
        "cluster_id": cluster_id,
        "version": version,
        "parent": parent,
        "aggregate": summary,
    }


def content_digest(content: dict) -> str:
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
