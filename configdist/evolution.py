from __future__ import annotations

import copy
from typing import Any


def evolve_spec_data(
    raw: dict[str, Any],
    config_updates: dict[str, dict[str, Any]] | None = None,
    target_updates: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Create a monotonic next specification without mutating the input."""

    evolved = copy.deepcopy(raw)
    versions = evolved.setdefault("versions", {"clusters": 1, "revisions": 1, "budgets": 1})
    for cluster_id, config in sorted((config_updates or {}).items()):
        revisions = [item for item in evolved["revisions"] if item["cluster_id"] == cluster_id]
        if not revisions:
            raise ValueError(f"cluster has no revision chain: {cluster_id}")
        latest = max(revisions, key=lambda item: item["version"])
        evolved["revisions"].append(
            {
                "cluster_id": cluster_id,
                "version": int(latest["version"]) + 1,
                "parent": int(latest["version"]),
                "config": dict(config),
            }
        )
        versions["revisions"] = int(versions.get("revisions", 1)) + 1
    for task_id, targets in sorted((target_updates or {}).items()):
        task = next((item for item in evolved["tasks"] if item["id"] == task_id), None)
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        task["target_clusters"] = list(targets)
    return evolved
