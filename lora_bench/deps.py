"""Two-layer dependency fingerprints and precise invalidation verdicts.

Base-level dependencies (base version, target-layer whitelist) fan out to
every task; task-level dependencies (sample contents, rank, alpha) are scoped
to a single task. Cache keys combine both layers so no stale artifact can be
reused after a base replacement, while untouched tasks keep their adapters.
"""

from __future__ import annotations

from .config import AdapterConfig
from .forward import sample_content_fingerprint
from .utils import hash_json


def task_data_fingerprint(task_entry: dict, sample_by_id: dict[str, dict]) -> str:
    """Fingerprint of the ordered train/validation sample contents of a task."""
    payload = {
        "task": task_entry["task"],
        "category": task_entry.get("category"),
        "train": [
            sample_content_fingerprint(sample_by_id[sample_id])
            for sample_id in sorted(task_entry["train_sample_ids"])
        ],
        "val": [
            sample_content_fingerprint(sample_by_id[sample_id])
            for sample_id in sorted(task_entry["val_sample_ids"])
        ],
    }
    return hash_json(payload)


def build_task_identity(
    task_entry: dict,
    sample_by_id: dict[str, dict],
    config: AdapterConfig,
    base_fingerprint: str,
) -> dict:
    """All dependency inputs that determine a task's adapter and metrics."""
    task = task_entry["task"]
    return {
        "task": task,
        "data_fingerprint": task_data_fingerprint(task_entry, sample_by_id),
        "rank": config.rank_for(task),
        "alpha": config.alpha_for(task),
        "target_layers": list(config.target_layers),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "metric": task_entry.get("metric", config.metric),
        "metric_threshold": task_entry.get(
            "metric_threshold", config.metric_threshold
        ),
        "base_fingerprint": base_fingerprint,
    }


def task_cache_key(identity: dict) -> str:
    """Stable per-task cache key derived from both dependency layers."""
    return hash_json(identity)


def stable_sort_key(task: str, target_layers: tuple[int, ...]) -> str:
    """Deterministic ordering key from task name and ascending layer order."""
    return f"{task}|layers={','.join(str(layer) for layer in sorted(target_layers))}"


def diff_reasons(old: dict | None, new: dict) -> list[str]:
    """Ordered list of dependency changes for one task (priority ordered)."""
    if old is None:
        return ["new_task"]
    reasons: list[str] = []
    if old["base_fingerprint"] != new["base_fingerprint"]:
        reasons.append("base_version")
    if old["target_layers"] != new["target_layers"]:
        reasons.append("target_layers")
    if old["data_fingerprint"] != new["data_fingerprint"]:
        reasons.append("task_samples")
    if old["rank"] != new["rank"]:
        reasons.append("rank")
    if old["alpha"] != new["alpha"]:
        reasons.append("alpha")
    if old["learning_rate"] != new["learning_rate"]:
        reasons.append("learning_rate")
    if old["batch_size"] != new["batch_size"]:
        reasons.append("batch_size")
    if old["epochs"] != new["epochs"]:
        reasons.append("epochs")
    if (
        old["metric"] != new["metric"]
        or old["metric_threshold"] != new["metric_threshold"]
    ):
        reasons.append("validation_rule")
    return reasons


def plan_invalidation(
    old_identities: dict[str, dict],
    new_identities: dict[str, dict],
) -> dict:
    """Pure dependency verdict shared by incremental and full-rebuild runs.

    The function only inspects fingerprints, never the cache directory, so an
    incremental run and a from-scratch run on the same edit produce identical
    invalidation / reuse lists. Removed tasks are reported separately.
    """
    invalidated: list[dict] = []
    reused: list[dict] = []
    for task in sorted(new_identities):
        new_identity = new_identities[task]
        old_identity = old_identities.get(task)
        reasons = diff_reasons(old_identity, new_identity)
        entry = {
            "task": task,
            "sort_key": stable_sort_key(
                task, tuple(new_identity["target_layers"])
            ),
            "cache_key": task_cache_key(new_identity),
            "reasons": reasons,
        }
        if reasons:
            invalidated.append(entry)
        else:
            reused.append(entry)
    removed = sorted(set(old_identities) - set(new_identities))
    invalidated.sort(key=lambda item: item["sort_key"])
    reused.sort(key=lambda item: item["sort_key"])
    return {
        "invalidated": invalidated,
        "reused": reused,
        "removed_tasks": removed,
    }
