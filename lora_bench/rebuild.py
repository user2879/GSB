"""From-scratch rebuild helper used as the golden reference for rule 6.

A full rebuild starts from an empty snapshot workspace, but the invalidation /
reuse *verdicts* are a pure function of dependency fingerprints. The rebuild
therefore accepts the same baseline identities the incremental run saw and
recomputes the same verdict, guaranteeing the lists match field by field while
every adapter is retrained from an empty cache.
"""

from __future__ import annotations

import os
import shutil

from .deps import plan_invalidation
from .pipeline import run_pipeline


def full_rebuild(
    input_dir: str,
    fresh_workspace: str,
    config: dict,
    baseline_identities: dict | None = None,
    base_subdir: str = "base_v1",
    fault_hooks: dict | None = None,
) -> dict:
    """Wipe ``fresh_workspace`` and run the pipeline from a blank state."""
    if os.path.exists(fresh_workspace):
        shutil.rmtree(fresh_workspace)
    os.makedirs(fresh_workspace, exist_ok=True)
    outcome = run_pipeline(
        input_dir=input_dir,
        workspace=fresh_workspace,
        config=config,
        base_subdir=base_subdir,
        strict_resume=False,
        fault_hooks=fault_hooks,
    )
    if baseline_identities is not None:
        verdict = plan_invalidation(
            baseline_identities, outcome["manifest"]["task_identities"]
        )
        outcome["invalidation"] = verdict
    return outcome


def compare_outcomes(incremental: dict, rebuild: dict) -> dict:
    """Field-level comparison of incremental and full-rebuild outcomes."""
    mismatches: list[str] = []
    inc_manifest = incremental["manifest"]
    reb_manifest = rebuild["manifest"]
    comparable_manifest_keys = (
        "config_hash",
        "base_fingerprint",
        "completed_tasks",
        "task_identities",
        "config_payload",
    )
    for key in comparable_manifest_keys:
        if inc_manifest.get(key) != reb_manifest.get(key):
            mismatches.append(f"manifest.{key}")
    if incremental["invalidation"] != rebuild["invalidation"]:
        mismatches.append("invalidation")
    if incremental["failures"] != rebuild["failures"]:
        mismatches.append("failures")

    inc_by_task = {item["task"]: item for item in incremental["task_results"]}
    reb_by_task = {item["task"]: item for item in rebuild["task_results"]}
    if sorted(inc_by_task) != sorted(reb_by_task):
        mismatches.append("task_set")
    for task in sorted(set(inc_by_task) & set(reb_by_task)):
        left = inc_by_task[task]
        right = reb_by_task[task]
        for key in (
            "rank",
            "alpha",
            "target_layers",
            "scaling",
            "metric",
            "epoch_losses",
            "epoch_metrics",
            "epoch_fingerprints",
            "best",
            "final_fingerprint",
        ):
            if left.get(key) != right.get(key):
                mismatches.append(f"task[{task}].{key}")
        left_arrays = left.get("arrays", {})
        right_arrays = right.get("arrays", {})
        for array_key in sorted(set(left_arrays) | set(right_arrays)):
            if array_key not in left_arrays or array_key not in right_arrays:
                mismatches.append(f"task[{task}].array[{array_key}]")
                continue
            if not _arrays_equal(left_arrays[array_key], right_arrays[array_key]):
                mismatches.append(f"task[{task}].array[{array_key}]")
    if incremental["validation_summary"] != rebuild["validation_summary"]:
        mismatches.append("validation_summary")
    return {"equal": not mismatches, "mismatches": mismatches}


def _arrays_equal(left, right) -> bool:
    import numpy as np

    if left.shape != right.shape:
        return False
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=False))
