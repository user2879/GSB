"""End-to-end acceptance demo on the deterministic local sample.

Flow:
  1. clean full rebuild under e2, snapshot to disk
  2. apply the edit command stream on the same snapshot root (incremental)
  3. clean rebuild under the final edited state, prove field-level equality
  4. repeat the same final state under e1, prove equal integers / distinct keys
  5. write a machine-readable acceptance summary
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import IssueLog
from .loader import load_edits, load_pipeline
from .orchestrator import Orchestrator
from .samplegen import generate


def _load(paths: dict[str, Path], version: str, snapshot_root: Path, log: IssueLog) -> Orchestrator:
    model = load_pipeline(
        paths["config"], paths["tokens"], paths["segments"], paths["graph"],
        e1_path=paths["e1"] if version == "e1" else None,
        e2_path=paths["e2"] if version == "e2" else None,
        bypass_path=paths["bypass"] if version == "e2" else None,
    )
    log.extend(model.log)
    return Orchestrator(model, snapshot_root, exchange_version=version)


def run_demo(workspace: str | Path) -> dict:
    workspace = Path(workspace)
    sample_root = workspace / "samples"
    paths = generate(sample_root)
    snap_root = workspace / "snapshots" / "demo"
    log = IssueLog()

    # 1. clean full run under e2
    first = _load(paths, "e2", snap_root / "baseline", log)
    first.start()
    _, first_diag, _ = first.run_to_completion()
    baseline_outputs = first.engine.all_outputs()

    # 2. incremental run: resume the committed baseline, then apply edits so
    #    exact invalidation + reuse is exercised against real cached entries.
    incremental = _load(paths, "e2", snap_root / "baseline", IssueLog())
    resumed_manifest = incremental.start()  # reloads committed ledger
    assert resumed_manifest is not None, "baseline snapshot must be resumable"
    edits = load_edits(paths["edits"], IssueLog())
    marker = len(incremental.engine.ledger.events)
    outcome = incremental.apply_edits(edits)
    _, inc_diag, _ = incremental.run_to_completion(resumed_manifest)
    incremental_outputs = incremental.engine.all_outputs()
    inc_events = incremental.engine.ledger.events[marker:]
    invalidated_now = sum(1 for event in inc_events if event.kind == "invalidated")
    reused_now = sum(1 for event in inc_events if event.kind == "reused")

    # 3. clean rebuild from a blank root under the identical final state
    clean = _load(paths, "e2", snap_root / "clean", IssueLog())
    clean.editor.apply(edits)
    clean.exchange_version = clean.editor.exchange_version
    clean.engine.exchange_version = clean.exchange_version
    clean.start()
    clean.run_to_completion()
    clean_outputs = clean.engine.all_outputs()

    # 4. identical quant semantics under e1 (no edits) to isolate the
    #    exchange-version component of cache keys; integer outputs must match.
    clean_e1 = _load(paths, "e1", snap_root / "clean-e1", IssueLog())
    clean_e1.start()
    clean_e1.run_to_completion()
    e1_outputs = clean_e1.engine.all_outputs()
    e2_equiv = _load(paths, "e2", snap_root / "clean-e2-equiv", IssueLog())
    e2_equiv.start()
    e2_equiv.run_to_completion()
    e2_equiv_outputs = e2_equiv.engine.all_outputs()

    incremental_match = _compare(incremental_outputs, clean_outputs)
    version_value_match = _compare_values(e2_equiv_outputs, e1_outputs)
    version_key_diff = _count_key_differences(e2_equiv_outputs, e1_outputs)

    ledger_events = [
        {"kind": event.kind, "entry_key": event.entry_key, "reason": event.reason}
        for event in inc_events
    ]
    summary = {
        "baseline": {
            "batches": len(first_diag and [1]) or 1,
            "stage_points": first_diag.stage_points,
            "completed_ops": first_diag.completed_ops,
        },
        "incremental": {
            "applied_edits": outcome.applied,
            "invalidated": invalidated_now,
            "reused": reused_now,
            "pending": inc_diag.pending,
            "code_counts": inc_diag.code_counts,
            "stage_points": inc_diag.stage_points,
        },
        "incremental_vs_clean_rebuild": incremental_match,
        "e2_vs_e1": {
            "identical_integer_outputs": version_value_match,
            "cache_keys_that_differ_by_version": version_key_diff,
        },
        "ledger_events": ledger_events,
        "issue_codes": log.count_by_code(),
    }
    out_path = workspace / "run" / "acceptance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _compare(left: dict, right: dict) -> dict:
    keys = set(left) | set(right)
    mismatches = []
    for key in sorted(keys):
        if key not in left or key not in right:
            mismatches.append({"entry": key, "reason": "missing-on-one-side"})
            continue
        for field in ("cache_key", "output_fp", "output"):
            if left[key][field] != right[key][field]:
                mismatches.append({"entry": key, "field": field})
    return {"entries_compared": len(keys), "mismatches": mismatches,
            "consistent": not mismatches}


def _compare_values(left: dict, right: dict) -> bool:
    return all(left[key]["output"] == right[key]["output"] for key in left if key in right)


def _count_key_differences(left: dict, right: dict) -> int:
    return sum(1 for key in left if key in right and left[key]["cache_key"] != right[key]["cache_key"])
