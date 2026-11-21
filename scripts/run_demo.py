"""End-to-end deterministic demo: incremental run, full rebuild, injection."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from workbench.injections import run_injections  # noqa: E402
from workbench.io_utils import read_json  # noqa: E402
from workbench.pipeline import (  # noqa: E402
    Pipeline,
    compare_runners,
    run_full_rebuild,
    write_comparison_report,
)

VOCAB = os.path.join(ROOT, "data/vocab/vocabulary.json")
CONFIG = os.path.join(ROOT, "data/config/hierarchy_config.json")
SAMPLES = os.path.join(ROOT, "data/samples/batch_0.jsonl")
PLAN = os.path.join(ROOT, "data/plans/edit_plan.json")
INJECT = os.path.join(ROOT, "data/injections/injection_manifest.json")
SNAP_INC = os.path.join(ROOT, "snapshots/incremental")
SNAP_FULL = os.path.join(ROOT, "snapshots/full_rebuild")
REPORTS = os.path.join(ROOT, "reports")


def main():
    plan = read_json(PLAN)
    pipeline = Pipeline(VOCAB, CONFIG, SAMPLES, SNAP_INC)
    outcome = pipeline.run_plan(plan)
    incremental = outcome["runner"]

    rebuilt = run_full_rebuild(VOCAB, CONFIG, SAMPLES, plan, SNAP_FULL)
    comparison = compare_runners(incremental, rebuilt)
    write_comparison_report(os.path.join(REPORTS, "consistency_report.json"),
                            comparison)

    injection_manifest = read_json(INJECT)
    injection_report = run_injections(
        VOCAB, CONFIG, injection_manifest,
        os.path.join(REPORTS, "injection_report.json"))

    summary = {
        "batches": len(incremental.batch_reports),
        "result_count": len(incremental.results),
        "failure_count": len(incremental.failures),
        "ledger_entries": len(incremental.ledger),
        "incremental_full_rebuild_identical": comparison["identical"],
        "injections_total": injection_report["total"],
        "injections_all_hit": injection_report["all_hit"],
        "snapshot_dirs": [SNAP_INC, SNAP_FULL],
    }
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "run_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not comparison["identical"]:
        print("MISMATCH:", json.dumps(comparison["mismatches"], indent=2))
        return 1
    if not injection_report["all_hit"]:
        print("INJECTION MISS:", json.dumps(injection_report["results"], indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
