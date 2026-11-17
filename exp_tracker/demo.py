"""Repeatable three-scenario demonstration used by CLI and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .samples import replacement_v2_document, replacement_v3_document
from .service import ExperimentService
from .util import atomic_write_json


def run_demo(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    sample_path = Path(__file__).resolve().parents[1] / "samples" / "legacy_experiment.json"

    service = ExperimentService.initialize(workspace, sample_path, force=True)
    scenario_one = {
        "scenario": "1_import_and_upgrade",
        "library_id": service.library.library_id,
        "upgrade": service.upgrade_result.to_dict(),
        "runs": service.list_runs(),
        "format_versions": service.library.format_versions(),
        "source_format_versions": service.library.source_format_versions(),
        "checkpoint": service.status()["checkpoint"],
    }

    replacement_v2 = service.commit_record(replacement_v2_document())
    scenario_two = {
        "scenario": "2_whole_record_replacement",
        "replacement": replacement_v2,
        "runs": service.list_runs(),
        "baseline_history_versions": next(
            run["history_versions"] for run in service.list_runs() if run["run_id"] == "baseline"
        ),
    }

    snapshot = service.snapshot("after-v2-before-v3")
    service.commit_record(replacement_v3_document())
    restore = service.restore_snapshot(snapshot["snapshot_id"], replay=True)
    scenario_three = {
        "scenario": "3_snapshot_restore_and_replay",
        "snapshot": snapshot,
        "restore": restore,
        "runs_after_restore": service.list_runs(),
        "final_format_versions": service.library.format_versions(),
        "final_checkpoint": service.status()["checkpoint"],
    }

    report = {
        "demo": "local-experiment-tracker",
        "workspace": str(workspace),
        "scenarios": [scenario_one, scenario_two, scenario_three],
    }
    atomic_write_json(workspace / "reports" / "demo-report.json", report)
    return report


def demo_json(workspace: Path) -> str:
    return json.dumps(
        run_demo(workspace), ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    )

