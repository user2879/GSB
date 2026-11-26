from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .engine import ReviewEngine, projections_equal
from .state import WorkspaceState, apply_command, load_commands, load_workspace


def run_injection_suite(
    base_state: WorkspaceState, work_root: str | Path
) -> dict[str, Any]:
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_dir = root / "baseline"
    shutil.rmtree(baseline_dir, ignore_errors=True)
    baseline = ReviewEngine(base_state.clone(), baseline_dir).run()

    cases: list[dict[str, Any]] = []
    for injection in base_state.injections:
        injection_id = injection["injection_id"]
        case_dir = root / "injections" / injection_id
        shutil.rmtree(case_dir, ignore_errors=True)
        result = ReviewEngine(
            base_state.clone(), case_dir, injection=injection
        ).run()
        report = result["report"]
        hits = [
            hit
            for hit in report["injection_hits"]
            if hit["injection_id"] == injection_id
        ]
        cases.append(
            {
                "injection_id": injection_id,
                "configured": injection,
                "hit": bool(hits),
                "hits": hits,
                "status": report["status"],
                "business_code_counts": report["business_code_counts"],
                "stage_fault_code_counts": report["stage_fault_code_counts"],
                "stage_faults": report["stage_faults"],
                "stage_completion": report["stage_completion"],
                "surviving_decision_event_ids": [
                    decision["event_id"]
                    for decision in result["stage_outputs"]["adjudication"]["decisions"]
                ],
            }
        )

    return {
        "all_injections_hit": all(case["hit"] for case in cases),
        "injection_count": len(cases),
        "hit_count": sum(1 for case in cases if case["hit"]),
        "cases": cases,
        "baseline": {
            "status": baseline["report"]["status"],
            "business_code_counts": baseline["report"]["business_code_counts"],
            "stage_completion": baseline["report"]["stage_completion"],
            "projection_fingerprint": baseline["projection"]["projection_fingerprint"],
        },
    }


def run_edit_equivalence(
    sample_dir: str | Path,
    command_path: str | Path,
    work_root: str | Path,
) -> dict[str, Any]:
    root = Path(work_root)
    incremental_dir = root / "incremental"
    shutil.rmtree(incremental_dir, ignore_errors=True)
    shutil.rmtree(root / "full-runs", ignore_errors=True)
    incremental_dir.mkdir(parents=True, exist_ok=True)

    incremental_state = load_workspace(sample_dir)
    commands = load_commands(command_path)
    steps: list[dict[str, Any]] = []
    all_equal = True

    for index, command in enumerate(commands, 1):
        apply_command(incremental_state, command)
        incremental = ReviewEngine(incremental_state, incremental_dir).run()

        full_state = incremental_state.clone()
        full_dir = root / "full-runs" / f"step-{index:02d}"
        shutil.rmtree(full_dir, ignore_errors=True)
        full = ReviewEngine(full_state, full_dir).run()

        equal = projections_equal(incremental["projection"], full["projection"])
        all_equal = all_equal and equal
        steps.append(
            {
                "step": index,
                "op": command["op"],
                "global_version": incremental_state.global_version,
                "equal": equal,
                "incremental_projection_fingerprint": incremental["projection"][
                    "projection_fingerprint"
                ],
                "full_projection_fingerprint": full["projection"][
                    "projection_fingerprint"
                ],
                "ledger_summary": incremental["report"]["ledger_summary"],
                "restricted_recompute_set": incremental["ledger_lists"][
                    "restricted_recompute_set"
                ],
                "reuse_artifact_ids": incremental["ledger_lists"]["reuse_artifact_ids"],
                "orders": incremental["projection"]["orders"],
            }
        )
        if not equal:
            break

    return {
        "all_steps_equal": all_equal,
        "command_count": len(commands),
        "completed_steps": len(steps),
        "steps": steps,
    }
