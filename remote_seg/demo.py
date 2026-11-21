"""End-to-end acceptance run for edits, incremental reuse and full rebuild."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .edits import apply_edit_operation
from .engine import RunOptions, run_until_complete
from .samplegen import generate_sample_workspace
from .util import atomic_write_json, read_json
from .verify import compare_states, load_state


def _prepare_workspace(output_dir: Path, sample_dir: Path, config_path: Path, sequence_path: Path) -> Path:
    if not (sample_dir / "index.json").exists():
        generate_sample_workspace(sample_dir)
    workspace = output_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    shutil.copytree(sample_dir / "tiles", workspace / "tiles")
    shutil.copytree(sample_dir / "pool", workspace / "pool")
    for name in ("index.json", "fusion_versions.json", "patch_plan.json"):
        shutil.copyfile(sample_dir / name, workspace / name)
    shutil.copyfile(config_path, workspace / "segmentation.json")
    shutil.copyfile(sequence_path, workspace / "edit_sequence.json")
    return workspace


def _run_workspace(workspace: Path, state_dir: Path, report_path: Path) -> dict[str, Any]:
    options = RunOptions(
        index_path=workspace / "index.json",
        config_path=workspace / "segmentation.json",
        fusion_versions_path=workspace / "fusion_versions.json",
        state_dir=state_dir,
        report_path=report_path,
        max_batches=100,
    )
    return run_until_complete(options)


def run_acceptance_demo(
    output_dir: Path,
    *,
    sample_dir: Path,
    config_path: Path,
    sequence_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    workspace = _prepare_workspace(output_dir, Path(sample_dir), Path(config_path), Path(sequence_path))
    incremental_state = output_dir / "incremental-state"
    sequence = read_json(workspace / "edit_sequence.json")
    initial_report = _run_workspace(
        workspace, incremental_state, output_dir / "reports" / "000-initial.json"
    )
    step_reports = []
    step_equivalence = []
    for index, operation in enumerate(sequence["operations"], start=1):
        outcome = apply_edit_operation(
            workspace,
            operation,
            index_path=workspace / "index.json",
            config_path=workspace / "segmentation.json",
            patch_plan_path=workspace / "patch_plan.json",
        )
        report = _run_workspace(
            workspace,
            incremental_state,
            output_dir / "reports" / f"{index:03d}-{operation['id']}.json",
        )
        step_full_state = output_dir / "step-full-states" / f"{index:03d}"
        step_full_report = _run_workspace(
            workspace,
            step_full_state,
            output_dir / "reports" / f"{index:03d}-{operation['id']}-full.json",
        )
        step_comparison = compare_states(
            load_state(incremental_state),
            load_state(step_full_state),
        )
        step_equivalence.append(
            {
                "step": index,
                "operation_id": operation.get("id"),
                "incremental_status": report.get("status"),
                "full_status": step_full_report.get("status"),
                "comparison": step_comparison,
            }
        )
        step_reports.append(
            {
                "operation": outcome,
                "report_status": report.get("status"),
                "snapshot_global_version": (report.get("snapshot") or {}).get("global_version"),
                "fusion_version": (report.get("snapshot") or {}).get("fusion_version"),
                "recomputed_tiles": report.get("recomputed_tiles", []),
                "recomputed_regions": report.get("recomputed_regions", []),
                "pending": (
                    report["version_convergence"][-1]["pending"]
                    if report.get("version_convergence")
                    else []
                ),
            }
        )

    full_state = output_dir / "full-rebuild-state"
    full_report = _run_workspace(workspace, full_state, output_dir / "reports" / "999-full-rebuild.json")
    comparison = compare_states(load_state(incremental_state), load_state(full_state))
    all_steps_equal = all(
        entry["comparison"]["equal"]
        and entry["incremental_status"] == "complete"
        and entry["full_status"] == "complete"
        for entry in step_equivalence
    )
    verification = {
        "status": "pass"
        if comparison["equal"]
        and full_report.get("status") == "complete"
        and all_steps_equal
        else "fail",
        "sample_dir": str(Path(sample_dir)),
        "initial": {
            "status": initial_report.get("status"),
            "global_version": (initial_report.get("snapshot") or {}).get("global_version"),
            "fusion_version": (initial_report.get("snapshot") or {}).get("fusion_version"),
        },
        "steps": step_reports,
        "step_equivalence": step_equivalence,
        "final_incremental": {
            "status": step_reports[-1]["report_status"] if step_reports else initial_report["status"],
            "global_version": (
                step_reports[-1]["snapshot_global_version"]
                if step_reports
                else (initial_report.get("snapshot") or {}).get("global_version")
            ),
            "fusion_version": (
                step_reports[-1]["fusion_version"]
                if step_reports
                else (initial_report.get("snapshot") or {}).get("fusion_version")
            ),
        },
        "final_full_rebuild": {
            "status": full_report.get("status"),
            "global_version": (full_report.get("snapshot") or {}).get("global_version"),
            "fusion_version": (full_report.get("snapshot") or {}).get("fusion_version"),
            "class_grid_summary": full_report.get("class_grid_summary"),
        },
        "comparison": comparison,
    }
    atomic_write_json(output_dir / "verification.json", verification)
    return verification
