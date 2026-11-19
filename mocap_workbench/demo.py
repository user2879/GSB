"""Deterministic one-command demonstration across all three data scales."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .consistency import compare_pose_results
from .constraints import ConstraintSet, FootContact, solve_constraints
from .errors import BusinessError, ErrorCode
from .fusion import fuse_observations
from .generator import GenerationConfig, generate_scene, save_scene, scene_file_hashes
from .incremental import IncrementalWorkbench
from .pipeline import run_all_strategies, save_solution, solve_scene
from .retarget import assert_retarget_tolerance, retarget_pose
from .serialization import canonical_json_bytes, rolling_digest, sha256_file, write_json
from .strategies import BudgetConfig, select_strategy


def _error_checks(seed: int) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    scene = generate_scene(GenerationConfig(frames=30, seed=seed))

    try:
        hidden = np.zeros_like(scene.visibility)
        fuse_observations(scene.source, scene.observations, scene.visibility, scene.sigma, visibility_override=hidden)
        checks[ErrorCode.E_OBS_MISSING.value] = {"raised": False}
    except BusinessError as exc:
        checks[exc.code.value] = {"raised": True, "exit_code": exc.as_dict()["exit_code"], "message": exc.message}

    fused = fuse_observations(scene.source, scene.observations, scene.visibility, scene.sigma)
    retargeted = retarget_pose(fused.pose, scene.source, scene.target)
    retargeted.roundtrip_residual[0, 0] = 1.0
    try:
        assert_retarget_tolerance(retargeted, 1e-8)
        checks[ErrorCode.E_RETARGET_TOL.value] = {"raised": False}
    except BusinessError as exc:
        checks[exc.code.value] = {
            "raised": True,
            "exit_code": exc.as_dict()["exit_code"],
            "message": exc.message,
            "frame": exc.details.get("frame"),
            "joint": exc.details.get("joint"),
        }

    impossible = replace(
        scene.constraints,
        foot_contacts=[
            FootContact("l_ankle", (100.0, 100.0, 100.0), period=1, phase=0, duration=14),
            FootContact("r_ankle", (100.0, 100.0, 100.0), period=1, phase=0, duration=14),
        ],
    )
    try:
        solve_constraints(retargeted.pose[:14], scene.target, impossible)
        checks[ErrorCode.E_CONSTRAINT_UNSAT.value] = {"raised": False}
    except BusinessError as exc:
        checks[exc.code.value] = {
            "raised": True,
            "exit_code": exc.as_dict()["exit_code"],
            "message": exc.message,
            "conflict_constraints": exc.details.get("conflict_constraints"),
            "unreachable_contact_frames": exc.details.get("unreachable_contact_frames"),
        }

    try:
        select_strategy(30000, 17, BudgetConfig(memory_mb=0.001, time_budget_ms=0.001))
        checks[ErrorCode.E_STRATEGY_BUDGET.value] = {"raised": False}
    except BusinessError as exc:
        checks[exc.code.value] = {"raised": True, "exit_code": exc.as_dict()["exit_code"], "message": exc.message}

    workbench = IncrementalWorkbench(scene)
    try:
        workbench._verify_closure({("fusion", 0)}, set())
        checks[ErrorCode.E_CLOSURE_MISMATCH.value] = {"raised": False}
    except BusinessError as exc:
        checks[exc.code.value] = {
            "raised": True,
            "exit_code": exc.as_dict()["exit_code"],
            "message": exc.message,
            "declared_only": exc.details.get("declared_only"),
            "computed_only": exc.details.get("computed_only"),
        }
    return checks


def run_demo(
    output_dir: str | Path,
    sizes: tuple[int, ...] = (300, 3000, 30000),
    seed: int = 20261119,
    budget: BudgetConfig | None = None,
    determinism_check: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    budget = budget or BudgetConfig()
    size_reports: list[dict[str, Any]] = []
    generated_by_size: dict[int, Any] = {}
    for frames in sizes:
        if frames <= 0:
            raise BusinessError(ErrorCode.E_INPUT, "demo sizes must be positive", frames=frames)
        scene_dir = output / "scenes" / str(frames)
        solution_dir = output / "solutions" / str(frames)
        config = GenerationConfig(frames=frames, seed=seed)
        scene = generate_scene(config)
        repeat_identical = True
        if determinism_check:
            repeat_identical = generate_scene(config).digest() == scene.digest()
        manifest = save_scene(scene, scene_dir)
        auto = solve_scene(scene, budget=budget)
        solution_report = save_solution(auto, solution_dir)
        strategy_results, consistency = run_all_strategies(scene, budget=budget)
        write_json(solution_dir / "consistency.json", consistency)
        generated_by_size[frames] = scene
        size_reports.append(
            {
                "frames": frames,
                "scene_directory": f"scenes/{frames}",
                "solution_directory": f"solutions/{frames}",
                "scene_digest": manifest["scene_digest"],
                "repeat_generation_identical": repeat_identical,
                "selected_strategy": auto.decision.strategy,
                "strategy_decision": auto.decision.to_report(),
                "pose_sequence_digest": auto.pose_sequence_digest,
                "frame_hashes_path": f"solutions/{frames}/frame_hashes.json",
                "constraint_residuals_path": f"solutions/{frames}/arrays/constraint_residuals.npy",
                "constraint_residual_names": auto.constraint_result["residual_names"],
                "constraint_report": auto.report()["constraints"],
                "fusion_report": auto.report()["fusion"],
                "retarget_report": auto.report()["retarget"],
                "strategy_boundaries": auto.strategy_metadata.get("boundary_frames", []),
                "strategy_execution": auto.strategy_metadata,
                "consistency": consistency,
                "scene_files_sha256": scene_file_hashes(scene_dir),
                "solution_arrays_sha256": solution_report["array_sha256"],
                "frame_hashes_sha256": sha256_file(solution_dir / "frame_hashes.json"),
            }
        )

    incremental_dir = output / "incremental"
    incremental_scene = generated_by_size.get(300)
    if incremental_scene is None:
        incremental_scene = generate_scene(GenerationConfig(frames=300, seed=seed))
    workbench = IncrementalWorkbench(incremental_scene, budget=budget)
    root_z = float(workbench.corrected_poses[57, 0, 2])
    edit_reports = []
    edit_reports.append(
        workbench.edit_frame_constraint(57, root_height=(root_z - 0.003, root_z + 0.003))
    )
    visible_camera = int(
        np.flatnonzero(incremental_scene.visibility[120, incremental_scene.source.index["l_elbow"]])[0]
    )
    edit_reports.append(workbench.edit_camera_weight(120, "l_elbow", visible_camera, 1.75))
    original_knee_length = float(incremental_scene.target.bone_lengths[incremental_scene.target.index["l_knee"]])
    edit_reports.append(workbench.edit_target_bone_length("l_knee", original_knee_length * 1.002))
    incremental_report = workbench.save(incremental_dir)

    error_checks = _error_checks(seed)
    report = {
        "ok": True,
        "format": "mocap-workbench-demo-v1",
        "seed": seed,
        "fps": 60,
        "sizes": list(sizes),
        "budget": budget.to_manifest(),
        "sizes_report": size_reports,
        "incremental": {
            "ledger_path": "incremental/incremental_ledger.json",
            "final_pose_sequence_digest": incremental_report["final_pose_sequence_digest"],
            "edits": [
                {
                    "edit": item["edit"],
                    "ledger_version": item["ledger_version"],
                    "closure_counts": item["closure_counts"],
                    "reuse_counts": item["reuse_counts"],
                    "closure_equals_actual": item["closure_equals_actual"],
                }
                for item in edit_reports
            ],
        },
        "error_code_checks": error_checks,
        "report_sha256_inputs": {
            "scene_digests": {str(item["frames"]): item["scene_digest"] for item in size_reports},
            "pose_sequence_digests": {
                str(item["frames"]): item["pose_sequence_digest"] for item in size_reports
            },
        },
    }
    write_json(output / "demo_report.json", report)
    return report


def demo_stdout(report: dict[str, Any]) -> None:
    import sys

    sys.stdout.buffer.write(canonical_json_bytes(report))
    sys.stdout.buffer.flush()
