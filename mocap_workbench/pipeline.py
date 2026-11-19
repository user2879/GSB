"""End-to-end fusion, retargeting, strategy selection and constraint solving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .consistency import compare_pose_results
from .constraints import ConstraintSet, solve_constraints
from .errors import BusinessError, ErrorCode
from .fusion import FusionResult, fuse_observations
from .generator import Scene, load_scene
from .retarget import RetargetResult, assert_retarget_tolerance, retarget_pose
from .serialization import (
    pose_frame_hashes,
    rolling_digest,
    write_json,
    write_npy,
)
from .skeleton import Skeleton
from .skeleton import rotation_vectors_from_matrices
from .strategies import (
    BudgetConfig,
    StrategyDecision,
    apply_strategy,
    select_strategy,
)


@dataclass
class PreparedStages:
    fusion: FusionResult
    retarget: RetargetResult


@dataclass
class SolveResult:
    scene: Scene
    prepared: PreparedStages
    decision: StrategyDecision
    strategy_pose: np.ndarray
    strategy_metadata: dict[str, Any]
    corrected_pose: np.ndarray
    constraint_result: dict[str, Any]
    frame_hashes: list[str]
    pose_sequence_digest: str

    def report(self) -> dict[str, Any]:
        residual = self.constraint_result["residuals"]
        names = self.constraint_result["residual_names"]
        per_constraint = []
        for index, name in enumerate(names):
            if residual.shape[0] == 0:
                maximum = 0.0
                frame = 0
            else:
                frame = int(np.argmax(residual[:, index]))
                maximum = float(residual[frame, index])
            per_constraint.append(
                {"constraint": name, "max_residual": maximum, "max_residual_frame": frame}
            )
        return {
            "frames": self.scene.frames,
            "fps": 60,
            "source_joints": self.scene.source.joint_count,
            "target_joints": self.scene.target.joint_count,
            "fusion": self.prepared.fusion.report(),
            "retarget": self.prepared.retarget.report(1e-8),
            "strategy": self.decision.to_report(),
            "strategy_execution": self.strategy_metadata,
            "constraints": {
                "feasible": bool(self.constraint_result["feasible"]),
                "iterations": int(self.constraint_result["iterations"]),
                "max_deviation_from_strategy_pose_m": float(
                    self.constraint_result["max_deviation_m"]
                ),
                "mean_deviation_from_strategy_pose_m": float(
                    self.constraint_result["mean_deviation_m"]
                ),
                "max_bone_length_error_m": float(
                    self.constraint_result["max_bone_length_error_m"]
                ),
                "per_constraint_maxima": per_constraint,
                "residual_matrix_shape": list(residual.shape),
                "residual_matrix_units": "degrees for joint/symmetry-angle constraints; metres otherwise",
            },
            "frame_hashes": {
                "algorithm": "sha256 of little-endian int64 joint positions quantised at 1e-6 m",
                "count": len(self.frame_hashes),
                "sequence_digest": self.pose_sequence_digest,
                "frames": self.frame_hashes,
            },
        }

    def consistency_payload(self) -> dict[str, Any]:
        return {
            "pose": self.corrected_pose,
            "constraint_residuals": self.constraint_result["residuals"],
            "constraint_residual_names": self.constraint_result["residual_names"],
            "boundary_frames": self.strategy_metadata.get("boundary_frames", []),
            "joint_names": list(self.scene.target.joints),
        }


def prepare_scene(scene: Scene, fusion_visibility=None, fusion_sigma=None) -> PreparedStages:
    fusion = fuse_observations(
        scene.source,
        scene.observations,
        scene.visibility,
        scene.sigma,
        visibility_override=fusion_visibility,
        sigma_override=fusion_sigma,
    )
    retarget = retarget_pose(fusion.pose, scene.source, scene.target, tolerance=1e-8)
    return PreparedStages(fusion=fusion, retarget=retarget)


def solve_prepared(
    scene: Scene,
    prepared: PreparedStages,
    decision: StrategyDecision,
    budget: BudgetConfig | None = None,
    strict_retarget: bool = False,
) -> SolveResult:
    if strict_retarget:
        assert_retarget_tolerance(prepared.retarget, 1e-8)
    strategy_pose, metadata = apply_strategy(prepared.retarget.pose, decision, budget)
    constraint_result = solve_constraints(strategy_pose, scene.target, scene.constraints)
    corrected = constraint_result["pose"]
    hashes = pose_frame_hashes(corrected)
    sequence_digest = rolling_digest(hashes)
    return SolveResult(
        scene=scene,
        prepared=prepared,
        decision=decision,
        strategy_pose=strategy_pose,
        strategy_metadata=metadata,
        corrected_pose=corrected,
        constraint_result=constraint_result,
        frame_hashes=hashes,
        pose_sequence_digest=sequence_digest,
    )


def solve_scene(
    scene: Scene,
    budget: BudgetConfig | None = None,
    strategy: str = "auto",
    strict_retarget: bool = False,
) -> SolveResult:
    budget = budget or BudgetConfig()
    decision = select_strategy(scene.frames, scene.source.joint_count, budget, requested=strategy)
    prepared = prepare_scene(scene)
    return solve_prepared(scene, prepared, decision, budget, strict_retarget=strict_retarget)


def run_all_strategies(
    scene: Scene,
    budget: BudgetConfig | None = None,
    position_tolerance_m: float = 0.05,
    residual_tolerance: float = 0.02,
) -> tuple[dict[str, SolveResult], dict[str, Any]]:
    budget = budget or BudgetConfig()
    prepared = prepare_scene(scene)
    results: dict[str, SolveResult] = {}
    unavailable: dict[str, Any] = {}
    for strategy in ("full_segment", "windowed", "sparse_anchor"):
        try:
            decision = select_strategy(scene.frames, scene.source.joint_count, budget, requested=strategy)
        except BusinessError as exc:
            if exc.code != ErrorCode.E_STRATEGY_BUDGET:
                raise
            unavailable[strategy] = exc.as_dict()
            continue
        results[strategy] = solve_prepared(scene, prepared, decision, budget)
    comparison = compare_pose_results(
        {name: result.consistency_payload() for name, result in results.items()},
        position_tolerance_m=position_tolerance_m,
        residual_tolerance=residual_tolerance,
    )
    comparison["unavailable"] = unavailable
    return results, comparison


def save_solution(result: SolveResult, output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "fused_source_pose": result.prepared.fusion.pose,
        "fusion_residual": result.prepared.fusion.least_squares_residual,
        "fusion_completed_mask": result.prepared.fusion.completed,
        "retarget_pose": result.prepared.retarget.pose,
        "retarget_roundtrip_residual": result.prepared.retarget.roundtrip_residual,
        "strategy_pose": result.strategy_pose,
        "retarget_joint_rotation_vectors": rotation_vectors_from_matrices(
            result.prepared.retarget.joint_rotations
        ),
        "corrected_target_pose": result.corrected_pose,
        "constraint_residuals": result.constraint_result["residuals"],
    }
    array_paths: dict[str, str] = {}
    array_hashes: dict[str, str] = {}
    for name, array in arrays.items():
        relative = f"arrays/{name}.npy"
        array_hashes[name] = write_npy(root / relative, array)
        array_paths[name] = relative
    report = result.report()
    hashes_path = root / "frame_hashes.json"
    write_json(
        hashes_path,
        {
            "algorithm": report["frame_hashes"]["algorithm"],
            "sequence_digest": result.pose_sequence_digest,
            "frames": result.frame_hashes,
        },
    )
    report["constraint_residual_names"] = result.constraint_result["residual_names"]
    report["arrays"] = array_paths
    report["array_sha256"] = array_hashes
    report["frame_hashes_path"] = "frame_hashes.json"
    report["ok"] = True
    write_json(root / "solution.json", report)
    return report


def load_and_solve(
    scene_dir: str | Path,
    output_dir: str | Path,
    budget: BudgetConfig | None = None,
    strategy: str = "auto",
) -> dict[str, Any]:
    scene = load_scene(scene_dir)
    result = solve_scene(scene, budget=budget, strategy=strategy)
    return save_solution(result, output_dir)
