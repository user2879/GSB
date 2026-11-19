"""Window-aware incremental editing with exact invalidation closures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constraints import (
    ConstraintSet,
    FootContact,
    RootHeight,
    solve_constraints,
)
from .errors import BusinessError, ErrorCode
from .fusion import fuse_observations
from .generator import Scene
from .pipeline import PreparedStages, prepare_scene
from .retarget import retarget_pose
from .serialization import pose_frame_hashes, rolling_digest, write_json
from .skeleton import Skeleton
from .strategies import BudgetConfig, StrategyDecision, _overlap_weights, _solve_tridiagonal, select_strategy


STAGES = ("fusion", "retarget", "strategy", "constraints")


@dataclass
class WindowSpec:
    start: int
    end: int
    weights: np.ndarray


def target_with_bone_length(target: Skeleton, joint_name: str, new_length: float) -> Skeleton:
    if joint_name not in target.index:
        raise BusinessError(ErrorCode.E_INPUT, "unknown target bone", joint=joint_name)
    joint = target.index[joint_name]
    if joint == 0:
        raise BusinessError(ErrorCode.E_INPUT, "the root has no parent bone")
    if new_length <= 1e-6:
        raise BusinessError(ErrorCode.E_INPUT, "bone length must be positive", length=new_length)
    rest = target.rest_world.copy()
    parent = int(target.parents[joint])
    offset = rest[joint] - rest[parent]
    old_length = float(np.linalg.norm(offset))
    direction = offset / old_length
    rest[target.subtree_indices(joint, include_self=True)] += direction * (float(new_length) - old_length)
    return Skeleton.from_positions(target.name, target.joints, target.parents, rest)


class IncrementalWorkbench:
    """Frame-cached workbench using an overlapping-window strategy.

    Window local solves are independent, so a camera-weight edit invalidates
    only windows containing that frame. Constraint edits are frame-local.
    """

    def __init__(self, scene: Scene, budget: BudgetConfig | None = None):
        self.scene = scene
        self.budget = budget or BudgetConfig()
        self.decision = select_strategy(
            scene.frames, scene.target.joint_count, self.budget, requested="windowed"
        )
        self.version = 0
        self._frame_overrides: dict[int, dict[str, Any]] = {}
        prepared = prepare_scene(scene)
        self.fusion_poses = prepared.fusion.pose.copy()
        self.retarget_poses = prepared.retarget.pose.copy()
        self.retarget_result = prepared.retarget
        self.sigma = scene.sigma.copy()
        self.window_size = min(max(16, int(self.budget.window_size)), max(16, scene.frames))
        self.overlap = min(max(1, int(self.budget.window_overlap)), self.window_size // 2)
        self.step = max(1, self.window_size - self.overlap)
        self.windows = self._build_windows(scene.frames)
        self.frame_windows: list[list[int]] = [[] for _ in range(scene.frames)]
        for window_index, window in enumerate(self.windows):
            for frame in range(window.start, window.end):
                self.frame_windows[frame].append(window_index)
        self.window_roots = [None for _ in self.windows]
        self._solve_all_windows()
        self.strategy_poses = self._assemble_strategy_poses(np.arange(scene.frames))
        self.corrected_poses = np.empty_like(self.retarget_poses)
        self.residuals = np.zeros((scene.frames, len(self._constraint_names)), dtype=np.float64)
        self._solve_constraint_frames(np.arange(scene.frames))
        self.versions = {stage: np.zeros((scene.frames,), dtype=np.int64) for stage in STAGES}
        self.last_edit_report: dict[str, Any] | None = None
        self.edit_history: list[dict[str, Any]] = []

    def _build_windows(self, frames: int) -> list[WindowSpec]:
        windows: list[WindowSpec] = []
        start = 0
        while start < frames:
            end = min(frames, start + self.window_size)
            ramp_in = self.overlap if start > 0 else 0
            ramp_out = self.overlap if end < frames else 0
            windows.append(
                WindowSpec(
                    start=start,
                    end=end,
                    weights=_overlap_weights(end - start, ramp_in, ramp_out),
                )
            )
            if end >= frames:
                break
            start += self.step
        return windows

    def _solve_all_windows(self) -> None:
        for index in range(len(self.windows)):
            self._solve_window(index)

    def _solve_window(self, index: int) -> None:
        window = self.windows[index]
        anchor = self.retarget_poses[window.start : window.end, 0, :]
        self.window_roots[index] = _solve_tridiagonal(anchor, self.budget.smooth_lambda)

    def _assemble_strategy_poses(self, frames: np.ndarray) -> np.ndarray:
        requested = np.asarray(frames, dtype=np.int64)
        out = self.retarget_poses[requested].copy()
        for local_index, frame in enumerate(requested):
            windows = self.frame_windows[int(frame)]
            if not windows:
                raise BusinessError(ErrorCode.E_CLOSURE_MISMATCH, "frame is not covered by a window", frame=int(frame))
            root_accum = np.zeros((3,), dtype=np.float64)
            weight_sum = 0.0
            for window_index in windows:
                window = self.windows[window_index]
                local_index = int(frame) - window.start
                weight = float(window.weights[local_index])
                root_accum += weight * self.window_roots[window_index][local_index]
                weight_sum += weight
            root = root_accum / max(weight_sum, 1e-15)
            out[local_index] += root - out[local_index, 0, :]
        return out

    @property
    def _constraint_names(self) -> list[str]:
        from .constraints import constraint_names

        return constraint_names(self.scene.target, self.scene.constraints)

    def _constraints_for_frame(self, frame: int) -> ConstraintSet:
        base = self.scene.constraints
        active_contacts = [contact.active(self.scene.frames)[frame] for contact in base.foot_contacts]
        contacts: list[FootContact] = []
        for contact, active in zip(base.foot_contacts, active_contacts):
            point = tuple(
                float(x)
                for x in self._frame_overrides.get(frame, {}).get(
                    f"foot_point:{contact.joint}", contact.point
                )
            )
            contacts.append(
                FootContact(
                    joint=contact.joint,
                    point=point,
                    period=1,
                    phase=0,
                    duration=1 if active else 0,
                )
            )
        root_height = base.root_height
        override = self._frame_overrides.get(frame, {}).get("root_height")
        if override is not None:
            root_height = RootHeight(float(override[0]), float(override[1]))
        return ConstraintSet(
            joint_limits=list(base.joint_limits),
            foot_contacts=contacts,
            root_height=root_height,
            symmetries=list(base.symmetries),
        )

    def _solve_constraint_frames(self, frames: np.ndarray) -> None:
        requested = np.asarray(frames, dtype=np.int64)
        override_frames = [int(frame) for frame in requested if int(frame) in self._frame_overrides]
        plain_frames = np.asarray(
            [int(frame) for frame in requested if int(frame) not in self._frame_overrides],
            dtype=np.int64,
        )
        if plain_frames.size:
            solved = solve_constraints(
                self.strategy_poses[plain_frames],
                self.scene.target,
                self.scene.constraints,
                frame_indices=plain_frames,
            )
            self.corrected_poses[plain_frames] = solved["pose"]
            rows = solved["residuals"]
            if rows.shape[1] != self.residuals.shape[1]:
                raise BusinessError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "constraint residual layout changed during incremental batch solve",
                )
            self.residuals[plain_frames] = rows
        for frame in override_frames:
            constraints = self._constraints_for_frame(int(frame))
            solved = solve_constraints(
                self.strategy_poses[int(frame) : int(frame) + 1], self.scene.target, constraints
            )
            self.corrected_poses[int(frame)] = solved["pose"][0]
            row = solved["residuals"][0]
            if row.shape[0] != self.residuals.shape[1]:
                raise BusinessError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "constraint residual layout changed during incremental solve",
                    frame=int(frame),
                )
            self.residuals[int(frame)] = row

    def _verify_closure(
        self, declared: set[tuple[str, int]], computed: set[tuple[str, int]]
    ) -> None:
        if declared != computed:
            raise BusinessError(
                ErrorCode.E_CLOSURE_MISMATCH,
                "declared invalidation closure differs from the actually recomputed set",
                declared_only=sorted(declared - computed)[:100],
                computed_only=sorted(computed - declared)[:100],
            )

    def _ledger(
        self, declared: set[tuple[str, int]], computed: set[tuple[str, int]], reason: str
    ) -> dict[str, Any]:
        self._verify_closure(declared, computed)
        self.version += 1
        records: list[dict[str, Any]] = []
        reuse_counts = {stage: 0 for stage in STAGES}
        compute_counts = {stage: 0 for stage in STAGES}
        for stage in STAGES:
            for frame in range(self.scene.frames):
                key = (stage, frame)
                if key in computed:
                    source_version = int(self.versions[stage][frame])
                    self.versions[stage][frame] = self.version
                    state = "computed"
                    compute_counts[stage] += 1
                else:
                    source_version = int(self.versions[stage][frame])
                    state = "reused"
                    reuse_counts[stage] += 1
                records.append(
                    {
                        "stage": stage,
                        "frame": frame,
                        "state": state,
                        "source_version": source_version,
                        "ledger_version": self.version,
                    }
                )
        report = {
            "edit": reason,
            "ledger_version": self.version,
            "closure": [
                {"stage": stage, "frame": frame} for stage, frame in sorted(declared)
            ],
            "closure_counts": compute_counts,
            "reuse_counts": reuse_counts,
            "closure_equals_actual": True,
            "records": records,
        }
        self.last_edit_report = report
        self.edit_history.append(report)
        return report

    def edit_camera_weight(
        self, frame: int, joint: str, camera: int, weight_multiplier: float
    ) -> dict[str, Any]:
        if not 0 <= frame < self.scene.frames:
            raise BusinessError(ErrorCode.E_INPUT, "frame is outside the scene", frame=frame)
        if joint not in self.scene.source.index:
            raise BusinessError(ErrorCode.E_INPUT, "unknown observation joint", joint=joint)
        if not 0 <= camera < len(self.scene.cameras):
            raise BusinessError(ErrorCode.E_INPUT, "unknown camera", camera=camera)
        if weight_multiplier <= 0:
            raise BusinessError(ErrorCode.E_INPUT, "weight multiplier must be positive")
        joint_index = self.scene.source.index[joint]
        if not self.scene.visibility[frame, joint_index, camera]:
            raise BusinessError(
                ErrorCode.E_OBS_MISSING,
                "cannot edit weight of an invisible observation",
                frame=frame,
                joint=joint,
                camera=camera,
            )
        self.sigma[frame, joint_index, camera] /= np.sqrt(weight_multiplier)
        fused = fuse_observations(
            self.scene.source,
            self.scene.observations[frame : frame + 1],
            self.scene.visibility[frame : frame + 1],
            self.sigma[frame : frame + 1],
        )
        self.fusion_poses[frame] = fused.pose[0]
        retargeted = retarget_pose(
            self.fusion_poses[frame : frame + 1], self.scene.source, self.scene.target
        )
        self.retarget_poses[frame] = retargeted.pose[0]
        affected_windows = self.frame_windows[frame]
        affected_frames: set[int] = set()
        for window_index in affected_windows:
            self._solve_window(window_index)
            window = self.windows[window_index]
            affected_frames.update(range(window.start, window.end))
        affected = np.asarray(sorted(affected_frames), dtype=np.int64)
        self.strategy_poses[affected] = self._assemble_strategy_poses(affected)
        self._solve_constraint_frames(affected)
        declared = {("fusion", frame), ("retarget", frame)}
        declared.update(("strategy", int(item)) for item in affected)
        declared.update(("constraints", int(item)) for item in affected)
        return self._ledger(declared, set(declared), "camera_observation_weight")

    def edit_frame_constraint(
        self,
        frame: int,
        root_height: tuple[float, float] | None = None,
        foot_point: tuple[str, tuple[float, float, float]] | None = None,
    ) -> dict[str, Any]:
        if not 0 <= frame < self.scene.frames:
            raise BusinessError(ErrorCode.E_INPUT, "frame is outside the scene", frame=frame)
        override = self._frame_overrides.setdefault(frame, {})
        if root_height is not None:
            if float(root_height[0]) > float(root_height[1]):
                raise BusinessError(ErrorCode.E_INPUT, "root-height minimum exceeds maximum")
            override["root_height"] = (float(root_height[0]), float(root_height[1]))
        if foot_point is not None:
            joint, point = foot_point
            override[f"foot_point:{joint}"] = tuple(float(x) for x in point)
        if not override:
            raise BusinessError(ErrorCode.E_INPUT, "no frame constraint parameter was supplied")
        self._solve_constraint_frames(np.asarray((frame,), dtype=np.int64))
        declared = {("constraints", frame)}
        return self._ledger(declared, set(declared), "frame_constraint_parameter")

    def edit_target_bone_length(self, joint: str, new_length: float) -> dict[str, Any]:
        self.scene.target = target_with_bone_length(self.scene.target, joint, new_length)
        retargeted = retarget_pose(self.fusion_poses, self.scene.source, self.scene.target)
        self.retarget_result = retargeted
        self.retarget_poses = retargeted.pose
        self._solve_all_windows()
        all_frames = np.arange(self.scene.frames, dtype=np.int64)
        self.strategy_poses = self._assemble_strategy_poses(all_frames)
        self.residuals = np.zeros(
            (self.scene.frames, len(self._constraint_names)), dtype=np.float64
        )
        self._solve_constraint_frames(all_frames)
        declared = {("retarget", int(frame)) for frame in all_frames}
        declared.update(("strategy", int(frame)) for frame in all_frames)
        declared.update(("constraints", int(frame)) for frame in all_frames)
        return self._ledger(declared, set(declared), "target_bone_length")

    def final_pose_hashes(self) -> list[str]:
        return pose_frame_hashes(self.corrected_poses)

    def save(self, output_dir: str | Path) -> dict[str, Any]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        from .serialization import write_npy

        write_npy(root / "arrays" / "incremental_corrected_pose.npy", self.corrected_poses)
        write_npy(root / "arrays" / "incremental_constraint_residuals.npy", self.residuals)
        provenance = {}
        for stage in STAGES:
            write_npy(root / "arrays" / f"{stage}_version.npy", self.versions[stage])
            provenance[stage] = {
                "version_path": f"arrays/{stage}_version.npy",
                "version_shape": [self.scene.frames],
            }
        report = {
            "ok": True,
            "strategy": self.decision.to_report(),
            "edits": self.edit_history,
            "final_pose_sequence_digest": rolling_digest(self.final_pose_hashes()),
            "provenance": provenance,
        }
        write_json(root / "incremental_ledger.json", report)
        return report
