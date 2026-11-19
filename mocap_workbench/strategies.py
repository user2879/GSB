"""Budget-aware full-segment, overlapping-window and sparse-anchor strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode


STRATEGIES = ("full_segment", "windowed", "sparse_anchor")


@dataclass(frozen=True)
class BudgetConfig:
    memory_mb: float = 512.0
    time_budget_ms: float = 12000.0
    window_size: int = 192
    window_overlap: int = 48
    sparse_anchor_step: int = 2
    smooth_lambda: float = 0.001

    def to_manifest(self) -> dict[str, Any]:
        return {
            "memory_mb": self.memory_mb,
            "time_budget_ms": self.time_budget_ms,
            "window_size": self.window_size,
            "window_overlap": self.window_overlap,
            "sparse_anchor_step": self.sparse_anchor_step,
            "smooth_lambda": self.smooth_lambda,
        }


@dataclass
class StrategyDecision:
    strategy: str
    requested: str
    estimates: dict[str, dict[str, float]]
    reason: str
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    def to_report(self) -> dict[str, Any]:
        return {
            "selected": self.strategy,
            "requested": self.requested,
            "reason": self.reason,
            "budget": self.budget.to_manifest(),
            "estimates": self.estimates,
        }


def estimate_strategies(frames: int, joints: int, budget: BudgetConfig) -> dict[str, dict[str, float]]:
    pose_state_mb = frames * joints * 3 * 8 / 1_000_000.0
    window = min(max(16, int(budget.window_size)), max(16, frames))
    step = max(1, window - min(budget.window_overlap, window // 2))
    windows = max(1, int(np.ceil(max(1, frames - window) / step)) + 1)
    anchors = max(2, int(np.ceil(frames / max(1, budget.sparse_anchor_step))) + 1) if frames > 1 else 1
    return {
        "full_segment": {
            "projected_memory_mb": float(frames * frames * 8.0 * 24.0 / 1_000_000.0),
            "projected_time_ms": float(frames * frames * 0.005 + frames * 0.01),
            "input_state_memory_mb": float(pose_state_mb),
        },
        "windowed": {
            "projected_memory_mb": float(
                window * window * 8.0 * 24.0 / 1_000_000.0 + pose_state_mb * 2.0
            ),
            "projected_time_ms": float(frames * 0.270 + windows * 2.0),
            "windows": float(windows),
            "window_size": float(window),
            "window_step": float(step),
        },
        "sparse_anchor": {
            "projected_memory_mb": float(pose_state_mb + anchors * 3.0 * 8.0 * 2.0 / 1_000_000.0),
            "projected_time_ms": float(frames * 0.270 + anchors * 0.02),
            "anchors": float(anchors),
        },
    }


def _fits(estimate: dict[str, float], budget: BudgetConfig) -> bool:
    return (
        estimate["projected_memory_mb"] <= budget.memory_mb + 1e-12
        and estimate["projected_time_ms"] <= budget.time_budget_ms + 1e-12
    )


def select_strategy(
    frames: int,
    joints: int,
    budget: BudgetConfig | None = None,
    requested: str = "auto",
) -> StrategyDecision:
    budget = budget or BudgetConfig()
    if requested not in ("auto",) + STRATEGIES:
        raise BusinessError(ErrorCode.E_INPUT, "unknown strategy", requested=requested)
    estimates = estimate_strategies(frames, joints, budget)
    if requested != "auto":
        if not _fits(estimates[requested], budget):
            raise BusinessError(
                ErrorCode.E_STRATEGY_BUDGET,
                "requested strategy exceeds the declared memory or time budget",
                requested=requested,
                estimates=estimates,
                budget=budget.to_manifest(),
            )
        return StrategyDecision(
            strategy=requested,
            requested=requested,
            estimates=estimates,
            reason="explicitly requested and inside budget",
            budget=budget,
        )

    if _fits(estimates["full_segment"], budget):
        return StrategyDecision(
            strategy="full_segment",
            requested=requested,
            estimates=estimates,
            reason="full-segment joint optimisation fits both declared budgets",
            budget=budget,
        )
    if _fits(estimates["windowed"], budget):
        if (
            frames >= 10_000
            and _fits(estimates["sparse_anchor"], budget)
            and estimates["sparse_anchor"]["projected_time_ms"]
            < estimates["windowed"]["projected_time_ms"]
        ):
            return StrategyDecision(
                strategy="sparse_anchor",
                requested=requested,
                estimates=estimates,
                reason="windowed and sparse both fit; long-sequence policy selects lower-time sparse anchors",
                budget=budget,
            )
        return StrategyDecision(
            strategy="windowed",
            requested=requested,
            estimates=estimates,
            reason="full-segment exceeds budget; overlapping windows fit both budgets",
            budget=budget,
        )
    if _fits(estimates["sparse_anchor"], budget):
        return StrategyDecision(
            strategy="sparse_anchor",
            requested=requested,
            estimates=estimates,
            reason="window accumulation exceeds time budget; sparse anchors fit both budgets",
            budget=budget,
        )
    raise BusinessError(
        ErrorCode.E_STRATEGY_BUDGET,
        "no available strategy fits the declared memory and time budgets",
        estimates=estimates,
        budget=budget.to_manifest(),
    )


def _solve_tridiagonal(anchor: np.ndarray, lam: float) -> np.ndarray:
    """Solve (I + lambda D'D) q = anchor with natural boundaries."""
    count = anchor.shape[0]
    if count <= 2:
        return anchor.copy()
    lower = np.full(count - 1, -lam, dtype=np.float64)
    upper = np.full(count - 1, -lam, dtype=np.float64)
    diagonal = np.ones(count, dtype=np.float64)
    diagonal[0] = 1.0 + lam
    diagonal[-1] = 1.0 + lam
    diagonal[1:-1] = 1.0 + 2.0 * lam

    c = np.zeros(count - 1, dtype=np.float64)
    d = np.zeros_like(anchor)
    c[0] = upper[0] / diagonal[0]
    d[0] = anchor[0] / diagonal[0]
    for row in range(1, count):
        denom = diagonal[row] - lower[row - 1] * c[row - 1]
        if row < count - 1:
            c[row] = upper[row] / denom
        d[row] = (anchor[row] - lower[row - 1] * d[row - 1]) / denom
    result = np.zeros_like(anchor)
    result[-1] = d[-1]
    for row in range(count - 2, -1, -1):
        result[row] = d[row] - c[row] * result[row + 1]
    return result


def _solve_dense(anchor: np.ndarray, lam: float) -> np.ndarray:
    count = anchor.shape[0]
    if count <= 2:
        return anchor.copy()
    matrix = np.eye(count, dtype=np.float64)
    if count > 1:
        difference = np.zeros((count - 1, count), dtype=np.float64)
        indexes = np.arange(count - 1)
        difference[indexes, indexes] = -1.0
        difference[indexes, indexes + 1] = 1.0
        matrix += lam * (difference.T @ difference)
    return np.linalg.solve(matrix, anchor)


def _overlap_weights(length: int, ramp_in: int, ramp_out: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float64)
    if ramp_in > 0:
        weights[:ramp_in] = (np.arange(ramp_in, dtype=np.float64) + 0.5) / ramp_in
    if ramp_out > 0:
        weights[-ramp_out:] = (np.arange(ramp_out, 0, -1, dtype=np.float64) - 0.5) / ramp_out
    return weights


def _apply_root_trajectory(pose: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    delta = trajectory - pose[:, 0, :]
    return pose + delta[:, None, :]


def apply_strategy(
    pose: np.ndarray,
    decision: StrategyDecision,
    budget: BudgetConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    budget = budget or decision.budget
    pose = np.asarray(pose, dtype=np.float64)
    frames = pose.shape[0]
    anchor_roots = pose[:, 0, :].copy()
    lam = float(budget.smooth_lambda)
    boundary_frames: list[int] = []
    anchor_frames: list[int] = []
    if decision.strategy == "full_segment":
        trajectory = _solve_dense(anchor_roots, lam)
    elif decision.strategy == "windowed":
        window = min(max(16, int(budget.window_size)), max(16, frames))
        overlap = min(max(1, int(budget.window_overlap)), window // 2)
        step = max(1, window - overlap)
        accumulated = np.zeros_like(anchor_roots)
        weight_sum = np.zeros((frames,), dtype=np.float64)
        start = 0
        while start < frames:
            end = min(frames, start + window)
            local = _solve_tridiagonal(anchor_roots[start:end], lam)
            ramp_in = overlap if start > 0 else 0
            ramp_out = overlap if end < frames else 0
            weights = _overlap_weights(end - start, ramp_in, ramp_out)
            accumulated[start:end] += local * weights[:, None]
            weight_sum[start:end] += weights
            if start > 0:
                boundary_frames.append(start)
            if end >= frames:
                break
            start += step
        trajectory = accumulated / np.maximum(weight_sum[:, None], 1e-15)
    elif decision.strategy == "sparse_anchor":
        step = max(1, int(budget.sparse_anchor_step))
        indexes = np.arange(0, frames, step, dtype=np.int64)
        if indexes.size == 0 or indexes[-1] != frames - 1:
            indexes = np.append(indexes, frames - 1)
        anchor_roots_local = anchor_roots[indexes]
        solved = _solve_tridiagonal(anchor_roots_local, lam)
        trajectory = np.empty_like(anchor_roots)
        for channel in range(3):
            trajectory[:, channel] = np.interp(np.arange(frames), indexes, solved[:, channel])
        anchor_frames = indexes.astype(int).tolist()
        boundary_frames = anchor_frames[1:]
    else:
        raise BusinessError(ErrorCode.E_INPUT, "unknown strategy", strategy=decision.strategy)

    result = _apply_root_trajectory(pose, trajectory)
    if boundary_frames:
        jumps = []
        root_seams = []
        for frame in boundary_frames:
            if 0 < frame < frames:
                jumps.append(float(np.max(np.linalg.norm(result[frame] - result[frame - 1], axis=-1))))
                if frame + 1 < frames:
                    root = result[:, 0, :]
                    root_seams.append(
                        float(
                            np.linalg.norm(
                                root[frame] - 0.5 * (root[frame - 1] + root[frame + 1])
                            )
                        )
                    )
        boundary_jump = max(jumps) if jumps else 0.0
        root_seam = max(root_seams) if root_seams else 0.0
    else:
        boundary_jump = 0.0
        root_seam = 0.0
    seam_tolerance = 0.05
    return result, {
        "strategy": decision.strategy,
        "boundary_frames": boundary_frames,
        "anchor_frames": anchor_frames,
        "boundary_max_pose_jump_m": boundary_jump,
        "boundary_root_seam_m": root_seam,
        "boundary_continuity_tolerance_m": seam_tolerance,
        "boundary_continuity_ok": bool(root_seam <= seam_tolerance),
        "root_trajectory_sha256_source": "deterministic_smoothing",
        "smooth_lambda": lam,
    }
