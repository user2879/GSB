"""Deterministic multi-view geometry primitives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .budget import ResidualCounter
from .config import SolverConfig
from .errors import (
    E_DISPARITY_INSUFFICIENT,
    E_TRIANG_DEGENERATE,
    BudgetExceeded,
    Failure,
    SolverFailure,
)
from .models import Camera, Observation, observations_fingerprint


def _failure(code: str, keypoint_id: str, stage: str, message: str, details: dict[str, Any] | None = None) -> Failure:
    return Failure(
        code=code,
        keypoint_id=keypoint_id,
        stage=stage,
        message=message,
        details=details or {},
    )


def _sort_observations(observations: list[Observation]) -> list[Observation]:
    return sorted(observations, key=lambda obs: obs.view_index)


def triangulate_dlt(
    observations: list[Observation],
    cameras: dict[int, Camera],
    cfg: SolverConfig,
    *,
    keypoint_id: str | None = None,
    stage: str = "triangulation",
) -> dict[str, Any]:
    """Solve the homogeneous DLT system with float64 SVD.

    Rows are length-normalized before SVD.  The second-smallest singular value
    decides rank: a second near-zero singular value means the nullspace is not
    one-dimensional (ambiguous/degenerate), and is reported accordingly.
    """

    ordered = _sort_observations(observations)
    point_id = keypoint_id or (ordered[0].keypoint_id if ordered else "")
    views = [obs.view_index for obs in ordered]
    if len(ordered) < cfg.min_views:
        raise SolverFailure(
            _failure(
                E_DISPARITY_INSUFFICIENT,
                point_id,
                stage,
                "not enough observations to support triangulation",
                {"participating_views": views, "min_views": cfg.min_views},
            ),
            {"participating_views": views},
        )
    if any(view not in cameras for view in views):
        missing = [view for view in views if view not in cameras]
        raise SolverFailure(
            _failure(
                E_DISPARITY_INSUFFICIENT,
                point_id,
                stage,
                "observations reference unavailable cameras",
                {"participating_views": views, "missing_views": missing},
            ),
            {"participating_views": views, "missing_views": missing},
        )

    rows: list[np.ndarray] = []
    for obs in ordered:
        P = cameras[obs.view_index].P.astype(np.float64, copy=False)
        x = np.array([obs.u, obs.v, 1.0], dtype=np.float64)
        rows.append(x[0] * P[2, :] - P[0, :])
        rows.append(x[1] * P[2, :] - P[1, :])
    A = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(A)):
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "triangulation matrix contains non-finite values",
                {"participating_views": views},
            ),
            {"participating_views": views},
        )
    row_norms = np.linalg.norm(A, axis=1)
    if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "triangulation matrix has a zero or invalid row",
                {"participating_views": views},
            ),
            {"participating_views": views},
        )
    A = A / row_norms[:, None]
    try:
        _, singular_values, vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                f"SVD failed: {exc}",
                {"participating_views": views},
            ),
            {"participating_views": views},
        ) from exc
    if singular_values.size < 4:
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "triangulation system has fewer than four singular values",
                {"participating_views": views},
            ),
            {"participating_views": views},
        )
    scale = float(singular_values[0]) if singular_values.size else 0.0
    threshold = cfg.degenerate_relative_tolerance * max(scale, 1.0)
    rank = int(np.count_nonzero(singular_values > threshold))
    gap = float(singular_values[-2] - singular_values[-1])
    if rank < 4 or gap <= threshold:
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "triangulation nullspace is degenerate or ambiguous",
                {
                    "participating_views": views,
                    "singular_values": [float(v) for v in singular_values],
                    "numerical_rank": rank,
                    "smallest_gap": gap,
                    "policy": cfg.degenerate_policy,
                },
            ),
            {
                "participating_views": views,
                "singular_values": [float(v) for v in singular_values],
                "numerical_rank": rank,
                "smallest_gap": gap,
            },
        )

    homogeneous = vh[-1, :].astype(np.float64)
    if not np.all(np.isfinite(homogeneous)) or abs(float(homogeneous[3])) <= 1e-14:
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "homogeneous solution has an invalid scale",
                {
                    "participating_views": views,
                    "homogeneous": homogeneous.tolist(),
                },
            ),
            {"participating_views": views},
        )
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)) or float(np.linalg.norm(point)) > 1e14:
        raise SolverFailure(
            _failure(
                E_TRIANG_DEGENERATE,
                point_id,
                stage,
                "triangulated point is non-finite or numerically unusable",
                {"participating_views": views, "point": point.tolist()},
            ),
            {"participating_views": views},
        )
    return {
        "point": point,
        "participating_views": views,
        "singular_values": [float(v) for v in singular_values],
        "numerical_rank": rank,
        "smallest_gap": gap,
        "homogeneous_solution": homogeneous.tolist(),
    }


def project_point(camera: Camera, point: np.ndarray) -> tuple[float, float, float]:
    homogeneous = np.array(
        [float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64
    )
    projected = camera.P @ homogeneous
    denominator = float(projected[2])
    if abs(denominator) <= 1e-15 or not math.isfinite(denominator):
        raise ValueError("projection denominator is zero or non-finite")
    u = float(projected[0] / denominator)
    v = float(projected[1] / denominator)
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError("projected coordinate is non-finite")
    return u, v, denominator


def compute_residuals(
    point: np.ndarray,
    observations: list[Observation],
    cameras: dict[int, Camera],
) -> list[dict[str, Any]]:
    ordered = _sort_observations(observations)
    residuals: list[dict[str, Any]] = []
    for obs in ordered:
        camera = cameras[obs.view_index]
        u_projected, v_projected, denominator = project_point(camera, point)
        dx = u_projected - obs.u
        dy = v_projected - obs.v
        residual = math.hypot(dx, dy)
        residuals.append(
            {
                "view_index": obs.view_index,
                "u": obs.u,
                "v": obs.v,
                "projected_u": u_projected,
                "projected_v": v_projected,
                "dx": dx,
                "dy": dy,
                "residual_px": residual,
                "depth": denominator,
            }
        )
    return residuals


def disparity_pairs(
    point: np.ndarray,
    observations: list[Observation],
    cameras: dict[int, Camera],
    min_disparity_deg: float,
) -> dict[str, Any]:
    """Compute pairwise viewing-ray angles and determine supported views.

    A view is retained only if it participates in at least one pair whose angle
    meets the threshold.  Isolated views are explicitly excluded; low-disparity
    pairs are never silently replaced by an all-view retriangulation.
    """

    ordered = _sort_observations(observations)
    rays: dict[int, np.ndarray] = {}
    for obs in ordered:
        center = cameras[obs.view_index].center.astype(np.float64, copy=False)
        ray = point.astype(np.float64, copy=False) - center
        norm = float(np.linalg.norm(ray))
        if norm <= 1e-15 or not math.isfinite(norm):
            rays[obs.view_index] = np.zeros(3, dtype=np.float64)
        else:
            rays[obs.view_index] = ray / norm
    views = [obs.view_index for obs in ordered]
    degree = {view: 0 for view in views}
    pairs: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    for i, left in enumerate(views):
        for right in views[i + 1 :]:
            dot = float(np.clip(np.dot(rays[left], rays[right]), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            excluded = angle < min_disparity_deg
            record = {
                "views": [left, right],
                "angle_deg": angle,
                "excluded": excluded,
            }
            pairs.append(record)
            if excluded:
                excluded_pairs.append(record)
            else:
                degree[left] += 1
                degree[right] += 1
    accepted_views = [view for view in views if degree[view] > 0]
    excluded_views = [view for view in views if degree[view] == 0]
    return {
        "pairs": pairs,
        "excluded_view_pairs": excluded_pairs,
        "accepted_views": accepted_views,
        "excluded_views": excluded_views,
        "degree": {str(view): degree[view] for view in views},
    }


def triangulation_stage(
    observations: list[Observation],
    cameras: dict[int, Camera],
    cfg: SolverConfig,
    *,
    keypoint_id: str,
    residual_counter: ResidualCounter | None = None,
) -> dict[str, Any]:
    """Triangulate, apply disparity screening, and triangulate the retained set."""

    ordered = _sort_observations(observations)
    all_views = [obs.view_index for obs in ordered]
    initial = triangulate_dlt(
        ordered, cameras, cfg, keypoint_id=keypoint_id, stage="triangulation"
    )
    disparity = disparity_pairs(
        initial["point"], ordered, cameras, cfg.min_disparity_deg
    )
    accepted_views = disparity["accepted_views"]
    accepted_set = set(accepted_views)
    accepted_observations = [obs for obs in ordered if obs.view_index in accepted_set]
    if len(accepted_observations) < cfg.min_views:
        failure = _failure(
            E_DISPARITY_INSUFFICIENT,
            keypoint_id,
            "triangulation",
            "minimum-disparity screening leaves too few views",
            {
                "participating_views": all_views,
                "accepted_views": accepted_views,
                "excluded_views": disparity["excluded_views"],
                "excluded_view_pairs": disparity["excluded_view_pairs"],
                "min_views": cfg.min_views,
            },
        )
        raise SolverFailure(
            failure,
            {
                "initial_point": initial["point"].tolist(),
                "initial_singular_values": initial["singular_values"],
                "participating_views": all_views,
                "disparity": disparity,
            },
        )
    final_triangulation = initial
    if accepted_views != all_views:
        final_triangulation = triangulate_dlt(
            accepted_observations,
            cameras,
            cfg,
            keypoint_id=keypoint_id,
            stage="triangulation",
        )
    if residual_counter is not None:
        try:
            residual_counter.charge(
                len(accepted_observations), stage="triangulation"
            )
        except BudgetExceeded as exc:
            exc.partial.update(
                {
                    "initial_point": initial["point"].tolist(),
                    "initial_singular_values": initial["singular_values"],
                    "participating_views": all_views,
                    "disparity": disparity,
                    "accepted_observations": [
                        obs.to_dict() for obs in accepted_observations
                    ],
                    "point": final_triangulation["point"].tolist(),
                    "triangulation_views": accepted_views,
                }
            )
            raise
    residuals = compute_residuals(
        final_triangulation["point"], accepted_observations, cameras
    )
    return {
        "status": "ok",
        "initial_point": initial["point"].tolist(),
        "initial_singular_values": initial["singular_values"],
        "participating_views": all_views,
        "disparity": disparity,
        "accepted_observations": [obs.to_dict() for obs in accepted_observations],
        "point": final_triangulation["point"].tolist(),
        "triangulation_views": accepted_views,
        "singular_values": final_triangulation["singular_values"],
        "numerical_rank": final_triangulation["numerical_rank"],
        "residuals": residuals,
        "residual_evaluations": (
            residual_counter.used if residual_counter is not None else len(residuals)
        ),
        "obs_fingerprint_after_disparity": observations_fingerprint(
            accepted_observations
        ),
    }
