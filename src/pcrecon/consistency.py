"""Incremental-vs-full reconstruction consistency checks."""
from __future__ import annotations

import numpy as np

from .errors import MeshMismatchError

# Declared tolerances.  Cached patches are byte-identical; full rebuilds only
# differ by floating-point summation order in global node accumulations
# (observed normal angle <= 3.4e-8 rad, position error 0.0).
POSITION_TOL = 1e-7
NORMAL_TOL = 1e-7


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return n / lengths


def compare_meshes(
    incremental: dict,
    full: dict,
    *,
    position_tol: float = POSITION_TOL,
    normal_tol: float = NORMAL_TOL,
    op_id: int | None = None,
) -> dict:
    """Compare two meshes by vertex position and face normal.

    Both meshes are produced by the same canonical tile ordering, so vertices
    and faces align 1:1; face count alone is never treated as proof.
    """
    iv, fv = incremental["vertices"], full["vertices"]
    iface, fface = incremental["faces"], full["faces"]
    report: dict = {
        "op_id": op_id,
        "position_tol": position_tol,
        "normal_tol": normal_tol,
        "vertex_count_incremental": int(iv.shape[0]),
        "vertex_count_full": int(fv.shape[0]),
        "face_count_incremental": int(iface.shape[0]),
        "face_count_full": int(fface.shape[0]),
    }
    if iv.shape != fv.shape or iface.shape != fface.shape:
        report["status"] = "mismatch"
        report["first_divergent_vertex"] = _first_shape_divergence(iv, fv)
        report["reason"] = "shape"
        raise MeshMismatchError("mesh shape divergence", context=report)

    pos_diff = np.linalg.norm(iv - fv, axis=1)
    report["position_max_abs_error"] = float(pos_diff.max(initial=0.0))
    report["position_rms_error"] = float(np.sqrt(np.mean(pos_diff**2)) if pos_diff.size else 0.0)
    report["position_error_histogram"] = _histogram(pos_diff)

    inorm = face_normals(iv, iface)
    fnorm = face_normals(fv, fface)
    normal_cos = np.clip(np.sum(inorm * fnorm, axis=1), -1.0, 1.0)
    normal_angle = np.arccos(normal_cos)
    report["normal_max_angle_rad"] = float(normal_angle.max(initial=0.0))
    report["normal_angle_histogram"] = _histogram(normal_angle)

    first_vertex = int(np.flatnonzero(pos_diff > position_tol)[0]) if np.any(pos_diff > position_tol) else -1
    first_face = (
        int(np.flatnonzero(normal_angle > normal_tol)[0]) if np.any(normal_angle > normal_tol) else -1
    )
    report["first_divergent_vertex"] = first_vertex
    report["first_divergent_face"] = first_face
    ok = (
        report["position_max_abs_error"] <= position_tol
        and report["normal_max_angle_rad"] <= normal_tol
        and np.array_equal(iface, fface)
    )
    report["status"] = "match" if ok else "mismatch"
    if not ok:
        raise MeshMismatchError(
            "incremental mesh diverges from full rebuild",
            context=report,
        )
    return report


def _first_shape_divergence(iv: np.ndarray, fv: np.ndarray) -> int:
    n = min(iv.shape[0], fv.shape[0])
    for i in range(n):
        if not np.allclose(iv[i], fv[i], atol=POSITION_TOL):
            return i
    return n


def _histogram(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"edges": [], "counts": []}
    edges = np.array([0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e0], dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "edges": [float(x) for x in edges],
        "counts": [int(x) for x in counts],
    }
