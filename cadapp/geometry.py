"""NURBS surface evaluation and tessellation.

All display artefacts (shaded mesh vertices, per-vertex normals and
isoparametric curve samples) are produced by the single evaluation
pipeline in this module, so a shaded-grid vertex and an isocurve sample
at the same (u, v) parameter agree to within GEOMETRY_TOLERANCE (in
practice they are bit-identical because both call evaluate_surface_point).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .errors import CadError, ErrorCode

TESSELLATION_RESOLUTION = 24  # quads per parametric direction
ISO_CURVE_COUNT = 8           # isocurves per parametric direction
ISO_CURVE_SAMPLES = 65        # sample points per isocurve


def validate_knot_vector(knots, degree: int, ctrl_count: int, location: str) -> None:
    """Knot vector must be non-decreasing and have ctrl_count+degree+1 entries."""
    for i in range(1, len(knots)):
        if knots[i] < knots[i - 1]:
            raise CadError(
                ErrorCode.E_KNOT_MONOTONIC,
                f"knot vector is not monotonically non-decreasing: "
                f"knots[{i-1}]={knots[i-1]!r} > knots[{i}]={knots[i]!r}",
                location=f"{location}[{i}]",
            )
    expected = ctrl_count + degree + 1
    if len(knots) != expected:
        raise CadError(
            ErrorCode.E_FORMAT_HEADER,
            f"knot count {len(knots)} != control count {ctrl_count} + degree {degree} + 1",
            location=location,
        )


def parameter_domain(knots, degree: int):
    return knots[degree], knots[len(knots) - degree - 1]


def _find_span(ctrl_count: int, degree: int, u: float, knots) -> int:
    """Piegl & Tiller A2.1.  u is clamped to the parameter domain."""
    n = ctrl_count - 1
    lo = knots[degree]
    hi = knots[n + 1]
    if u <= lo:
        return degree
    if u >= hi:
        return n
    low = degree
    high = n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _basis_funs(span: int, u: float, degree: int, knots):
    """Piegl & Tiller A2.2: non-zero basis values at u."""
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    values = [0.0] * (degree + 1)
    values[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            temp = values[r] / (right[r + 1] + left[j - r])
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved
    return values


def _basis_ders(span: int, u: float, degree: int, knots):
    """Piegl & Tiller A2.3: basis values and first derivatives."""
    ndu = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    ndu[0][0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j][j] = saved
    ders0 = [ndu[j][degree] for j in range(degree + 1)]
    ders1 = [0.0] * (degree + 1)
    for r in range(degree + 1):
        term = 0.0
        if r >= 1:
            denom = ndu[degree][r - 1]
            if denom != 0.0:
                term += degree * ndu[r - 1][degree - 1] / denom
        if r <= degree - 1:
            denom = ndu[degree][r]
            if denom != 0.0:
                term -= degree * ndu[r][degree - 1] / denom
        ders1[r] = term
    return ders0, ders1


def evaluate_surface_point(patch, u: float, v: float):
    """Rational B-spline surface point S(u, v) as a length-3 numpy array."""
    point, _, _ = evaluate_surface_point_and_derivatives(patch, u, v)
    return point


def evaluate_surface_point_and_derivatives(patch, u: float, v: float):
    """Return (S, dS/du, dS/dv) for the patch at (u, v)."""
    cp = patch.control_points  # list[list[[x,y,z]]]
    w = patch.weights
    nu = len(cp)
    nv = len(cp[0])
    pu = patch.degree_u
    pv = patch.degree_v
    span_u = _find_span(nu, pu, u, patch.knots_u)
    span_v = _find_span(nv, pv, v, patch.knots_v)
    basis_u, der_u = _basis_ders(span_u, u, pu, patch.knots_u)
    basis_v, der_v = _basis_ders(span_v, v, pv, patch.knots_v)

    a = np.zeros(3)
    a_u = np.zeros(3)
    a_v = np.zeros(3)
    w_sum = 0.0
    w_sum_u = 0.0
    w_sum_v = 0.0
    for i in range(pu + 1):
        ci = span_u - pu + i
        for j in range(pv + 1):
            cj = span_v - pv + j
            weight = w[ci][cj]
            p = np.asarray(cp[ci][cj], dtype=np.float64)
            bu = basis_u[i]
            bv = basis_v[j]
            bw = bu * bv * weight
            a += bw * p
            w_sum += bw
            bdu = der_u[i] * bv * weight
            a_u += bdu * p
            w_sum_u += bdu
            bdv = bu * der_v[j] * weight
            a_v += bdv * p
            w_sum_v += bdv
    s = a / w_sum
    ds_du = (a_u - s * w_sum_u) / w_sum
    ds_dv = (a_v - s * w_sum_v) / w_sum
    return s, ds_du, ds_dv


def evaluate_surface_normal(patch, u: float, v: float):
    """Unit normal from the analytic partial derivatives."""
    _, ds_du, ds_dv = evaluate_surface_point_and_derivatives(patch, u, v)
    normal = np.cross(ds_du, ds_dv)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


@dataclass
class PatchGeometry:
    """Tessellated display geometry for one surface patch."""

    vertices: np.ndarray          # (N, 3) float64, row-major over (res+1)^2 grid
    normals: np.ndarray           # (N, 3) float64 unit normals
    faces: np.ndarray             # (M, 3) int64 triangle indices
    params: np.ndarray            # (N, 2) float64 (u, v) per vertex
    isocurves: list = field(default_factory=list)  # list of (ISO_CURVE_SAMPLES, 3) arrays
    control_lines: list = field(default_factory=list)  # list of (2, 3) segments


def tessellate_patch(patch, resolution: int = TESSELLATION_RESOLUTION) -> PatchGeometry:
    """Build the shared tessellation used by all display modes."""
    u0, u1 = parameter_domain(patch.knots_u, patch.degree_u)
    v0, v1 = parameter_domain(patch.knots_v, patch.degree_v)
    us = np.linspace(u0, u1, resolution + 1)
    vs = np.linspace(v0, v1, resolution + 1)
    count = (resolution + 1) * (resolution + 1)
    vertices = np.zeros((count, 3), dtype=np.float64)
    normals = np.zeros((count, 3), dtype=np.float64)
    params = np.zeros((count, 2), dtype=np.float64)
    k = 0
    for j in range(resolution + 1):
        for i in range(resolution + 1):
            u = float(us[i])
            v = float(vs[j])
            point, du, dv = evaluate_surface_point_and_derivatives(patch, u, v)
            normal = np.cross(du, dv)
            norm = float(np.linalg.norm(normal))
            vertices[k] = point
            normals[k] = normal / norm if norm != 0.0 else (0.0, 0.0, 1.0)
            params[k] = (u, v)
            k += 1
    faces = []
    row = resolution + 1
    for j in range(resolution):
        for i in range(resolution):
            a = j * row + i
            b = j * row + i + 1
            c = (j + 1) * row + i
            d = (j + 1) * row + i + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    isocurves = []
    for t in range(1, ISO_CURVE_COUNT + 1):
        frac = t / (ISO_CURVE_COUNT + 1)
        u_iso = u0 + frac * (u1 - u0)
        v_iso = v0 + frac * (v1 - v0)
        curve_u = np.array(
            [evaluate_surface_point(patch, u_iso, v0 + s / (ISO_CURVE_SAMPLES - 1) * (v1 - v0))
             for s in range(ISO_CURVE_SAMPLES)]
        )
        curve_v = np.array(
            [evaluate_surface_point(patch, u0 + s / (ISO_CURVE_SAMPLES - 1) * (u1 - u0), v_iso)
             for s in range(ISO_CURVE_SAMPLES)]
        )
        isocurves.append(curve_u)
        isocurves.append(curve_v)
    control_lines = []
    cp = patch.control_points
    for i in range(len(cp)):
        for j in range(len(cp[0])):
            if i + 1 < len(cp):
                control_lines.append(np.array([cp[i][j], cp[i + 1][j]], dtype=np.float64))
            if j + 1 < len(cp[0]):
                control_lines.append(np.array([cp[i][j], cp[i][j + 1]], dtype=np.float64))
    return PatchGeometry(
        vertices=vertices,
        normals=normals,
        faces=np.asarray(faces, dtype=np.int64),
        params=params,
        isocurves=isocurves,
        control_lines=control_lines,
    )
