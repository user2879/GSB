"""NURBS data model and the single evaluation pipeline used by the app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .canonical import array_to_list, canonical_json_bytes, content_hash, hash_float_array, sha256_bytes
from .errors import DomainError, ErrorCode, knot_error, schema_error


EVALUATION_CONVENTION = "tensor_product_rational_bspline_d1_cross_du_dv.v1"


@dataclass(frozen=True)
class Material:
    name: str
    base_color: tuple[float, float, float]
    ambient: float
    diffuse: float
    specular: float
    shininess: float
    opacity: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Material":
        try:
            return cls(
                name=str(value["name"]),
                base_color=tuple(float(item) for item in value["base_color"]),  # type: ignore[arg-type]
                ambient=float(value["ambient"]),
                diffuse=float(value["diffuse"]),
                specular=float(value["specular"]),
                shininess=float(value["shininess"]),
                opacity=float(value["opacity"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error("invalid material channel", error=str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_color": list(self.base_color),
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "specular": self.specular,
            "shininess": self.shininess,
            "opacity": self.opacity,
        }


@dataclass(frozen=True)
class SurfaceEvaluation:
    u: float
    v: float
    point: tuple[float, float, float]
    du: tuple[float, float, float]
    dv: tuple[float, float, float]
    normal: tuple[float, float, float]
    degree_u: int
    degree_v: int
    span_u: int
    span_v: int
    knot_interval_u: tuple[float, float]
    knot_interval_v: tuple[float, float]
    support_u: tuple[int, ...]
    support_v: tuple[int, ...]
    convention: str = EVALUATION_CONVENTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "u": self.u,
            "v": self.v,
            "point": list(self.point),
            "du": list(self.du),
            "dv": list(self.dv),
            "normal": list(self.normal),
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "span_u": self.span_u,
            "span_v": self.span_v,
            "knot_interval_u": list(self.knot_interval_u),
            "knot_interval_v": list(self.knot_interval_v),
            "support_u": list(self.support_u),
            "support_v": list(self.support_v),
            "convention": self.convention,
        }

    def content_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class EvaluationGrid:
    u_values: tuple[float, ...]
    v_values: tuple[float, ...]
    points: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    normals: np.ndarray
    span_u: np.ndarray
    span_v: np.ndarray
    support_u: tuple[tuple[int, ...], ...]
    support_v: tuple[tuple[int, ...], ...]
    convention: str = EVALUATION_CONVENTION

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.u_values), len(self.v_values))


class NURBSPatch:
    """A serializable tensor-product NURBS patch with rational weights."""

    def __init__(
        self,
        *,
        key: str,
        degree_u: int,
        degree_v: int,
        knots_u: Sequence[float],
        knots_v: Sequence[float],
        control_points: np.ndarray,
        weights: np.ndarray,
        material: Material,
    ) -> None:
        self.key = str(key)
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)
        self.knots_u = np.asarray(knots_u, dtype="<f8")
        self.knots_v = np.asarray(knots_v, dtype="<f8")
        self.control_points = np.ascontiguousarray(control_points, dtype="<f8")
        self.weights = np.ascontiguousarray(weights, dtype="<f8")
        self.material = material
        self._validate()

    @property
    def control_shape(self) -> tuple[int, int]:
        return (int(self.control_points.shape[0]), int(self.control_points.shape[1]))

    @property
    def domain_u(self) -> tuple[float, float]:
        return (float(self.knots_u[self.degree_u]), float(self.knots_u[-self.degree_u - 1]))

    @property
    def domain_v(self) -> tuple[float, float]:
        return (float(self.knots_v[self.degree_v]), float(self.knots_v[-self.degree_v - 1]))

    def _validate(self) -> None:
        if self.degree_u < 1 or self.degree_v < 1:
            raise schema_error("spline degree must be at least one", patch_key=self.key)
        if self.control_points.ndim != 3 or self.control_points.shape[2] != 3:
            raise schema_error(
                "control point grid must have shape (nu, nv, 3)",
                patch_key=self.key,
                shape=list(self.control_points.shape),
            )
        nu, nv = self.control_shape
        if self.degree_u >= nu or self.degree_v >= nv:
            raise schema_error(
                "degree/control-count mismatch",
                patch_key=self.key,
                degree_u=self.degree_u,
                control_u=nu,
                degree_v=self.degree_v,
                control_v=nv,
            )
        expected_u = nu + self.degree_u + 1
        expected_v = nv + self.degree_v + 1
        if len(self.knots_u) != expected_u or len(self.knots_v) != expected_v:
            raise schema_error(
                "knot vector length does not match degree and control count",
                patch_key=self.key,
                expected_u=expected_u,
                actual_u=len(self.knots_u),
                expected_v=expected_v,
                actual_v=len(self.knots_v),
            )
        if self.weights.shape != (nu, nv):
            raise schema_error(
                "weight grid shape mismatch",
                patch_key=self.key,
                expected=[nu, nv],
                actual=list(self.weights.shape),
            )
        for name, knots in (("u", self.knots_u), ("v", self.knots_v)):
            if not np.all(np.isfinite(knots)):
                raise knot_error("knot vector contains non-finite values", patch_key=self.key, axis=name)
            if np.any(knots[1:] < knots[:-1]):
                raise knot_error("knot vector is not nondecreasing", patch_key=self.key, axis=name)
            domain = (float(knots[0]), float(knots[-1]))
            if domain[1] <= domain[0]:
                raise knot_error("knot vector has an empty parameter domain", patch_key=self.key, axis=name)
        if not np.all(np.isfinite(self.control_points)) or not np.all(np.isfinite(self.weights)):
            raise schema_error("control points and weights must be finite", patch_key=self.key)
        if np.any(self.weights <= 0.0):
            raise schema_error("NURBS weights must be strictly positive", patch_key=self.key)

    def copy(self) -> "NURBSPatch":
        return NURBSPatch(
            key=self.key,
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            knots_u=self.knots_u.copy(),
            knots_v=self.knots_v.copy(),
            control_points=self.control_points.copy(),
            weights=self.weights.copy(),
            material=self.material,
        )

    def with_control_point(self, index_u: int, index_v: int, value: Sequence[float]) -> "NURBSPatch":
        self.validate_control_index(index_u, index_v)
        points = self.control_points.copy()
        points[index_u, index_v] = np.asarray(value, dtype="<f8")
        return NURBSPatch(
            key=self.key,
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            knots_u=self.knots_u.copy(),
            knots_v=self.knots_v.copy(),
            control_points=points,
            weights=self.weights.copy(),
            material=self.material,
        )

    def validate_control_index(self, index_u: int, index_v: int) -> None:
        nu, nv = self.control_shape
        if not (0 <= index_u < nu and 0 <= index_v < nv):
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "control point index is outside the patch",
                patch_key=self.key,
                index=[index_u, index_v],
                shape=[nu, nv],
            )

    def control_content_hash(self) -> str:
        return content_hash(
            {
                "patch_key": self.key,
                "degree_u": self.degree_u,
                "degree_v": self.degree_v,
                "knots_u": self.knots_u.tolist(),
                "knots_v": self.knots_v.tolist(),
                "control_points": self.control_points.tolist(),
                "weights": self.weights.tolist(),
                "material": self.material.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u.tolist(),
            "knots_v": self.knots_v.tolist(),
            "control_points": array_to_list(self.control_points),
            "weights": array_to_list(self.weights),
            "material": self.material.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NURBSPatch":
        try:
            return cls(
                key=str(value["key"]),
                degree_u=int(value["degree_u"]),
                degree_v=int(value["degree_v"]),
                knots_u=[float(item) for item in value["knots_u"]],
                knots_v=[float(item) for item in value["knots_v"]],
                control_points=np.asarray(value["control_points"], dtype="<f8"),
                weights=np.asarray(value["weights"], dtype="<f8"),
                material=Material.from_dict(value["material"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error("invalid NURBS patch payload", error=str(exc)) from exc

    @staticmethod
    def find_span(n_control: int, degree: int, u: float, knots: np.ndarray) -> int:
        n = n_control - 1
        if u >= float(knots[n + 1]):
            return n
        if u <= float(knots[degree]):
            return degree
        low = degree
        high = n + 1
        while high - low > 1:
            mid = (low + high) // 2
            if u < float(knots[mid]):
                high = mid
            else:
                low = mid
        return low

    @staticmethod
    def _basis_values(span: int, degree: int, u: float, knots: np.ndarray) -> np.ndarray:
        basis = np.zeros(degree + 1, dtype="<f8")
        basis[0] = 1.0
        left = np.zeros(degree + 1, dtype="<f8")
        right = np.zeros(degree + 1, dtype="<f8")
        for j in range(1, degree + 1):
            left[j] = u - float(knots[span + 1 - j])
            right[j] = float(knots[span + j]) - u
            saved = 0.0
            for r in range(j):
                denominator = float(right[r + 1] + left[j - r])
                temp = 0.0 if denominator == 0.0 else basis[r] / denominator
                basis[r] = saved + float(right[r + 1]) * temp
                saved = float(left[j - r]) * temp
            basis[j] = saved
        return basis

    @classmethod
    def _basis_and_derivative(
        cls,
        span: int,
        degree: int,
        u: float,
        knots: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        basis = cls._basis_values(span, degree, u, knots)
        derivative = np.zeros(degree + 1, dtype="<f8")
        domain_start = float(knots[degree])
        domain_end = float(knots[-degree - 1])
        one_sided = False
        derivative_u = u
        if u == domain_start:
            step = max(1e-9, (domain_end - domain_start) * 1e-7)
            derivative_u = min(domain_start + step, domain_end)
            one_sided = derivative_u != u
        elif u == domain_end:
            step = max(1e-9, (domain_end - domain_start) * 1e-7)
            derivative_u = max(domain_end - step, domain_start)
            one_sided = derivative_u != u
        if derivative_u != u:
            span = cls.find_span(len(knots) - degree - 1, degree, derivative_u, knots)
            local_basis = cls._basis_values(span, degree, derivative_u, knots)
            base = span - degree
            for r in range(degree + 1):
                left_denominator = float(knots[base + r + degree] - knots[base + r])
                right_denominator = float(knots[base + r + degree + 1] - knots[base + r + 1])
                left_term = 0.0 if left_denominator == 0.0 else degree * float(local_basis[r]) / left_denominator
                right_term = (
                    0.0
                    if r == degree or right_denominator == 0.0
                    else degree * float(local_basis[r + 1]) / right_denominator
                )
                derivative[r] = left_term - right_term
        else:
            base = span - degree
            for r in range(degree + 1):
                left_denominator = float(knots[base + r + degree] - knots[base + r])
                right_denominator = float(knots[base + r + degree + 1] - knots[base + r + 1])
                left_term = 0.0 if left_denominator == 0.0 else degree * float(basis[r]) / left_denominator
                right_term = (
                    0.0
                    if r == degree or right_denominator == 0.0
                    else degree * float(basis[r + 1]) / right_denominator
                )
                derivative[r] = left_term - right_term
        return basis, derivative, one_sided

    def evaluate(self, u: float, v: float) -> SurfaceEvaluation:
        u = float(u)
        v = float(v)
        du0, du1 = self.domain_u
        dv0, dv1 = self.domain_v
        if not (du0 <= u <= du1 and dv0 <= v <= dv1):
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "surface parameter is outside the knot domain",
                patch_key=self.key,
                parameter=[u, v],
                domain=[[du0, du1], [dv0, dv1]],
            )
        nu, nv = self.control_shape
        span_u = self.find_span(nu, self.degree_u, u, self.knots_u)
        span_v = self.find_span(nv, self.degree_v, v, self.knots_v)
        basis_u, derivative_u, one_sided_u = self._basis_and_derivative(
            span_u, self.degree_u, u, self.knots_u
        )
        basis_v, derivative_v, one_sided_v = self._basis_and_derivative(
            span_v, self.degree_v, v, self.knots_v
        )
        base_u = span_u - self.degree_u
        base_v = span_v - self.degree_v
        patch_points = self.control_points[
            base_u : base_u + self.degree_u + 1,
            base_v : base_v + self.degree_v + 1,
            :,
        ]
        patch_weights = self.weights[
            base_u : base_u + self.degree_u + 1,
            base_v : base_v + self.degree_v + 1,
        ]
        weighted = patch_points * patch_weights[:, :, None]
        weights_2d = basis_u[:, None] * basis_v[None, :]
        numerator = np.sum(weighted * weights_2d[:, :, None], axis=(0, 1))
        denominator = float(np.sum(patch_weights * weights_2d))

        derivative_u_2d = derivative_u[:, None] * basis_v[None, :]
        numerator_du = np.sum(weighted * derivative_u_2d[:, :, None], axis=(0, 1))
        denominator_du = float(np.sum(patch_weights * derivative_u_2d))

        derivative_v_2d = basis_u[:, None] * derivative_v[None, :]
        numerator_dv = np.sum(weighted * derivative_v_2d[:, :, None], axis=(0, 1))
        denominator_dv = float(np.sum(patch_weights * derivative_v_2d))

        if denominator == 0.0:
            raise DomainError(ErrorCode.GEOM_STALE, "zero rational denominator", patch_key=self.key)
        point = numerator / denominator
        du = (numerator_du * denominator - numerator * denominator_du) / (denominator * denominator)
        dv = (numerator_dv * denominator - numerator * denominator_dv) / (denominator * denominator)
        normal = np.cross(du, dv)
        length = float(np.linalg.norm(normal))
        if length <= 1e-15:
            normal = np.array([0.0, 0.0, 1.0], dtype="<f8")
        else:
            normal = normal / length

        support_u = tuple(int(base_u + idx) for idx in np.flatnonzero(basis_u != 0.0))
        support_v = tuple(int(base_v + idx) for idx in np.flatnonzero(basis_v != 0.0))
        normal_support_u = tuple(
            int(base_u + idx) for idx in np.flatnonzero((basis_u != 0.0) | (derivative_u != 0.0))
        )
        normal_support_v = tuple(
            int(base_v + idx) for idx in np.flatnonzero((basis_v != 0.0) | (derivative_v != 0.0))
        )
        convention = EVALUATION_CONVENTION
        if one_sided_u or one_sided_v:
            convention = EVALUATION_CONVENTION + ";one_sided_derivative_at_domain_boundary"
        return SurfaceEvaluation(
            u=u,
            v=v,
            point=tuple(float(item) for item in point),  # type: ignore[arg-type]
            du=tuple(float(item) for item in du),  # type: ignore[arg-type]
            dv=tuple(float(item) for item in dv),  # type: ignore[arg-type]
            normal=tuple(float(item) for item in normal),  # type: ignore[arg-type]
            degree_u=self.degree_u,
            degree_v=self.degree_v,
            span_u=span_u,
            span_v=span_v,
            knot_interval_u=(float(self.knots_u[span_u]), float(self.knots_u[span_u + 1])),
            knot_interval_v=(float(self.knots_v[span_v]), float(self.knots_v[span_v + 1])),
            support_u=normal_support_u,
            support_v=normal_support_v,
            convention=convention,
        )

    def evaluate_grid(self, u_values: Iterable[float], v_values: Iterable[float]) -> EvaluationGrid:
        us = tuple(float(value) for value in u_values)
        vs = tuple(float(value) for value in v_values)
        if not us or not vs:
            raise DomainError(ErrorCode.GEOM_STALE, "evaluation grid must not be empty", patch_key=self.key)
        points = np.empty((len(us), len(vs), 3), dtype="<f8")
        du = np.empty_like(points)
        dv = np.empty_like(points)
        normals = np.empty_like(points)
        span_u = np.empty((len(us), len(vs)), dtype=np.int64)
        span_v = np.empty((len(us), len(vs)), dtype=np.int64)
        support_u: list[tuple[int, ...]] = []
        support_v: list[tuple[int, ...]] = []
        for i, u in enumerate(us):
            row_support_u: list[tuple[int, ...]] = []
            row_support_v: list[tuple[int, ...]] = []
            for j, v in enumerate(vs):
                sample = self.evaluate(u, v)
                points[i, j] = sample.point
                du[i, j] = sample.du
                dv[i, j] = sample.dv
                normals[i, j] = sample.normal
                span_u[i, j] = sample.span_u
                span_v[i, j] = sample.span_v
                row_support_u.append(sample.support_u)
                row_support_v.append(sample.support_v)
            support_u.append(tuple(row_support_u))
            support_v.append(tuple(row_support_v))
        return EvaluationGrid(
            u_values=us,
            v_values=vs,
            points=points,
            du=du,
            dv=dv,
            normals=normals,
            span_u=span_u,
            span_v=span_v,
            support_u=tuple(support_u),
            support_v=tuple(support_v),
        )

    def geometry_hash(self, grid: EvaluationGrid) -> str:
        return sha256_bytes(
            (
                content_hash({"patch_key": self.key, "control_hash": self.control_content_hash()})
                + hash_float_array("surface_points", grid.points)
                + hash_float_array("surface_du", grid.du)
                + hash_float_array("surface_dv", grid.dv)
            ).encode("ascii")
        )
