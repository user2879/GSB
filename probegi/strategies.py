"""Neighbourhood-probe interpolation strategies and the deterministic
strategy-selection policy.

Strategies
----------
``full``              Trilinear interpolation of the SH coefficient field
                      over the 8 surrounding probes, then irradiance.
``nearest_gradient``  Nearest probe coefficients plus a first-order gradient
                      correction, the gradient coming from central
                      differences of the coefficient field.
``cluster``           Probes are reduced to block representatives (block mean
                      of coefficients, deterministic); queries trilinearly
                      interpolate the coarse representative field.

Cost / error model (declared constants, part of the reproducibility
contract): with grid spacing ``h`` and mean field energy ``s`` the estimated
error bounds are ``C * h^2 * s`` with ``C`` per strategy; cluster uses the
coarse spacing ``block * h``.  Estimated ordering full < nearest_gradient <
cluster is checked against measured errors by diagnostics (E_STRATEGY_DIVERGE
on contradiction).
"""
import numpy as np

from . import sh
from .errors import BudgetDeniedError

STRATEGIES = ("full", "nearest_gradient", "cluster")

# Declared model constants.
C_TRILINEAR = 0.5
C_NN_GRADIENT = 1.5
CLUSTER_BLOCK = 2
OPS_PER_MS = 2.0e5          # declared throughput for time estimates
COEFF_BYTES = 4             # resident coefficients are stored as float32

# Order factor for error bounds: lower order -> larger bound.
ORDER_ERROR_FACTOR = {1: 4.0, 2: 1.5, 3: 1.0}


class ProbeField:
    """Resident SH coefficient field on a regular grid."""

    def __init__(self, grid_n: int, bbox, coeffs: np.ndarray):
        self.grid_n = int(grid_n)
        self.bbox = (np.asarray(bbox[0], dtype=np.float64),
                     np.asarray(bbox[1], dtype=np.float64))
        self.coeffs = np.asarray(coeffs, dtype=np.float64)  # (P, K, 3)
        self.order = int(np.sqrt(self.coeffs.shape[1])) - 1
        self.spacing = (self.bbox[1] - self.bbox[0]) / (self.grid_n - 1)
        self._grid = None  # lazy (n, n, n, K, 3) view

    @property
    def grid(self):
        if self._grid is None:
            n = self.grid_n
            self._grid = self.coeffs.reshape(n, n, n, *self.coeffs.shape[1:])
        return self._grid

    @property
    def field_scale(self) -> float:
        """Mean energy of the DC term (drives the error model)."""
        c00 = self.coeffs[:, 0, :]
        return float(np.linalg.norm(c00, axis=1).mean())

    @property
    def memory_bytes(self) -> int:
        return int(np.prod(self.coeffs.shape)) * COEFF_BYTES

    def cell_coordinates(self, points: np.ndarray):
        """Normalised grid coordinates clamped into [0, n-1]."""
        t = (points - self.bbox[0]) / self.spacing
        return np.clip(t, 0.0, self.grid_n - 1.0 - 1e-12)


def _trilinear_gather(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Trilinear sample of a (n,n,n,...) grid at float coords (Q, 3)."""
    n = grid.shape[0]
    i0 = np.floor(coords).astype(np.int64)
    f = coords - i0
    i1 = np.minimum(i0 + 1, n - 1)
    out = np.zeros((coords.shape[0],) + grid.shape[3:], dtype=np.float64)
    for c in range(8):
        a, b, cc = (c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1
        w = ((1 - f[:, 0]) * (1 - a) + f[:, 0] * a)
        w *= ((1 - f[:, 1]) * (1 - b) + f[:, 1] * b)
        w *= ((1 - f[:, 2]) * (1 - cc) + f[:, 2] * cc)
        out += w.reshape((-1,) + (1,) * (grid.ndim - 3)) * \
            grid[i1[:, 0] if a else i0[:, 0],
                 i1[:, 1] if b else i0[:, 1],
                 i1[:, 2] if cc else i0[:, 2]]
    return out


def interpolate_coeffs(field: ProbeField, strategy: str,
                       points: np.ndarray) -> np.ndarray:
    """Interpolated SH coefficients at points (Q, 3) -> (Q, K, 3)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if strategy == "full":
        return _trilinear_gather(field.grid, field.cell_coordinates(points))
    if strategy == "nearest_gradient":
        coords = field.cell_coordinates(points)
        idx = np.rint(coords).astype(np.int64)
        base = field.grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        delta = points - (field.bbox[0] + idx * field.spacing)
        grad = np.gradient(field.grid, *field.spacing, axis=(0, 1, 2),
                           edge_order=1)  # 3 x (n,n,n,K,3)
        corr = np.zeros_like(base)
        for axis in range(3):
            g = grad[axis][idx[:, 0], idx[:, 1], idx[:, 2]]
            corr += g * delta[:, axis][:, None, None]
        return base + corr
    if strategy == "cluster":
        n = field.grid_n
        b = CLUSTER_BLOCK
        coarse = field.grid.reshape(n // b, b, n // b, b, n // b, b,
                                    *field.coeffs.shape[1:]).mean(
                                        axis=(1, 3, 5))
        coarse_field = ProbeField(n // b, (field.bbox[0], field.bbox[1]),
                                  coarse.reshape(-1, *field.coeffs.shape[1:]))
        return _trilinear_gather(coarse_field.grid,
                                 coarse_field.cell_coordinates(points))
    raise ValueError(f"unknown strategy {strategy!r}")


def evaluate(field: ProbeField, strategy: str, points: np.ndarray,
             normals: np.ndarray) -> np.ndarray:
    """Reconstructed irradiance (Q, 3) at points with surface normals."""
    coeffs = interpolate_coeffs(field, strategy, points)
    return sh.irradiance(coeffs, np.asarray(normals, dtype=np.float64)
                         .reshape(-1, 3))


# ---------------------------------------------------------------------------
# Cost / error model

def est_memory_bytes(strategy: str, grid_n: int, order: int) -> int:
    probes = grid_n ** 3 if strategy != "cluster" \
        else (grid_n // CLUSTER_BLOCK) ** 3
    return probes * sh.num_coeffs(order) * 3 * COEFF_BYTES


def est_ops_per_query(strategy: str, order: int) -> float:
    k3 = sh.num_coeffs(order) * 3
    return {"full": 8.0 * k3, "nearest_gradient": 4.0 * k3,
            "cluster": 8.0 * k3}[strategy]


def est_time_ms(strategy: str, grid_n: int, order: int, n_queries: int) -> float:
    return est_ops_per_query(strategy, order) * n_queries / OPS_PER_MS


def est_error_bound(strategy: str, grid_n: int, order: int,
                    field_scale: float, bbox_extent: float = 10.0) -> float:
    h = bbox_extent / (grid_n - 1)
    if strategy == "full":
        c, hh = C_TRILINEAR, h
    elif strategy == "nearest_gradient":
        c, hh = C_NN_GRADIENT, h
    elif strategy == "cluster":
        c, hh = C_TRILINEAR, CLUSTER_BLOCK * h
    else:
        raise ValueError(strategy)
    return c * hh * hh * field_scale * ORDER_ERROR_FACTOR[order]


# ---------------------------------------------------------------------------
# Deterministic selection

def select_strategy(grid_n: int, order: int, n_queries: int,
                    field_scale: float, memory_budget_bytes: int,
                    time_budget_ms: float,
                    accuracy_budget: float) -> dict:
    """Pick a strategy under declared budgets.  Pure function of its inputs,
    so the same inputs always produce the same serialisable manifest."""
    candidates = []
    for name in STRATEGIES:
        mem = est_memory_bytes(name, grid_n, order)
        tms = est_time_ms(name, grid_n, order, n_queries)
        err = est_error_bound(name, grid_n, order, field_scale)
        candidates.append({
            "strategy": name,
            "est_memory_bytes": mem,
            "est_time_ms": tms,
            "est_error_bound": err,
            "feasible": mem <= memory_budget_bytes and tms <= time_budget_ms,
        })
    feasible = [c for c in candidates if c["feasible"]]
    if not feasible:
        raise BudgetDeniedError(
            "no interpolation strategy fits the declared budgets",
            grid_n=grid_n, order=order,
            memory_budget_bytes=memory_budget_bytes,
            time_budget_ms=time_budget_ms)
    # Minimise estimated error; deterministic tie-break by strategy order.
    best = min(feasible,
               key=lambda c: (c["est_error_bound"], STRATEGIES.index(c["strategy"])))
    return {
        "strategy": best["strategy"],
        "reason": "min est_error_bound among budget-feasible strategies",
        "accuracy_budget": accuracy_budget,
        "accuracy_budget_exceeded": bool(best["est_error_bound"] > accuracy_budget),
        "est_error_bound": best["est_error_bound"],
        "est_memory_bytes": best["est_memory_bytes"],
        "est_time_ms": best["est_time_ms"],
        "budgets": {"memory_bytes": memory_budget_bytes,
                    "time_ms": time_budget_ms,
                    "n_queries_planned": n_queries},
        "candidates": candidates,
    }
