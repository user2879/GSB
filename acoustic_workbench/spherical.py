"""Real spherical-harmonic fitting on a deterministic full-sphere sampling grid."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import lpmv

from .config import (
    CHANNEL_SPECS,
    MAX_ORDER_BOUNDS_DECLARATION,
    SPHERICAL_COVERAGE_DECLARATION,
    channel_spec,
    sph_order_bound,
)
from .errors import E_SPH_ORDER, fail


def fibonacci_directions(sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        fail(E_SPH_ORDER, "spherical sample count must be positive", sample_count=sample_count)
    indices = np.arange(sample_count, dtype=np.float64) + 0.5
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * indices / sample_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * indices
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def directions_from_lon_lat(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def equirectangular_directions(
    width: int,
    height: int,
    bounds: tuple[float, float, float, float] = (-180.0, 180.0, -90.0, 90.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bounds
    lon_axis = lon_min + (np.arange(width, dtype=np.float64) + 0.5) * (lon_max - lon_min) / width
    lat_axis = lat_max - (np.arange(height, dtype=np.float64) + 0.5) * (lat_max - lat_min) / height
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    directions = directions_from_lon_lat(lon_grid, lat_grid)
    return directions.reshape(-1, 3), lon_grid, lat_grid


def forward_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def support_mask(channel_id: int, directions: np.ndarray) -> np.ndarray:
    spec = channel_spec(channel_id)
    forward = forward_direction(spec.azimuth_deg, spec.elevation_deg)
    cosine = np.clip(directions @ forward, -1.0, 1.0)
    return np.arccos(cosine) <= math.radians(spec.support_deg)


def real_sh_basis(directions: np.ndarray, order: int) -> np.ndarray:
    if order < 0:
        fail(E_SPH_ORDER, "spherical-harmonic order must be non-negative", order=order)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        fail(E_SPH_ORDER, "direction array must have shape (N, 3)")
    x = np.clip(directions[:, 2], -1.0, 1.0)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    columns = []
    for degree in range(order + 1):
        for rank in range(-degree, degree + 1):
            absolute = abs(rank)
            legendre = lpmv(absolute, degree, x)
            normalization = math.sqrt(
                (2.0 * degree + 1.0)
                / (4.0 * math.pi)
                * math.factorial(degree - absolute)
                / math.factorial(degree + absolute)
            )
            if rank > 0:
                value = math.sqrt(2.0) * normalization * legendre * np.cos(rank * phi)
            elif rank < 0:
                value = math.sqrt(2.0) * normalization * legendre * np.sin(absolute * phi)
            else:
                value = normalization * legendre
            columns.append(value)
    return np.column_stack(columns)


def _path_lobe(directions: np.ndarray, path_direction: np.ndarray, cutoff_deg: float = 82.0) -> np.ndarray:
    cosine = np.clip(directions @ path_direction, -1.0, 1.0)
    cutoff = math.cos(math.radians(cutoff_deg))
    return np.maximum(0.0, (cosine - cutoff) / (1.0 - cutoff)) ** 2


def fit_channel(
    metadata: dict[str, Any],
    sample_density: int,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    bound = sph_order_bound(sample_density)
    if order < 0 or order > 8:
        fail(E_SPH_ORDER, "spherical-harmonic order is outside [0, 8]", order=order)
    if order > bound:
        fail(E_SPH_ORDER, "spherical-harmonic order exceeds sampling-density bound", order=order, bound=bound)
    sample_directions = fibonacci_directions(sample_density)
    mask = support_mask(int(metadata["channel_id"]), sample_directions)
    energy = np.full(sample_density, 1e-7, dtype=np.float64)
    for path in metadata["paths"]:
        direction = np.asarray(path["arrival_direction"], dtype=np.float64)
        energy += float(path["energy"]) * _path_lobe(sample_directions, direction)
    energy *= mask.astype(np.float64)
    basis = real_sh_basis(sample_directions, order)
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(basis, energy, rcond=None)
    approximation = basis @ coefficients
    relative_residual = float(np.linalg.norm(approximation - energy) / max(np.linalg.norm(energy), 1e-15))
    fit_metadata = {
        "channel_id": int(metadata["channel_id"]),
        "sample_density": int(sample_density),
        "coefficient_count": int(coefficients.size),
        "truncation_order": int(order),
        "order_bound": int(bound),
        "order_bound_declaration": MAX_ORDER_BOUNDS_DECLARATION,
        "coverage": SPHERICAL_COVERAGE_DECLARATION,
        "sample_count": int(sample_density),
        "support_cap_deg": float(CHANNEL_SPECS[int(metadata["channel_id"])].support_deg),
        "relative_fit_residual": relative_residual,
        "lstsq_rank": int(rank),
        "lstsq_singular_values_sha256": _array_sha256(singular_values),
    }
    return coefficients, fit_metadata


def evaluate_channel(
    coefficients: np.ndarray,
    order: int,
    channel_id: int,
    directions: np.ndarray,
    *,
    apply_support: bool = True,
) -> np.ndarray:
    basis = real_sh_basis(directions, order)
    field = np.maximum(0.0, basis @ coefficients)
    if apply_support:
        field *= support_mask(channel_id, directions).astype(np.float64)
    return field


def coefficients_to_bytes(coefficients: np.ndarray) -> bytes:
    return np.asarray(coefficients, dtype="<f8").tobytes(order="C")


def coefficients_from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<f8").copy()


def _array_sha256(values: np.ndarray) -> str:
    from .hashing import sha256_bytes

    return sha256_bytes(np.asarray(values, dtype="<f8").tobytes(order="C"))
