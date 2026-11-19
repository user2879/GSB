"""Spectral interpolation and deterministic CIE XYZ integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .canonical import array_digest, stable_float
from .cmf import WAVELENGTHS, Observer, get_observer
from .config import NORMALIZATIONS, QUADRATURES
from .errors import DiagnosticError, ErrorCode


@dataclass(frozen=True)
class IntegrationConvention:
    observer: str
    density_nm: float
    quadrature: str = "trapezoid"
    normalization: str = "y100_reflector"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "observer": self.observer,
            "density_nm": self.density_nm,
            "quadrature": self.quadrature,
            "normalization": self.normalization,
        }


@dataclass(frozen=True)
class IntegrationResult:
    xyz: np.ndarray
    white_xyz: np.ndarray
    weights: np.ndarray
    normalization_scale: float
    convention: IntegrationConvention
    observer_name: str
    weight_sum_y: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "convention": self.convention.fingerprint(),
            "observer_name": self.observer_name,
            "normalization": self.convention.normalization,
            "normalization_scale": stable_float(self.normalization_scale),
            "weight_sum_y": stable_float(self.weight_sum_y),
            "spectral_weights_sha256": array_digest(self.weights),
            "xyz_sha256": array_digest(self.xyz),
            "white_xyz_sha256": array_digest(self.white_xyz),
            "white_xyz": self.white_xyz.tolist(),
        }


def dense_grid(density_nm: float, start_nm: float = 380.0, stop_nm: float = 780.0) -> np.ndarray:
    if not np.isfinite(density_nm) or density_nm <= 0:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="integration density must be a positive finite number",
            details={"density_nm": density_nm},
        )
    intervals = round((stop_nm - start_nm) / density_nm)
    if intervals < 1:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="integration density is too coarse for the spectral interval",
            details={"density_nm": density_nm, "interval_nm": [start_nm, stop_nm]},
        )
    actual = (stop_nm - start_nm) / intervals
    grid = start_nm + np.arange(intervals + 1, dtype=float) * actual
    if not np.isclose(grid[-1], stop_nm, rtol=0.0, atol=1e-10):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="integration grid does not close on the spectral interval",
            details={"density_nm": density_nm, "last_wavelength_nm": float(grid[-1])},
        )
    return grid


def quadrature_weights(wavelengths: np.ndarray, quadrature: str = "trapezoid") -> np.ndarray:
    if quadrature not in QUADRATURES:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="unsupported integration quadrature",
            details={"quadrature": quadrature, "allowed": list(QUADRATURES)},
        )
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="quadrature requires at least two wavelengths",
            details={"shape": list(wavelengths.shape)},
        )
    deltas = np.diff(wavelengths)
    if np.any(deltas <= 0):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="wavelength grid must be strictly increasing",
            details={"wavelengths": wavelengths.tolist()},
        )
    weights = np.empty_like(wavelengths)
    weights[0] = deltas[0] / 2.0
    weights[-1] = deltas[-1] / 2.0
    weights[1:-1] = (deltas[:-1] + deltas[1:]) / 2.0
    return weights


def _as_spectra(values: np.ndarray, name: str, expected_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape[-1] != expected_count:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message=f"{name} must have {expected_count} values on its declared wavelength grid",
            details={"name": name, "shape": list(result.shape), "expected_count": expected_count},
        )
    if not np.all(np.isfinite(result)):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message=f"{name} contains non-finite values",
            details={"name": name},
        )
    return result


def _interpolate_to_grid(values: np.ndarray, source_grid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if source_grid.size == grid.size and np.allclose(source_grid, grid, rtol=0.0, atol=1e-10):
        return values
    base = np.asarray(source_grid, dtype=float)
    flat = values.reshape((-1, values.shape[-1]))
    interpolated = np.vstack([np.interp(grid, base, row, left=row[0], right=row[-1]) for row in flat])
    return interpolated.reshape(values.shape[:-1] + (grid.size,))


def integrate_reflectance(
    reflectance: np.ndarray,
    spd: np.ndarray,
    convention: IntegrationConvention,
    wavelengths_nm: np.ndarray | None = None,
) -> IntegrationResult:
    """Integrate reflectance under an SPD, separating light and reflectance terms.

    The unnormalised white response is ``sum(SPD * CMF * dlambda)``.  The
    normalisation scales that response to Y=100 for a perfect reflector unless
    ``unity`` is selected.  Reflectance is applied only after the light/observer
    weights have been formed.
    """

    observer: Observer = get_observer(convention.observer)
    if convention.normalization not in NORMALIZATIONS:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="unsupported integration normalisation",
            details={
                "normalization": convention.normalization,
                "allowed": list(NORMALIZATIONS),
            },
        )
    if convention.quadrature not in QUADRATURES:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="unsupported integration quadrature",
            details={"quadrature": convention.quadrature, "allowed": list(QUADRATURES)},
        )

    if wavelengths_nm is None:
        source_grid = np.asarray(WAVELENGTHS, dtype=float)
    else:
        source_grid = np.asarray(wavelengths_nm, dtype=float)
        if source_grid.ndim != 1 or source_grid.size < 2 or np.any(np.diff(source_grid) <= 0):
            raise DiagnosticError(
                code=ErrorCode.INPUT_INVALID,
                message="source wavelength grid must be strictly increasing",
                details={"wavelengths_nm": source_grid.tolist()},
            )
    spd_array = _as_spectra(spd, "spd", source_grid.size)
    if spd_array.ndim != 1:
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="spd must be one-dimensional",
            details={"shape": list(spd_array.shape)},
        )
    if np.any(spd_array < 0):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="spd must be non-negative",
            details={"min_spd": float(np.min(spd_array))},
        )
    reflectance_array = _as_spectra(reflectance, "reflectance", source_grid.size)
    if np.any(reflectance_array < 0):
        raise DiagnosticError(
            code=ErrorCode.INPUT_INVALID,
            message="reflectance must be non-negative",
            details={"min_reflectance": float(np.min(reflectance_array))},
        )

    grid = dense_grid(convention.density_nm)
    weights_dl = quadrature_weights(grid, convention.quadrature)
    spd_dense = _interpolate_to_grid(spd_array, source_grid, grid)
    cmf_dense = np.vstack(
        [np.interp(grid, np.asarray(WAVELENGTHS), observer.cmf[:, component]) for component in range(3)]
    ).T
    reflectance_dense = _interpolate_to_grid(reflectance_array, source_grid, grid)

    # The light and observer terms are combined before reflectance is applied.
    light_observer = spd_dense[:, None] * cmf_dense * weights_dl[:, None]
    white_unnormalised = np.sum(light_observer, axis=0)
    if not np.all(np.isfinite(white_unnormalised)) or np.any(white_unnormalised <= 0):
        component = int(np.argmin(white_unnormalised))
        raise DiagnosticError(
            code=ErrorCode.WHITE_POINT,
            message="source SPD produces a non-positive white point",
            details={
                "white_xyz": white_unnormalised.tolist(),
                "component": component,
                "component_name": ("X", "Y", "Z")[component],
                "sample_index": 0,
            },
        )

    if convention.normalization == "y100_reflector":
        scale = 100.0 / float(white_unnormalised[1])
    elif convention.normalization == "unity":
        scale = 1.0
    else:  # pragma: no cover - guarded above
        raise AssertionError("unreachable normalisation branch")

    weights = light_observer * scale
    xyz = np.tensordot(reflectance_dense, weights, axes=([-1], [0]))
    white_xyz = white_unnormalised * scale
    return IntegrationResult(
        xyz=np.asarray(xyz, dtype=float),
        white_xyz=np.asarray(white_xyz, dtype=float),
        weights=np.asarray(weights, dtype=float),
        normalization_scale=float(scale),
        convention=convention,
        observer_name=observer.name,
        weight_sum_y=float(np.sum(weights[:, 1])),
    )


def integration_metadata(result: IntegrationResult) -> dict[str, Any]:
    return result.to_metadata()


def convention_dict(convention: IntegrationConvention) -> dict[str, Any]:
    return asdict(convention)
