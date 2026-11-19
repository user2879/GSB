"""Serializable configuration objects and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from .errors import ErrorCode, WorkbenchError


def _require(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise WorkbenchError(ErrorCode.E_INJECT_CONFIG, message, details)


@dataclass(frozen=True)
class DarkInconsistencyInjection:
    offset_dn: float = 0.0
    band_start: int = 0
    band_end: int = 64


@dataclass(frozen=True)
class DegenerateEndmemberInjection:
    source: int = 0
    target: int = 1
    blend: float = 0.99999


@dataclass(frozen=True)
class CenterShiftInjection:
    dx_px: float = 0.0
    dy_px: float = 0.0
    frame_start: int = 0
    frame_end: int = 64


@dataclass(frozen=True)
class SaturationOverloadInjection:
    frame: int = 0
    y0: int = 0
    x0: int = 0
    height: int = 16
    width: int = 16


@dataclass(frozen=True)
class ConcentrationBiasInjection:
    fluorophore: int = 0
    delta: float = 0.0
    frame: int = 0
    y0: int = 5
    x0: int = 5
    height: int = 5
    width: int = 5


@dataclass(frozen=True)
class InjectionConfig:
    """All fault paths enter through this serializable parameter object."""

    dark_inconsistency: Optional[DarkInconsistencyInjection] = None
    degenerate_endmembers: Optional[DegenerateEndmemberInjection] = None
    center_shift: Optional[CenterShiftInjection] = None
    saturation_overload: Optional[SaturationOverloadInjection] = None
    concentration_bias: Optional[ConcentrationBiasInjection] = None

    def validate(self, frames: int, fluorophores: int, height: int, width: int) -> None:
        if self.dark_inconsistency is not None:
            item = self.dark_inconsistency
            _require(item.band_start >= 0 and item.band_end <= frames, "dark injection band range invalid")
            _require(item.band_start < item.band_end, "dark injection band range empty")
            _require(abs(item.offset_dn) <= 65535.0, "dark injection offset out of range")
        if self.degenerate_endmembers is not None:
            item = self.degenerate_endmembers
            _require(0 <= item.source < fluorophores, "degenerate source index invalid")
            _require(0 <= item.target < fluorophores, "degenerate target index invalid")
            _require(item.source != item.target, "degenerate source and target must differ")
            _require(0.99 <= item.blend <= 1.0, "degenerate blend must be in [0.99, 1.0]")
        if self.center_shift is not None:
            item = self.center_shift
            _require(item.frame_start >= 0 and item.frame_end <= frames, "center-shift frame range invalid")
            _require(item.frame_start < item.frame_end, "center-shift frame range empty")
            _require(abs(item.dx_px) <= width / 4 and abs(item.dy_px) <= height / 4, "center shift too large")
        if self.saturation_overload is not None:
            item = self.saturation_overload
            _require(0 <= item.frame < frames, "saturation frame invalid")
            _require(item.y0 >= 0 and item.x0 >= 0, "saturation origin invalid")
            _require(item.height > 0 and item.width > 0, "saturation size invalid")
            _require(item.y0 + item.height <= height and item.x0 + item.width <= width, "saturation box outside frame")
        if self.concentration_bias is not None:
            item = self.concentration_bias
            _require(0 <= item.fluorophore < fluorophores, "bias fluorophore invalid")
            _require(0 <= item.frame < frames, "bias frame invalid")
            _require(item.y0 >= 0 and item.x0 >= 0, "bias origin invalid")
            _require(item.height > 0 and item.width > 0, "bias size invalid")
            _require(item.y0 + item.height <= height and item.x0 + item.width <= width, "bias box outside frame")

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "InjectionConfig":
        if not value:
            return cls()
        try:
            return cls(
                dark_inconsistency=(
                    DarkInconsistencyInjection(**value["dark_inconsistency"])
                    if value.get("dark_inconsistency")
                    else None
                ),
                degenerate_endmembers=(
                    DegenerateEndmemberInjection(**value["degenerate_endmembers"])
                    if value.get("degenerate_endmembers")
                    else None
                ),
                center_shift=CenterShiftInjection(**value["center_shift"]) if value.get("center_shift") else None,
                saturation_overload=(
                    SaturationOverloadInjection(**value["saturation_overload"])
                    if value.get("saturation_overload")
                    else None
                ),
                concentration_bias=(
                    ConcentrationBiasInjection(**value["concentration_bias"])
                    if value.get("concentration_bias")
                    else None
                ),
            )
        except (TypeError, KeyError) as exc:
            raise WorkbenchError(ErrorCode.E_INJECT_CONFIG, "invalid injection object", {"error": str(exc)}) from exc


@dataclass(frozen=True)
class CubeConfig:
    seed: int = 20261119
    frames: int = 64
    height: int = 256
    width: int = 256
    fluorophores: int = 4
    endmember_scale_dn: float = 6000.0
    background_dn: float = 12.0
    noise_sigma_dn: float = 2.0
    dark_level_dn: float = 100.0
    flat_mean_dn: float = 10000.0
    saturation_dn: int = 65535
    drift_dx_per_frame: float = 0.012
    drift_dy_per_frame: float = -0.008
    blob_sigma_px: float = 5.5

    def validate(self) -> None:
        _require(self.frames > 0 and self.height > 0 and self.width > 0, "cube dimensions must be positive")
        _require(self.fluorophores > 0, "fluorophore count must be positive")
        _require(self.frames >= self.fluorophores, "frames must cover fluorophore spectra")
        _require(self.saturation_dn == 65535, "uint16 saturation level must be 65535")
        _require(self.noise_sigma_dn >= 0.0, "noise sigma must be non-negative")
        _require(self.blob_sigma_px > 0.0, "blob sigma must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CubeConfig":
        try:
            cfg = cls(**value)
        except TypeError as exc:
            raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "invalid cube configuration", {"error": str(exc)}) from exc
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class CalibrationConfig:
    flat_epsilon_dn: float = 1.0e-6
    max_dark_dn: float = 1000.0
    max_zero_flat_fraction: float = 0.01
    max_saturation_fraction: float = 0.05
    invalid_value: float = 0.0

    def hash_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnmixConfig:
    order: Optional[int] = None
    cond_warn: float = 30.0
    cond_fatal: float = 1.0e6
    endmember_file: Optional[str] = None
    tolerance: float = 0.0

    def hash_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftConfig:
    compensate: bool = True
    max_shift_px: float = 1.5
    window_radius_px: int = 22
    background_percentile: float = 20.0
    min_center_signal_dn: float = 20.0

    def hash_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    unmix: UnmixConfig = field(default_factory=UnmixConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    concentration_tolerance: float = 0.08
    closure_rtol: float = 1.0e-6
    closure_atol: float = 1.0e-7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "PipelineConfig":
        if not value:
            return cls()
        try:
            return cls(
                calibration=CalibrationConfig(**value.get("calibration", {})),
                unmix=UnmixConfig(**value.get("unmix", {})),
                drift=DriftConfig(**value.get("drift", {})),
                concentration_tolerance=float(value.get("concentration_tolerance", 0.08)),
                closure_rtol=float(value.get("closure_rtol", 1.0e-6)),
                closure_atol=float(value.get("closure_atol", 1.0e-7)),
            )
        except (TypeError, ValueError) as exc:
            raise WorkbenchError(ErrorCode.E_INJECT_CONFIG, "invalid pipeline configuration", {"error": str(exc)}) from exc


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_drop_none(item) for item in value]
    return value


def resolve_centers(height: int, width: int) -> Tuple[Tuple[float, float], ...]:
    """Return well-separated nominal blob centers in (x, y) order."""
    return (
        (0.25 * width, 0.25 * height),
        (0.75 * width, 0.25 * height),
        (0.25 * width, 0.75 * height),
        (0.75 * width, 0.75 * height),
    )
