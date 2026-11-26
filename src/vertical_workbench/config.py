from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .hashing import fingerprint

ALLOWED_QUANT_GEARS = ("coarse", "normal", "fine")
ALLOWED_FRAC_BITS = (4, 8, 16)
ALLOWED_WINDOW_SIZES = (16, 32, 48, 64, 96, 128, 192, 256)


@dataclass
class WorkbenchConfig:
    schema: str = "vertical-workbench-config/v1"
    width: int = 512
    height: int = 768
    seed: int = 20261127
    sample_count: int = 6
    window_size: int = 64
    stride: int = 32
    quant_gear: str = "normal"
    fixed_frac_bits: int = 8
    ink_threshold: int = 176
    candidate_score_threshold: int = 900
    area_boundary_threshold: int = 12
    min_component_area: int = 5
    merge_gap: int = 15
    split_height: int = 58
    max_workers: int = 4
    max_active_leases: int = 48
    lease_ttl_ms: int = 600_000
    global_point_budget: int = 240
    column_point_quota: int = 12
    max_component_comparisons: int = 50_000
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WorkbenchConfig":
        raw = dict(raw or {})
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in raw.items():
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        kwargs["extra"] = extra
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(data.pop("extra", {}) or {})
        return data

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.width != 512 or self.height != 768:
            errors.append("fixed layout must be 512x768")
        for name in ("window_size", "stride"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{name} must be a positive integer")
        if isinstance(self.window_size, int) and self.window_size > 0:
            if self.width % self.window_size or self.height % self.window_size:
                errors.append("window_size must divide both layout edges")
            if self.window_size not in ALLOWED_WINDOW_SIZES:
                errors.append(
                    "window_size must be one of " + ",".join(map(str, ALLOWED_WINDOW_SIZES))
                )
        if isinstance(self.stride, int) and isinstance(self.window_size, int):
            if self.stride <= 0 or self.stride > self.window_size:
                errors.append("stride must be in [1, window_size]")
        if self.quant_gear not in ALLOWED_QUANT_GEARS:
            errors.append(f"unknown quant_gear: {self.quant_gear!r}")
        if self.fixed_frac_bits not in ALLOWED_FRAC_BITS:
            errors.append(
                "fixed_frac_bits must be one of " + ",".join(map(str, ALLOWED_FRAC_BITS))
            )
        for name in (
            "max_workers",
            "max_active_leases",
            "lease_ttl_ms",
            "global_point_budget",
            "column_point_quota",
            "max_component_comparisons",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{name} must be a positive integer")
        if isinstance(self.max_workers, int) and self.max_workers > 16:
            errors.append("max_workers must not exceed 16")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    @property
    def config_hash(self) -> str:
        return fingerprint(self.to_dict())

    @property
    def quant_step(self) -> int:
        return {"coarse": 4, "normal": 2, "fine": 1}[self.quant_gear]

    def quant_fingerprint(self) -> dict[str, Any]:
        return {
            "quant_gear": self.quant_gear,
            "fixed_frac_bits": self.fixed_frac_bits,
            "quant_step": self.quant_step,
        }

    def kernel_fingerprint(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "ink_threshold": self.ink_threshold,
            "candidate_score_threshold": self.candidate_score_threshold,
            "area_boundary_threshold": self.area_boundary_threshold,
            "min_component_area": self.min_component_area,
            "merge_gap": self.merge_gap,
            "split_height": self.split_height,
            "quant": self.quant_fingerprint(),
            "font": "builtin-stroke-dotfont-v1",
        }

    @property
    def kernel_hash(self) -> str:
        return fingerprint(self.kernel_fingerprint())
