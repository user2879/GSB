from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, WorkbenchError
from .util import dec, read_json, sha256_json


@dataclass(frozen=True)
class WorkbenchConfig:
    schema_version: int
    coordinate_tolerance: Decimal
    style_quantization_levels: int
    fixed_fraction_bits: int
    rounding: str
    repair_neighborhood_radius: int
    feather_radius_max: Decimal
    recompute_region_budget: int
    thread_pool_workers: int
    snapshot_commit_interval_commands: int
    pixel_difference_tolerance: int = 0
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.style_quantization_levels < 2:
            raise WorkbenchError(E_CONFIG_INVALID, "style_quantization_levels must be >= 2", stage="config")
        if not 4 <= self.fixed_fraction_bits <= 30:
            raise WorkbenchError(E_CONFIG_INVALID, "fixed_fraction_bits must be in [4, 30]", stage="config")
        if self.rounding != "half_up":
            raise WorkbenchError(E_CONFIG_INVALID, "only rounding=half_up is supported", stage="config")
        if self.repair_neighborhood_radius < 1:
            raise WorkbenchError(E_CONFIG_INVALID, "repair_neighborhood_radius must be >= 1", stage="config")
        if self.feather_radius_max < 0:
            raise WorkbenchError(E_CONFIG_INVALID, "feather_radius_max must be >= 0", stage="config")
        if self.recompute_region_budget <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "recompute_region_budget must be > 0", stage="config")
        if not 1 <= self.thread_pool_workers <= 64:
            raise WorkbenchError(E_CONFIG_INVALID, "thread_pool_workers must be in [1, 64]", stage="config")
        if self.snapshot_commit_interval_commands < 1:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "snapshot_commit_interval_commands must be >= 1",
                stage="config",
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "coordinate_tolerance": float(self.coordinate_tolerance),
            "style_quantization_levels": self.style_quantization_levels,
            "fixed_fraction_bits": self.fixed_fraction_bits,
            "rounding": self.rounding,
            "repair_neighborhood_radius": self.repair_neighborhood_radius,
            "feather_radius_max": float(self.feather_radius_max),
            "recompute_region_budget": self.recompute_region_budget,
            "thread_pool_workers": self.thread_pool_workers,
            "snapshot_commit_interval_commands": self.snapshot_commit_interval_commands,
            "pixel_difference_tolerance": self.pixel_difference_tolerance,
        }
        if self.image_size is not None:
            result["image_size"] = list(self.image_size)
        return result

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def render_config_hash(self) -> str:
        """Hash of only the fields that can affect rendered bytes and stage summaries."""
        return sha256_json(
            {
                "style_quantization_levels": self.style_quantization_levels,
                "fixed_fraction_bits": self.fixed_fraction_bits,
                "rounding": self.rounding,
                "repair_neighborhood_radius": self.repair_neighborhood_radius,
                "image_size": list(self.image_size) if self.image_size else None,
            }
        )

    def updated(self, **changes: Any) -> "WorkbenchConfig":
        normalized = dict(changes)
        for key in ("coordinate_tolerance", "feather_radius_max"):
            if key in normalized:
                normalized[key] = dec(normalized[key])
        if "image_size" in normalized and normalized["image_size"] is not None:
            normalized["image_size"] = tuple(int(v) for v in normalized["image_size"])
        return replace(self, **normalized)


def config_from_dict(data: dict[str, Any]) -> WorkbenchConfig:
    if not isinstance(data, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "config root must be an object", stage="config")
    required = [
        "schema_version",
        "coordinate_tolerance",
        "style_quantization_levels",
        "fixed_fraction_bits",
        "rounding",
        "repair_neighborhood_radius",
        "feather_radius_max",
        "recompute_region_budget",
        "thread_pool_workers",
        "snapshot_commit_interval_commands",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise WorkbenchError(E_CONFIG_INVALID, f"missing config fields: {', '.join(missing)}", stage="config")
    image_size = data.get("image_size")
    return WorkbenchConfig(
        schema_version=int(data["schema_version"]),
        coordinate_tolerance=dec(data["coordinate_tolerance"]),
        style_quantization_levels=int(data["style_quantization_levels"]),
        fixed_fraction_bits=int(data["fixed_fraction_bits"]),
        rounding=str(data["rounding"]),
        repair_neighborhood_radius=int(data["repair_neighborhood_radius"]),
        feather_radius_max=dec(data["feather_radius_max"]),
        recompute_region_budget=int(data["recompute_region_budget"]),
        thread_pool_workers=int(data["thread_pool_workers"]),
        snapshot_commit_interval_commands=int(data["snapshot_commit_interval_commands"]),
        pixel_difference_tolerance=int(data.get("pixel_difference_tolerance", 0)),
        image_size=(int(image_size[0]), int(image_size[1])) if image_size else None,
    )


def load_config(path: Path) -> WorkbenchConfig:
    return config_from_dict(read_json(path))
