"""Representation configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import codes
from .models import Failure
from .util import ensure_finite_number, hash_value


@dataclass
class SceneConfig:
    image_size: tuple[int, int]
    base_scale: float = 1.2
    levels: int = 3
    scale_decay: Any = 0.6
    level_density: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    sh_degree: int = 1
    frac_bits: int = 12
    quant_gear: str = "NEAREST"
    render_samples: int = 4
    kernel_budgets: list[int] = field(default_factory=lambda: [4096, 4096, 4096])
    point_budget: int = 20000
    concurrency: int = 4
    snapshot_interval: int = 2
    tolerance: dict[str, float] = field(
        default_factory=lambda: {"kernel": 0.001, "render": 0.02, "composite": 0.02}
    )
    tile_size: int = 16
    level_scale_overrides: list[float | None] = field(default_factory=list)
    seed: int = 20240517
    pixel_mode: str = "RGB"

    def __post_init__(self) -> None:
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.level_density = [float(item) for item in self.level_density]
        self.kernel_budgets = [int(item) for item in self.kernel_budgets]
        if not self.level_scale_overrides:
            self.level_scale_overrides = [None] * self.levels
        else:
            self.level_scale_overrides = [
                None if item is None else float(item) for item in self.level_scale_overrides
            ]

    def decays(self) -> list[float]:
        if isinstance(self.scale_decay, (list, tuple)):
            values = [float(item) for item in self.scale_decay]
            if len(values) == 1:
                return values * self.levels
            return values
        return [float(self.scale_decay)] * self.levels

    def effective_scales(self) -> list[float]:
        values: list[float] = []
        running = float(self.base_scale)
        for level in range(self.levels):
            if level > 0:
                running *= float(self.decays()[level - 1])
            override = self.level_scale_overrides[level]
            values.append(float(running if override is None else override))
        return values

    def scale_ratios(self) -> list[float]:
        return [value / float(self.base_scale) for value in self.effective_scales()]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_scale": float(self.base_scale),
            "concurrency": int(self.concurrency),
            "frac_bits": int(self.frac_bits),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "kernel_budgets": [int(item) for item in self.kernel_budgets],
            "level_density": [float(item) for item in self.level_density],
            "level_scale_overrides": list(self.level_scale_overrides),
            "levels": int(self.levels),
            "pixel_mode": self.pixel_mode,
            "point_budget": int(self.point_budget),
            "quant_gear": self.quant_gear,
            "render_samples": int(self.render_samples),
            "scale_decay": self.scale_decay,
            "seed": int(self.seed),
            "sh_degree": int(self.sh_degree),
            "snapshot_interval": int(self.snapshot_interval),
            "tile_size": int(self.tile_size),
            "tolerance": {key: float(value) for key, value in sorted(self.tolerance.items())},
        }

    def fingerprint(self) -> str:
        return hash_value(self.as_dict())

    def content_fingerprint(self) -> str:
        """Content-defining configuration; scheduling/tolerance knobs excluded."""

        value = self.as_dict()
        for key in ("concurrency", "point_budget", "snapshot_interval", "tolerance"):
            value.pop(key, None)
        return hash_value(value)

    def family_fingerprint(self) -> str:
        """Parameters that define a compatible on-disk cache layout."""

        return hash_value(
            {
                "image_size": list(self.image_size),
                "levels": self.levels,
                "pixel_mode": self.pixel_mode,
                "schema": 1,
                "sh_degree": self.sh_degree,
                "tile_size": self.tile_size,
            }
        )

    def copy(self) -> "SceneConfig":
        return SceneConfig.from_dict(self.as_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneConfig":
        kwargs = dict(value)
        if "image_size" in kwargs:
            kwargs["image_size"] = tuple(kwargs["image_size"])
        return cls(**kwargs)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def validate_config_dict(value: Any) -> tuple[SceneConfig | None, list[Failure]]:
    failures: list[Failure] = []

    def add(code: str, message: str) -> None:
        failures.append(Failure(code=code, message=message, source="config"))

    if not isinstance(value, dict):
        add(codes.E_CONFIG_JSON, "config root must be a JSON object")
        return None, failures

    required = [
        "image_size",
        "base_scale",
        "levels",
        "scale_decay",
        "level_density",
        "sh_degree",
        "frac_bits",
        "quant_gear",
        "render_samples",
        "kernel_budgets",
        "concurrency",
        "snapshot_interval",
        "tolerance",
    ]
    for key in required:
        if key not in value:
            add(codes.E_CONFIG_JSON, f"missing required config key: {key}")

    levels = _as_int(value.get("levels"))
    if levels is None or levels <= 0:
        add(codes.E_CONFIG_LEVELS, "levels must be a positive integer")
        levels = 0

    image_size = value.get("image_size")
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(_as_int(item) is None or int(item) <= 0 for item in image_size)
    ):
        add(codes.E_CONFIG_JSON, "image_size must be a positive [width, height] pair")
        image_size = (1, 1)

    base_scale = value.get("base_scale")
    if not ensure_finite_number(base_scale) or float(base_scale) <= 0:
        add(codes.E_CONFIG_JSON, "base_scale must be a positive finite number")

    decay = value.get("scale_decay")
    decay_values = decay if isinstance(decay, (list, tuple)) else [decay]
    if not decay_values or any(not ensure_finite_number(item) or float(item) <= 0 for item in decay_values):
        add(codes.E_CONFIG_SCALE_DECAY, "scale_decay must contain only positive finite numbers")
    elif levels and isinstance(decay, (list, tuple)) and len(decay_values) not in (1, levels):
        add(codes.E_CONFIG_SCALE_DECAY, "scale_decay list length must be 1 or levels")

    density = value.get("level_density")
    if not isinstance(density, list) or len(density) != levels:
        add(codes.E_CONFIG_DENSITY, "level_density must be a list with one value per level")
    elif any(not ensure_finite_number(item) or float(item) <= 0 for item in density):
        add(codes.E_CONFIG_DENSITY, "level_density values must be positive finite numbers")

    budgets = value.get("kernel_budgets")
    if not isinstance(budgets, list) or len(budgets) != levels:
        add(codes.E_CONFIG_KERNEL_BUDGET, "kernel_budgets must have one positive value per level")
    elif any(_as_int(item) is None or int(item) <= 0 for item in budgets):
        add(codes.E_CONFIG_KERNEL_BUDGET, "kernel_budgets values must be positive integers")

    frac_bits = _as_int(value.get("frac_bits"))
    if frac_bits not in codes.ALLOWED_FRAC_BITS:
        add(codes.E_CONFIG_FRAC_BITS, f"frac_bits must be one of {codes.ALLOWED_FRAC_BITS}")

    if value.get("quant_gear") not in codes.QUANT_GEARS:
        add(codes.E_CONFIG_QUANT_GEAR, f"quant_gear must be one of {codes.QUANT_GEARS}")

    sh_degree = _as_int(value.get("sh_degree"))
    if sh_degree is None or not 0 <= sh_degree <= 3:
        add(codes.E_CONFIG_SH_DEGREE, "sh_degree must be an integer in [0, 3]")

    render_samples = _as_int(value.get("render_samples"))
    if render_samples is None or render_samples <= 0:
        add(codes.E_CONFIG_RENDER_SAMPLES, "render_samples must be a positive integer")

    point_budget = _as_int(value.get("point_budget", 20000))
    if point_budget is None or point_budget <= 0:
        add(codes.E_CONFIG_POINT_BUDGET, "point_budget must be a positive integer")

    concurrency = _as_int(value.get("concurrency"))
    if concurrency is None or concurrency <= 0:
        add(codes.E_CONFIG_CONCURRENCY, "concurrency must be a positive integer")

    snapshot_interval = _as_int(value.get("snapshot_interval"))
    if snapshot_interval is None or snapshot_interval < 0:
        add(codes.E_CONFIG_SNAPSHOT_INTERVAL, "snapshot_interval must be a non-negative integer")

    tile_size = _as_int(value.get("tile_size", 16))
    if tile_size is None or tile_size <= 0:
        add(codes.E_CONFIG_TILE_SIZE, "tile_size must be a positive integer")

    tolerance = value.get("tolerance")
    if not isinstance(tolerance, dict):
        add(codes.E_CONFIG_TOLERANCE, "tolerance must be an object")
    else:
        for key in ("kernel", "render", "composite"):
            if not ensure_finite_number(tolerance.get(key)) or float(tolerance[key]) < 0:
                add(codes.E_CONFIG_TOLERANCE, f"tolerance.{key} must be a non-negative finite number")

    overrides = value.get("level_scale_overrides", [None] * levels)
    if not isinstance(overrides, list) or len(overrides) != levels:
        add(codes.E_CONFIG_LEVELS, "level_scale_overrides must have one item per level")
    elif any(item is not None and (not ensure_finite_number(item) or float(item) <= 0) for item in overrides):
        add(codes.E_CONFIG_LEVELS, "level_scale_overrides values must be positive finite numbers or null")

    if failures:
        return None, failures
    try:
        config = SceneConfig.from_dict(dict(value))
    except Exception as exc:  # pragma: no cover - defensive path
        add(codes.E_CONFIG_JSON, f"config coercion failed: {exc}")
        return None, failures
    return config, failures


def kernel_grid_size(config: SceneConfig, level: int) -> tuple[int, int]:
    width, height = config.image_size
    ratio = config.scale_ratios()[level]
    density = config.level_density[level]
    grid_w = max(1, int(round(float(width) * ratio * density)))
    grid_h = max(1, int(round(float(height) * ratio * density)))
    # Avoid unbounded allocation for malformed, adversarial density values.
    grid_w = min(grid_w, max(width * 8, 1))
    grid_h = min(grid_h, max(height * 8, 1))
    return grid_w, grid_h


def level_fingerprint(config: SceneConfig, level: int, grid_size: tuple[int, int], level_budget: int) -> str:
    scale = config.effective_scales()[level]
    seed_material = hash_value(
        {
            "base_scale": float(config.base_scale),
            "density": float(config.level_density[level]),
            "effective_scale": float(scale),
            "grid": list(grid_size),
            "image_size": list(config.image_size),
            "level": int(level),
            "seed": int(config.seed),
            "sh_degree": int(config.sh_degree),
        }
    )
    seed = int(hash_value(seed_material)[:16], 16)
    rng = np.random.default_rng(seed)
    projection = rng.uniform(-0.75, 0.75, size=(4, 4)).astype(np.float64)
    from .util import array_fingerprint

    return hash_value(
        {
            "base_scale": config.base_scale,
            "density": config.level_density[level],
            "effective_scale": scale,
            "grid": list(grid_size),
            "level": level,
            "level_budget": int(level_budget),
            "projection": array_fingerprint(projection),
            "sh_coefficients": (config.sh_degree + 1) ** 2,
            "sh_degree": config.sh_degree,
        }
    )


def projection_matrix_for_level(config: SceneConfig, level: int) -> tuple[np.ndarray, int]:
    scale = config.effective_scales()[level]
    grid = kernel_grid_size(config, level)
    seed_material = hash_value(
        {
            "base_scale": float(config.base_scale),
            "density": float(config.level_density[level]),
            "effective_scale": float(scale),
            "grid": list(grid),
            "image_size": list(config.image_size),
            "level": int(level),
            "seed": int(config.seed),
            "sh_degree": int(config.sh_degree),
        }
    )
    seed = int(hash_value(seed_material)[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.75, 0.75, size=(4, 4)).astype(np.float64), seed
