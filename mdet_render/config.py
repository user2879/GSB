"""Configuration loading and schema/enum validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .fixed import FixedPoint
from .util import read_json_file


def _require_int(value: Any, field_name: str, *, minimum: int | None = None,
                 maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be an integer", {"field": field_name})
    if minimum is not None and value < minimum:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} is below its minimum", {"field": field_name,
                                                                                       "value": value,
                                                                                       "minimum": minimum})
    if maximum is not None and value > maximum:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} is above its maximum", {"field": field_name,
                                                                                       "value": value,
                                                                                       "maximum": maximum})
    return value


def _require_number(value: Any, field_name: str, *, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be a number", {"field": field_name})
    number = float(value)
    if not math.isfinite(number):
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be finite", {"field": field_name})
    if minimum is not None and number < minimum:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} is below its minimum", {"field": field_name,
                                                                                       "value": number,
                                                                                       "minimum": minimum})
    if maximum is not None and number > maximum:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} is above its maximum", {"field": field_name,
                                                                                       "value": number,
                                                                                       "maximum": maximum})
    return number


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be a non-empty string", {"field": field_name})
    return value


def _dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be an object", {"field": field_name})
    return value


@dataclass(frozen=True)
class FrameSpec:
    frame: int
    prompt_path: Path
    image_path: Path | None


@dataclass
class RenderConfig:
    source_path: Path
    base_dir: Path
    schema_version: int
    width: int
    height: int
    pixel_mode: str
    condition_dim: int
    vocabulary_path: Path
    motion_path: Path
    frames: list[FrameSpec]
    fixed: FixedPoint
    weights: dict[str, float]
    image_grid_rows: int
    image_grid_cols: int
    motion_block_rows: int
    motion_block_cols: int
    noise_seed: int
    noise_table_size: int
    latent_seed: int
    denoise_steps: int
    guidance_strength: float
    decode_block_rows: int
    projection_seed: int
    max_workers: int
    point_budget: int
    max_denoise_steps: int
    lease_ttl_ms: int

    @property
    def frame_indices(self) -> list[int]:
        return [spec.frame for spec in self.frames]

    @property
    def denoise_points_per_step(self) -> int:
        return self.width * self.height * 3

    def frame_spec(self, frame: int) -> FrameSpec:
        for spec in self.frames:
            if spec.frame == frame:
                return spec
        raise BusinessError("E_FRAME", "unknown frame index", {"frame": frame, "known": self.frame_indices})

    def validate(self) -> None:
        if self.schema_version != 1:
            raise BusinessError("E_CONFIG_SCHEMA", "unsupported config schema_version",
                                {"schema_version": self.schema_version})
        if self.width <= 0 or self.height <= 0:
            raise BusinessError("E_CONFIG_SCHEMA", "resolution must be positive")
        if self.pixel_mode != "RGB":
            raise BusinessError("E_PNG_PIXEL_MODE", "configured pixel mode must be RGB",
                                {"pixel_mode": self.pixel_mode})
        if self.condition_dim <= 0:
            raise BusinessError("E_CONFIG_SCHEMA", "condition_dim must be positive")
        if not (1 <= self.image_grid_rows <= self.height):
            raise BusinessError("E_CONFIG_SCHEMA", "image grid rows are outside the image",
                                {"grid_rows": self.image_grid_rows, "height": self.height})
        if not (1 <= self.image_grid_cols <= self.width):
            raise BusinessError("E_CONFIG_SCHEMA", "image grid columns are outside the image",
                                {"grid_cols": self.image_grid_cols, "width": self.width})
        if self.height % self.image_grid_rows != 0 or self.width % self.image_grid_cols != 0:
            raise BusinessError("E_VECTOR_DIM", "resolution must be divisible by the image block grid",
                                {"resolution": [self.width, self.height],
                                 "grid": [self.image_grid_rows, self.image_grid_cols]})
        image_dim = self.image_grid_rows * self.image_grid_cols * 5
        if image_dim != self.condition_dim:
            raise BusinessError("E_VECTOR_DIM", "image condition vector dimension is incompatible with condition_dim",
                                {"image_dim": image_dim, "condition_dim": self.condition_dim})
        motion_dim = self.motion_block_rows * self.motion_block_cols * 2
        if motion_dim != self.condition_dim:
            raise BusinessError("E_MOTION_DIM", "motion vector table dimension is incompatible with condition_dim",
                                {"motion_dim": motion_dim, "condition_dim": self.condition_dim})
        if self.motion_block_rows <= 0 or self.motion_block_cols <= 0:
            raise BusinessError("E_MOTION_DIM", "motion block rows/columns must be positive")
        self.fixed.validate()
        if set(self.weights) != {"text", "image", "motion"}:
            raise BusinessError("E_CONFIG_SCHEMA", "fusion weights must define text, image and motion exactly",
                                {"keys": sorted(self.weights)})
        for name, weight in self.weights.items():
            _require_number(weight, f"fusion.weights.{name}", minimum=0.0, maximum=8.0)
        if not (1 <= self.denoise_steps <= self.max_denoise_steps):
            raise BusinessError("E_CONFIG_SCHEMA", "denoise.steps must be between 1 and execution.max_denoise_steps",
                                {"steps": self.denoise_steps, "max": self.max_denoise_steps})
        if not (1 <= self.decode_block_rows <= self.height):
            raise BusinessError("E_CONFIG_SCHEMA", "decode block rows must be within the image height",
                                {"block_rows": self.decode_block_rows, "height": self.height})
        if not (1 <= self.noise_table_size <= 128):
            raise BusinessError("E_CONFIG_SCHEMA", "noise.table_size must be in 1..128")
        if not (1 <= self.max_workers <= 32):
            raise BusinessError("E_CONFIG_SCHEMA", "execution.max_workers must be in 1..32")
        if self.point_budget < self.denoise_points_per_step:
            raise BusinessError(
                "E_POINT_BUDGET",
                "execution.point_budget must cover at least one denoise step",
                {"point_budget": self.point_budget, "step_points": self.denoise_points_per_step},
            )
        if self.lease_ttl_ms <= 0:
            raise BusinessError("E_LEASE_CONFIG", "lease.ttl_ms must be positive")
        frames = [spec.frame for spec in self.frames]
        if not frames:
            raise BusinessError("E_CONFIG_SCHEMA", "at least one frame is required")
        if len(frames) != len(set(frames)):
            raise BusinessError("E_CONFIG_SCHEMA", "frame indices must be unique", {"frames": frames})
        if sorted(frames) != list(range(len(frames))):
            raise BusinessError("E_CONFIG_SCHEMA", "frame indices must be contiguous from zero", {"frames": frames})

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resolution": {"width": self.width, "height": self.height},
            "pixel_mode": self.pixel_mode,
            "condition_dim": self.condition_dim,
            "frames": [
                {"frame": spec.frame, "prompt": str(spec.prompt_path),
                 "image": None if spec.image_path is None else str(spec.image_path)}
                for spec in self.frames
            ],
            "fixed_point": self.fixed.as_dict(),
            "fusion": {"weights": self.weights},
            "image_encoding": {"grid_rows": self.image_grid_rows, "grid_cols": self.image_grid_cols},
            "motion_encoding": {"block_rows": self.motion_block_rows, "block_cols": self.motion_block_cols},
            "noise": {"seed": self.noise_seed, "table_size": self.noise_table_size},
            "latent": {"seed": self.latent_seed},
            "denoise": {"steps": self.denoise_steps, "guidance_strength": self.guidance_strength},
            "decode": {"block_rows": self.decode_block_rows, "projection_seed": self.projection_seed},
            "execution": {
                "max_workers": self.max_workers,
                "point_budget": self.point_budget,
                "max_denoise_steps": self.max_denoise_steps,
            },
            "lease": {"ttl_ms": self.lease_ttl_ms},
        }


def _path(base_dir: Path, value: Any, field_name: str, *, nullable: bool = False) -> Path | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise BusinessError("E_CONFIG_SCHEMA", f"{field_name} must be a non-empty path", {"field": field_name})
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def load_render_config(path: str | Path) -> RenderConfig:
    source_path = Path(path).resolve()
    raw = _dict(read_json_file(source_path, code="E_CONFIG_JSON"), "config")
    base_dir = source_path.parent
    resolution = _dict(raw.get("resolution"), "resolution")
    fixed_raw = _dict(raw.get("fixed_point"), "fixed_point")
    fusion = _dict(raw.get("fusion"), "fusion")
    weights_raw = _dict(fusion.get("weights"), "fusion.weights")
    image_encoding = _dict(raw.get("image_encoding"), "image_encoding")
    motion_encoding = _dict(raw.get("motion_encoding"), "motion_encoding")
    noise = _dict(raw.get("noise"), "noise")
    latent = _dict(raw.get("latent"), "latent")
    denoise = _dict(raw.get("denoise"), "denoise")
    decode = _dict(raw.get("decode"), "decode")
    execution = _dict(raw.get("execution"), "execution")
    lease = _dict(raw.get("lease"), "lease")
    frames_raw = raw.get("frames")
    if not isinstance(frames_raw, list):
        raise BusinessError("E_CONFIG_SCHEMA", "frames must be an array")
    frames: list[FrameSpec] = []
    for index, item in enumerate(frames_raw):
        item = _dict(item, f"frames[{index}]")
        frame = _require_int(item.get("frame"), f"frames[{index}].frame", minimum=0)
        prompt_path = _path(base_dir, item.get("prompt"), f"frames[{index}].prompt")
        image_path = _path(base_dir, item.get("image"), f"frames[{index}].image", nullable=True)
        assert prompt_path is not None
        frames.append(FrameSpec(frame=frame, prompt_path=prompt_path, image_path=image_path))
    weights = {
        name: _require_number(weight, f"fusion.weights.{name}", minimum=0.0, maximum=8.0)
        for name, weight in weights_raw.items()
    }
    cfg = RenderConfig(
        source_path=source_path,
        base_dir=base_dir,
        schema_version=_require_int(raw.get("schema_version"), "schema_version"),
        width=_require_int(resolution.get("width"), "resolution.width", minimum=1),
        height=_require_int(resolution.get("height"), "resolution.height", minimum=1),
        pixel_mode=_require_str(raw.get("pixel_mode"), "pixel_mode"),
        condition_dim=_require_int(raw.get("condition_dim"), "condition_dim", minimum=1),
        vocabulary_path=_path(base_dir, raw.get("vocabulary"), "vocabulary"),
        motion_path=_path(base_dir, raw.get("motion"), "motion"),
        frames=frames,
        fixed=FixedPoint(
            fraction_bits=_require_int(fixed_raw.get("fraction_bits"), "fixed_point.fraction_bits"),
            storage_bits=_require_int(fixed_raw.get("storage_bits"), "fixed_point.storage_bits"),
            quant_level=_require_int(fixed_raw.get("quant_level"), "fixed_point.quant_level"),
        ),
        weights=weights,
        image_grid_rows=_require_int(image_encoding.get("grid_rows"), "image_encoding.grid_rows", minimum=1),
        image_grid_cols=_require_int(image_encoding.get("grid_cols"), "image_encoding.grid_cols", minimum=1),
        motion_block_rows=_require_int(motion_encoding.get("block_rows"), "motion_encoding.block_rows",
                                       minimum=1),
        motion_block_cols=_require_int(motion_encoding.get("block_cols"), "motion_encoding.block_cols",
                                       minimum=1),
        noise_seed=_require_int(noise.get("seed"), "noise.seed"),
        noise_table_size=_require_int(noise.get("table_size"), "noise.table_size", minimum=1),
        latent_seed=_require_int(latent.get("seed"), "latent.seed"),
        denoise_steps=_require_int(denoise.get("steps"), "denoise.steps", minimum=1),
        guidance_strength=_require_number(denoise.get("guidance_strength"), "denoise.guidance_strength",
                                          minimum=0.0, maximum=8.0),
        decode_block_rows=_require_int(decode.get("block_rows"), "decode.block_rows", minimum=1),
        projection_seed=_require_int(decode.get("projection_seed"), "decode.projection_seed"),
        max_workers=_require_int(execution.get("max_workers"), "execution.max_workers", minimum=1),
        point_budget=_require_int(execution.get("point_budget"), "execution.point_budget", minimum=1),
        max_denoise_steps=_require_int(execution.get("max_denoise_steps"), "execution.max_denoise_steps",
                                       minimum=1),
        lease_ttl_ms=_require_int(lease.get("ttl_ms"), "lease.ttl_ms", minimum=1),
    )
    assert cfg.vocabulary_path is not None and cfg.motion_path is not None
    cfg.validate()
    return cfg
