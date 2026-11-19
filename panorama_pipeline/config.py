from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_PARAM_INVALID, PipelineError
from .io_utils import canonical_json_bytes, read_json, sha256_bytes


RETENTION_MODES = {"prefer_non_crowd", "confidence", "drop_crowd"}


@dataclass(frozen=True)
class RetentionPolicy:
    mode: str = "prefer_non_crowd"
    min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in RETENTION_MODES:
            raise PipelineError(
                E_PARAM_INVALID,
                f"unsupported retention mode: {self.mode}",
            )
        if not 0.0 <= self.min_coverage <= 1.0:
            raise PipelineError(
                E_PARAM_INVALID,
                "pixel retention min_coverage must be in [0, 1]",
            )


@dataclass(frozen=True)
class PipelineConfig:
    iou_match_threshold: float
    pixel_retention: RetentionPolicy
    confidence_default: float
    default_crowd: bool
    palette: dict[int, tuple[int, int, int, int]]
    interaction_budget_ms: int
    max_workers: int
    snapshot_commit_interval: int
    boundary_epsilon: float = 1e-6
    frame: str = "panorama_000"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_match_threshold <= 1.0:
            raise PipelineError(
                E_PARAM_INVALID,
                "iou_match_threshold must be in [0, 1]",
            )
        if not 0.0 <= self.confidence_default <= 1.0:
            raise PipelineError(
                E_PARAM_INVALID,
                "confidence_default must be in [0, 1]",
            )
        if self.interaction_budget_ms <= 0:
            raise PipelineError(
                E_PARAM_INVALID,
                "interaction_budget_ms must be positive",
            )
        if self.max_workers <= 0:
            raise PipelineError(
                E_PARAM_INVALID,
                "max_workers must be positive",
            )
        if self.snapshot_commit_interval <= 0:
            raise PipelineError(
                E_PARAM_INVALID,
                "snapshot_commit_interval must be positive",
            )
        if self.boundary_epsilon < 0:
            raise PipelineError(
                E_PARAM_INVALID,
                "boundary_epsilon must be non-negative",
            )

    @property
    def config_hash(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_match_threshold": self.iou_match_threshold,
            "pixel_retention": {
                "mode": self.pixel_retention.mode,
                "min_coverage": self.pixel_retention.min_coverage,
            },
            "confidence_default": self.confidence_default,
            "default_crowd": self.default_crowd,
            "palette": {
                str(key): list(value) for key, value in sorted(self.palette.items())
            },
            "interaction_budget_ms": self.interaction_budget_ms,
            "max_workers": self.max_workers,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "boundary_epsilon": self.boundary_epsilon,
            "frame": self.frame,
        }

    def params_snapshot(self) -> dict[str, Any]:
        return {
            "iou_match_threshold": self.iou_match_threshold,
            "pixel_retention": {
                "mode": self.pixel_retention.mode,
                "min_coverage": self.pixel_retention.min_coverage,
            },
        }

    def params_snapshot_version(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.params_snapshot()))

    def with_params(
        self,
        *,
        iou_match_threshold: float | None = None,
        retention_mode: str | None = None,
        min_coverage: float | None = None,
    ) -> "PipelineConfig":
        return dataclasses.replace(
            self,
            iou_match_threshold=(
                self.iou_match_threshold
                if iou_match_threshold is None
                else float(iou_match_threshold)
            ),
            pixel_retention=RetentionPolicy(
                mode=(
                    self.pixel_retention.mode
                    if retention_mode is None
                    else str(retention_mode)
                ),
                min_coverage=(
                    self.pixel_retention.min_coverage
                    if min_coverage is None
                    else float(min_coverage)
                ),
            ),
        )

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        source_path: str | None = None,
    ) -> "PipelineConfig":
        try:
            retention = value.get("pixel_retention", {})
            if isinstance(retention, str):
                retention = {"mode": retention}
            palette_value = value["palette"]
            if isinstance(palette_value, list):
                palette_items = enumerate(palette_value)
            else:
                palette_items = palette_value.items()
            palette: dict[int, tuple[int, int, int, int]] = {}
            for key, entry in palette_items:
                if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                    raise ValueError(f"palette entry {key} must contain RGBA")
                channels = tuple(int(channel) for channel in entry)
                if any(channel < 0 or channel > 255 for channel in channels):
                    raise ValueError(f"palette entry {key} is outside byte range")
                palette[int(key)] = channels
            return cls(
                iou_match_threshold=float(value["iou_match_threshold"]),
                pixel_retention=RetentionPolicy(
                    mode=str(retention.get("mode", "prefer_non_crowd")),
                    min_coverage=float(retention.get("min_coverage", 0.5)),
                ),
                confidence_default=float(value.get("confidence_default", 0.5)),
                default_crowd=bool(value.get("default_crowd", False)),
                palette=palette,
                interaction_budget_ms=int(value["interaction_budget_ms"]),
                max_workers=int(value["max_workers"]),
                snapshot_commit_interval=int(value["snapshot_commit_interval"]),
                boundary_epsilon=float(value.get("boundary_epsilon", 1e-6)),
                frame=str(value.get("frame", "panorama_000")),
                source_path=source_path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(
                E_PARAM_INVALID,
                f"invalid pipeline configuration: {exc}",
            ) from exc

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(read_json(path), source_path=str(Path(path).resolve()))
