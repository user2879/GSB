"""Pipeline parameter objects, validation, and parameter version stamps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .errors import mask_invalid, region_bounds
from .util import sha256_json


@dataclass(frozen=True)
class ScopedRect:
    x: int
    y: int
    width: int
    height: int
    frame_start: int | None = None
    frame_end: int | None = None

    def applies_to(self, frame_index: int) -> bool:
        if self.frame_start is not None and frame_index < self.frame_start:
            return False
        if self.frame_end is not None and frame_index >= self.frame_end:
            return False
        return True

    def to_dict(self, *, include_scope: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }
        if include_scope:
            value["frame_start"] = self.frame_start
            value["frame_end"] = self.frame_end
        return value


@dataclass(frozen=True)
class SamplingSpec:
    regions: tuple[ScopedRect, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"regions": [region.to_dict() for region in self.regions]}


@dataclass(frozen=True)
class WatermarkMask:
    regions: tuple[ScopedRect, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"regions": [region.to_dict() for region in self.regions]}


@dataclass(frozen=True)
class AnalysisWindow:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class PipelineConfig:
    gop_size: int
    sampling: SamplingSpec
    watermark_mask: WatermarkMask
    analysis_window: AnalysisWindow
    suspicion_threshold: float = 0.35
    feature_version: str = "frame-feature-v1"

    def validate(self, width: int, height: int, frame_count: int) -> None:
        if self.gop_size <= 0:
            raise region_bounds("gop_size must be positive", gop_size=self.gop_size)
        if not 0.0 <= self.suspicion_threshold <= 1.0:
            raise region_bounds(
                "suspicion_threshold must be in [0, 1]",
                threshold=self.suspicion_threshold,
            )
        if not self.sampling.regions:
            raise region_bounds("at least one sampling region is required")
        for index, region in enumerate(self.sampling.regions):
            _validate_rect(
                region,
                width,
                height,
                frame_count,
                "sampling_region",
                region_bounds,
                index,
            )
        for index, region in enumerate(self.watermark_mask.regions):
            _validate_rect(
                region,
                width,
                height,
                frame_count,
                "watermark_mask",
                mask_invalid,
                index,
            )
        window = self.analysis_window
        if window.start_frame < 0 or window.end_frame > frame_count:
            raise region_bounds(
                "analysis window lies outside the source",
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                frame_count=frame_count,
            )
        if window.end_frame <= window.start_frame:
            raise region_bounds(
                "analysis window must contain at least one frame",
                start_frame=window.start_frame,
                end_frame=window.end_frame,
            )

    def parameter_versions(self) -> dict[str, str]:
        return {
            "sampling_region": sha256_json(self.sampling.to_dict()),
            "watermark_mask": sha256_json(self.watermark_mask.to_dict()),
            "analysis_window": sha256_json(
                {
                    "start_frame": self.analysis_window.start_frame,
                    "end_frame": self.analysis_window.end_frame,
                }
            ),
            "threshold": sha256_json(self.suspicion_threshold),
            "gop_size": sha256_json(self.gop_size),
            "feature_version": sha256_json(self.feature_version),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "gop_size": self.gop_size,
            "sampling": self.sampling.to_dict(),
            "watermark_mask": self.watermark_mask.to_dict(),
            "analysis_window": {
                "start_frame": self.analysis_window.start_frame,
                "end_frame": self.analysis_window.end_frame,
            },
            "suspicion_threshold": self.suspicion_threshold,
            "feature_version": self.feature_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PipelineConfig":
        return cls(
            gop_size=int(value["gop_size"]),
            sampling=SamplingSpec(
                tuple(_rect_from_dict(item) for item in value["sampling"]["regions"])
            ),
            watermark_mask=WatermarkMask(
                tuple(
                    _rect_from_dict(item)
                    for item in value["watermark_mask"]["regions"]
                )
            ),
            analysis_window=AnalysisWindow(
                start_frame=int(value["analysis_window"]["start_frame"]),
                end_frame=int(value["analysis_window"]["end_frame"]),
            ),
            suspicion_threshold=float(value.get("suspicion_threshold", 0.35)),
            feature_version=str(value.get("feature_version", "frame-feature-v1")),
        )

    def with_regions(
        self,
        *,
        sampling: Iterable[ScopedRect] | None = None,
        mask: Iterable[ScopedRect] | None = None,
        window: AnalysisWindow | None = None,
        threshold: float | None = None,
    ) -> "PipelineConfig":
        return replace(
            self,
            sampling=SamplingSpec(tuple(sampling)) if sampling is not None else self.sampling,
            watermark_mask=WatermarkMask(tuple(mask)) if mask is not None else self.watermark_mask,
            analysis_window=window if window is not None else self.analysis_window,
            suspicion_threshold=threshold
            if threshold is not None
            else self.suspicion_threshold,
        )


def _rect_from_dict(value: dict[str, Any]) -> ScopedRect:
    return ScopedRect(
        x=int(value["x"]),
        y=int(value["y"]),
        width=int(value["width"]),
        height=int(value["height"]),
        frame_start=None
        if value.get("frame_start") is None
        else int(value["frame_start"]),
        frame_end=None if value.get("frame_end") is None else int(value["frame_end"]),
    )


def _validate_rect(
    region: ScopedRect,
    width: int,
    height: int,
    frame_count: int,
    name: str,
    error_factory: Any,
    index: int,
) -> None:
    if region.width <= 0 or region.height <= 0:
        raise error_factory(
            f"{name} width and height must be positive",
            parameter=name,
            region_index=index,
            region=region.to_dict(),
        )
    if region.x < 0 or region.y < 0:
        raise error_factory(
            f"{name} origin must be non-negative",
            parameter=name,
            region_index=index,
            region=region.to_dict(),
        )
    if region.x + region.width > width or region.y + region.height > height:
        raise error_factory(
            f"{name} exceeds frame bounds",
            parameter=name,
            region_index=index,
            region=region.to_dict(),
            frame_size={"width": width, "height": height},
        )
    start = 0 if region.frame_start is None else region.frame_start
    end = frame_count if region.frame_end is None else region.frame_end
    if start < 0 or end > frame_count or end <= start:
        raise error_factory(
            f"{name} has an invalid frame scope",
            parameter=name,
            region_index=index,
            region=region.to_dict(),
            frame_count=frame_count,
        )


def changed_parameter_keys(
    previous: dict[str, str], current: dict[str, str]
) -> list[str]:
    keys = sorted(set(previous) | set(current))
    return [key for key in keys if previous.get(key) != current.get(key)]

