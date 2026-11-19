"""Serializable scene and rendering data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Vec3 = tuple[float, float, float]


def q(value: float, places: int = 6) -> float:
    return round(float(value), places)


@dataclass(frozen=True)
class Camera:
    position: Vec3
    target: Vec3
    up: Vec3
    fov_y_deg: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_deg": q(self.fov_y_deg),
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class TransparentSurface:
    surface_id: str
    kind: str
    center: Vec3
    normal: Vec3
    radius: float
    half_width: float
    half_height: float
    half_depth: float
    curvature: float
    ior: float
    transmission: float
    tint: Vec3
    sort_index: int

    def with_ior(self, ior: float) -> "TransparentSurface":
        return TransparentSurface(
            surface_id=self.surface_id,
            kind=self.kind,
            center=self.center,
            normal=self.normal,
            radius=self.radius,
            half_width=self.half_width,
            half_height=self.half_height,
            half_depth=self.half_depth,
            curvature=self.curvature,
            ior=float(ior),
            transmission=self.transmission,
            tint=self.tint,
            sort_index=self.sort_index,
        )

    def to_dict(self) -> dict[str, Any]:
        back_normal = tuple(-value for value in self.normal)
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "center": list(self.center),
            "normal": list(self.normal),
            "faces": [
                {
                    "face": "front",
                    "normal": list(self.normal),
                    "ior": q(self.ior),
                },
                {
                    "face": "back",
                    "normal": list(back_normal),
                    "ior": q(self.ior),
                },
            ],
            "radius": q(self.radius),
            "half_width": q(self.half_width),
            "half_height": q(self.half_height),
            "half_depth": q(self.half_depth),
            "curvature": q(self.curvature),
            "ior": q(self.ior),
            "transmission": q(self.transmission),
            "tint": list(self.tint),
            "sort_index": self.sort_index,
        }


@dataclass(frozen=True)
class LightCone:
    cone_id: str
    apex: Vec3
    direction: Vec3
    length: float
    radius_start: float
    radius_end: float
    scatter: float
    absorption: float
    color: Vec3
    shadow_samples: int
    jitter_seed: int

    def with_scatter(self, scatter: float) -> "LightCone":
        return LightCone(
            cone_id=self.cone_id,
            apex=self.apex,
            direction=self.direction,
            length=self.length,
            radius_start=self.radius_start,
            radius_end=self.radius_end,
            scatter=float(scatter),
            absorption=self.absorption,
            color=self.color,
            shadow_samples=self.shadow_samples,
            jitter_seed=self.jitter_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cone_id": self.cone_id,
            "apex": list(self.apex),
            "direction": list(self.direction),
            "length": q(self.length),
            "radius_start": q(self.radius_start),
            "radius_end": q(self.radius_end),
            "scatter": q(self.scatter),
            "absorption": q(self.absorption),
            "color": list(self.color),
            "shadow_samples": self.shadow_samples,
            "jitter_seed": self.jitter_seed,
        }


@dataclass(frozen=True)
class Occluder:
    occluder_id: str
    center: Vec3
    half_extents: Vec3
    albedo: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "occluder_id": self.occluder_id,
            "center": list(self.center),
            "half_extents": list(self.half_extents),
            "albedo": list(self.albedo),
        }


@dataclass(frozen=True)
class Ground:
    plane_y: float
    albedo_a: Vec3
    albedo_b: Vec3
    checker_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_y": q(self.plane_y),
            "albedo_a": list(self.albedo_a),
            "albedo_b": list(self.albedo_b),
            "checker_scale": q(self.checker_scale),
        }


@dataclass(frozen=True)
class VolumeSettings:
    base_step: float
    max_steps: int
    near_plane: float
    near_clip: float
    far_clip: float
    phase_isotropic: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_step": q(self.base_step),
            "max_steps": self.max_steps,
            "near_plane": q(self.near_plane),
            "near_clip": q(self.near_clip),
            "far_clip": q(self.far_clip),
            "phase_isotropic": q(self.phase_isotropic),
        }


@dataclass(frozen=True)
class Scene:
    schema_version: int
    seed: int
    camera: Camera
    surfaces: tuple[TransparentSurface, ...]
    light_cone: LightCone
    occluders: tuple[Occluder, ...]
    ground: Ground
    volume: VolumeSettings
    metadata: dict[str, Any] = field(default_factory=dict)

    def surface_by_id(self, surface_id: str) -> TransparentSurface:
        for surface in self.surfaces:
            if surface.surface_id == surface_id:
                return surface
        raise KeyError(surface_id)

    def replace_surface_ior(self, surface_id: str, ior: float) -> "Scene":
        surfaces = tuple(
            surface.with_ior(ior) if surface.surface_id == surface_id else surface
            for surface in self.surfaces
        )
        return Scene(
            schema_version=self.schema_version,
            seed=self.seed,
            camera=self.camera,
            surfaces=surfaces,
            light_cone=self.light_cone,
            occluders=self.occluders,
            ground=self.ground,
            volume=self.volume,
            metadata=self.metadata,
        )

    def replace_cone_scatter(self, scatter: float) -> "Scene":
        return Scene(
            schema_version=self.schema_version,
            seed=self.seed,
            camera=self.camera,
            surfaces=self.surfaces,
            light_cone=self.light_cone.with_scatter(scatter),
            occluders=self.occluders,
            ground=self.ground,
            volume=self.volume,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "camera": self.camera.to_dict(),
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "light_cone": self.light_cone.to_dict(),
            "occluders": [occluder.to_dict() for occluder in self.occluders],
            "ground": self.ground.to_dict(),
            "volume": self.volume.to_dict(),
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass
class TileResult:
    key: str
    tx: int
    ty: int
    x0: int
    y0: int
    x1: int
    y1: int
    image: np.ndarray
    visibility_mask: np.ndarray
    volume_trace: np.ndarray
    volume_inside: np.ndarray
    status: str = "COMPLETED"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0
