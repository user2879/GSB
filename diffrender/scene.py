"""Scene description, flat parameter vector, and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .errors import DiffRenderError, E_SCENE_INVALID
from .hashing import canonical_json_hash, digest_sequence, float_bits
from .math3d import camera_basis, dot, norm, vsub


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float
    albedo: float
    roughness: float


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple[float, float, float]
    ambient: float


@dataclass(frozen=True)
class Camera:
    eye: tuple[float, float, float]
    yaw: float
    pitch: float
    roll: float
    focal: float


@dataclass(frozen=True)
class Scene:
    width: int
    height: int
    tile_size: int
    spheres: tuple[Sphere, ...]
    light: DirectionalLight
    camera: Camera
    background: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.tile_size <= 0:
            raise DiffRenderError(E_SCENE_INVALID, "image and tile dimensions must be positive")
        if self.width % self.tile_size or self.height % self.tile_size:
            raise DiffRenderError(E_SCENE_INVALID, "image dimensions must be divisible by tile size")
        if not self.spheres:
            raise DiffRenderError(E_SCENE_INVALID, "scene requires at least one sphere")
        for i, sphere in enumerate(self.spheres):
            if sphere.radius <= 0.0 or not math.isfinite(sphere.radius):
                raise DiffRenderError(
                    E_SCENE_INVALID, f"sphere {i} has invalid radius", {"sphere": i}
                )
            if not (0.0 <= sphere.albedo <= 1.0):
                raise DiffRenderError(
                    E_SCENE_INVALID, f"sphere {i} albedo must be in [0, 1]", {"sphere": i}
                )
            if sphere.roughness < 0.0 or not math.isfinite(sphere.roughness):
                raise DiffRenderError(
                    E_SCENE_INVALID, f"sphere {i} roughness must be finite and >= 0", {"sphere": i}
                )
            if not all(math.isfinite(v) for v in sphere.center):
                raise DiffRenderError(E_SCENE_INVALID, f"sphere {i} center is not finite")
        if norm(self.light.direction) <= 1.0e-12:
            raise DiffRenderError(E_SCENE_INVALID, "light direction must be non-zero")
        if not (0.0 <= self.light.ambient <= 1.0):
            raise DiffRenderError(E_SCENE_INVALID, "ambient must be in [0, 1]")
        if self.camera.focal <= 0.0 or not math.isfinite(self.camera.focal):
            raise DiffRenderError(E_SCENE_INVALID, "camera focal length must be positive")
        if not all(math.isfinite(v) for v in self.camera.eye):
            raise DiffRenderError(E_SCENE_INVALID, "camera eye is not finite")
        right, up, forward = camera_basis(self.camera.yaw, self.camera.pitch, self.camera.roll)
        for i, sphere in enumerate(self.spheres):
            delta = vsub(sphere.center, self.camera.eye)
            depth = dot(delta, forward)
            if depth <= sphere.radius + 1.0e-9:
                raise DiffRenderError(
                    E_SCENE_INVALID,
                    f"sphere {i} is not fully in front of the camera",
                    {"sphere": i, "camera_depth": depth},
                )

    def sphere_param_keys(self, index: int) -> tuple[str, ...]:
        p = f"sphere.{index}"
        return (f"{p}.cx", f"{p}.cy", f"{p}.cz", f"{p}.radius", f"{p}.albedo", f"{p}.roughness")

    def light_param_keys(self) -> tuple[str, ...]:
        return ("light.dx", "light.dy", "light.dz", "light.ambient")

    def camera_param_keys(self) -> tuple[str, ...]:
        return ("camera.ex", "camera.ey", "camera.ez", "camera.yaw", "camera.pitch", "camera.roll", "camera.focal")

    def param_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for i in range(len(self.spheres)):
            keys.extend(self.sphere_param_keys(i))
        keys.extend(self.light_param_keys())
        keys.extend(self.camera_param_keys())
        return tuple(keys)

    def sphere_geometry_keys(self, index: int) -> tuple[str, ...]:
        p = f"sphere.{index}"
        return (f"{p}.cx", f"{p}.cy", f"{p}.cz", f"{p}.radius")

    def sphere_material_keys(self, index: int) -> tuple[str, ...]:
        p = f"sphere.{index}"
        return (f"{p}.albedo", f"{p}.roughness")

    def light_keys(self) -> tuple[str, ...]:
        return self.light_param_keys()

    def value(self, key: str) -> float:
        parts = key.split(".")
        if parts[0] == "sphere":
            idx = int(parts[1])
            sphere = self.spheres[idx]
            field = parts[2]
            if field in ("cx", "cy", "cz"):
                return sphere.center[{"cx": 0, "cy": 1, "cz": 2}[field]]
            return float(getattr(sphere, field))
        if parts[0] == "light":
            if parts[1] in ("dx", "dy", "dz"):
                return self.light.direction[{"dx": 0, "dy": 1, "dz": 2}[parts[1]]]
            return float(self.light.ambient)
        if parts[0] == "camera":
            if parts[1] in ("ex", "ey", "ez"):
                return self.camera.eye[{"ex": 0, "ey": 1, "ez": 2}[parts[1]]]
            return float(getattr(self.camera, parts[1]))
        raise DiffRenderError(E_SCENE_INVALID, f"unknown parameter key {key!r}")

    def with_value(self, key: str, value: float) -> "Scene":
        if not math.isfinite(value):
            raise DiffRenderError(E_SCENE_INVALID, f"parameter {key} must be finite")
        parts = key.split(".")
        spheres = list(self.spheres)
        light = self.light
        camera = self.camera
        if parts[0] == "sphere":
            idx = int(parts[1])
            sphere = spheres[idx]
            field = parts[2]
            if field in ("cx", "cy", "cz"):
                center = list(sphere.center)
                center[{"cx": 0, "cy": 1, "cz": 2}[field]] = float(value)
                spheres[idx] = replace(sphere, center=tuple(center))
            elif field in ("radius", "albedo", "roughness"):
                spheres[idx] = replace(sphere, **{field: float(value)})
            else:
                raise DiffRenderError(E_SCENE_INVALID, f"unknown sphere parameter {key!r}")
        elif parts[0] == "light":
            if parts[1] in ("dx", "dy", "dz"):
                direction = list(light.direction)
                direction[{"dx": 0, "dy": 1, "dz": 2}[parts[1]]] = float(value)
                light = replace(light, direction=tuple(direction))
            elif parts[1] == "ambient":
                light = replace(light, ambient=float(value))
            else:
                raise DiffRenderError(E_SCENE_INVALID, f"unknown light parameter {key!r}")
        elif parts[0] == "camera":
            if parts[1] in ("ex", "ey", "ez"):
                eye = list(camera.eye)
                eye[{"ex": 0, "ey": 1, "ez": 2}[parts[1]]] = float(value)
                camera = replace(camera, eye=tuple(eye))
            elif parts[1] in ("yaw", "pitch", "roll", "focal"):
                camera = replace(camera, **{parts[1]: float(value)})
            else:
                raise DiffRenderError(E_SCENE_INVALID, f"unknown camera parameter {key!r}")
        else:
            raise DiffRenderError(E_SCENE_INVALID, f"unknown parameter key {key!r}")
        return Scene(
            width=self.width,
            height=self.height,
            tile_size=self.tile_size,
            spheres=tuple(spheres),
            light=light,
            camera=camera,
            background=self.background,
        )

    def to_vector(self) -> tuple[float, ...]:
        return tuple(self.value(key) for key in self.param_keys())

    def vector_map(self) -> dict[str, float]:
        return {key: self.value(key) for key in self.param_keys()}

    def param_content_hash(self, key: str) -> str:
        return canonical_json_hash(
            {"key": key, "value_bits": float_bits(self.value(key)).hex()}
        )

    def parameter_hash(self) -> str:
        return digest_sequence(float_bits(self.value(key)) for key in self.param_keys())

    def topology(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "sphere_count": len(self.spheres),
            "background": self.background,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "background": self.background,
            "spheres": [
                {
                    "center": list(s.center),
                    "radius": s.radius,
                    "albedo": s.albedo,
                    "roughness": s.roughness,
                }
                for s in self.spheres
            ],
            "light": {"direction": list(self.light.direction), "ambient": self.light.ambient},
            "camera": {
                "eye": list(self.camera.eye),
                "yaw": self.camera.yaw,
                "pitch": self.camera.pitch,
                "roll": self.camera.roll,
                "focal": self.camera.focal,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        try:
            return cls(
                width=int(data["width"]),
                height=int(data["height"]),
                tile_size=int(data["tile_size"]),
                background=float(data.get("background", 0.0)),
                spheres=tuple(
                    Sphere(
                        center=tuple(float(v) for v in item["center"]),
                        radius=float(item["radius"]),
                        albedo=float(item["albedo"]),
                        roughness=float(item["roughness"]),
                    )
                    for item in data["spheres"]
                ),
                light=DirectionalLight(
                    direction=tuple(float(v) for v in data["light"]["direction"]),
                    ambient=float(data["light"]["ambient"]),
                ),
                camera=Camera(
                    eye=tuple(float(v) for v in data["camera"]["eye"]),
                    yaw=float(data["camera"]["yaw"]),
                    pitch=float(data["camera"]["pitch"]),
                    roll=float(data["camera"]["roll"]),
                    focal=float(data["camera"]["focal"]),
                ),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DiffRenderError(E_SCENE_INVALID, f"invalid serialized scene: {exc}") from exc


def default_scene() -> Scene:
    """A compact, asymmetric scene with useful occlusion and gradients."""

    return Scene(
        width=48,
        height=48,
        tile_size=8,
        spheres=(
            Sphere((-0.95, 0.22, 0.00), 0.62, 0.74, 0.25),
            Sphere((0.92, -0.22, -0.10), 0.48, 0.42, 0.70),
            Sphere((0.05, 0.78, -0.55), 0.34, 0.92, 0.12),
        ),
        light=DirectionalLight((0.35, 0.70, 0.62), 0.18),
        camera=Camera((0.0, 0.0, 5.2), yaw=0.04, pitch=-0.04, roll=0.01, focal=42.0),
    )


def target_scene_from(base: Scene) -> Scene:
    """Ground truth used by the inverse-imaging demo."""

    updates = (
        ("sphere.0.radius", base.value("sphere.0.radius") + 0.075),
        ("sphere.1.cx", base.value("sphere.1.cx") - 0.055),
        ("sphere.1.albedo", 0.63),
        ("sphere.2.radius", base.value("sphere.2.radius") + 0.035),
        ("light.dy", 0.83),
        ("light.dx", 0.31),
        ("camera.yaw", 0.075),
    )
    scene = base
    for key, value in updates:
        scene = scene.with_value(key, value)
    return scene
