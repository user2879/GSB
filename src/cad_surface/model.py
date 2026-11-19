"""Deterministic, local-only CAD sample model generation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .canonical import SCHEMA_VERSION, canonical_json_bytes, content_hash
from .errors import schema_error
from .nurbs import Material, NURBSPatch


SAMPLE_GENERATOR = "cad-surface.deterministic-sample.v1"


class DeterministicStream:
    """SHA-256 counter stream, independent of NumPy/Python RNG versions."""

    def __init__(self, seed: int, label: str) -> None:
        self._seed = int(seed)
        self._label = label.encode("utf-8")
        self._counter = 0
        self._buffer = b""

    def _bytes(self, count: int) -> bytes:
        while len(self._buffer) < count:
            digest = hashlib.sha256(
                self._label + b":" + str(self._seed).encode("ascii") + b":" + str(self._counter).encode("ascii")
            ).digest()
            self._buffer += digest
            self._counter += 1
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        integer = int.from_bytes(self._bytes(8), "little", signed=False)
        value = integer / float((1 << 64) - 1)
        return float(low + (high - low) * value)

    def signed(self, magnitude: float = 1.0) -> float:
        return self.uniform(-magnitude, magnitude)


@dataclass(frozen=True)
class Camera:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_y_degrees: float
    near: float
    far: float
    viewport: tuple[int, int]
    background: tuple[int, int, int]
    light_direction: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_degrees": self.fov_y_degrees,
            "near": self.near,
            "far": self.far,
            "viewport": list(self.viewport),
            "background": list(self.background),
            "light_direction": list(self.light_direction),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Camera":
        try:
            viewport = tuple(int(item) for item in value["viewport"])
            if len(viewport) != 2 or viewport[0] <= 0 or viewport[1] <= 0:
                raise ValueError("viewport must contain two positive integers")
            return cls(
                eye=tuple(float(item) for item in value["eye"]),  # type: ignore[arg-type]
                target=tuple(float(item) for item in value["target"]),  # type: ignore[arg-type]
                up=tuple(float(item) for item in value["up"]),  # type: ignore[arg-type]
                fov_y_degrees=float(value["fov_y_degrees"]),
                near=float(value["near"]),
                far=float(value["far"]),
                viewport=viewport,  # type: ignore[arg-type]
                background=tuple(int(item) for item in value["background"]),  # type: ignore[arg-type]
                light_direction=tuple(float(item) for item in value["light_direction"]),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error("invalid camera payload", error=str(exc)) from exc


class NURBSModel:
    def __init__(
        self,
        *,
        model_id: str,
        seed: int,
        generator: str,
        patches: Sequence[NURBSPatch],
        camera: Camera,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.schema_version = str(schema_version)
        self.model_id = str(model_id)
        self.seed = int(seed)
        self.generator = str(generator)
        self.patches = tuple(patches)
        self.camera = camera
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise schema_error(
                "unsupported schema version",
                expected=SCHEMA_VERSION,
                actual=self.schema_version,
            )
        if not self.patches:
            raise schema_error("model must contain at least one surface patch")
        keys = [patch.key for patch in self.patches]
        if len(keys) != len(set(keys)):
            raise schema_error("patch keys must be unique", keys=keys)
        if not all(patch.key for patch in self.patches):
            raise schema_error("patch key must not be empty")

    @property
    def patch_map(self) -> dict[str, NURBSPatch]:
        return {patch.key: patch for patch in self.patches}

    def patch(self, key: str) -> NURBSPatch:
        try:
            return self.patch_map[key]
        except KeyError as exc:
            raise schema_error("unknown patch key", patch_key=key) from exc

    def with_patch(self, replacement: NURBSPatch) -> "NURBSModel":
        found = False
        patches: list[NURBSPatch] = []
        for patch in self.patches:
            if patch.key == replacement.key:
                patches.append(replacement)
                found = True
            else:
                patches.append(patch)
        if not found:
            raise schema_error("replacement patch key is not in the model", patch_key=replacement.key)
        return NURBSModel(
            model_id=self.model_id,
            seed=self.seed,
            generator=self.generator,
            patches=patches,
            camera=self.camera,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "seed": self.seed,
            "generator": self.generator,
            "patches": [patch.to_dict() for patch in self.patches],
            "camera": self.camera.to_dict(),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NURBSModel":
        try:
            patches = [NURBSPatch.from_dict(item) for item in value["patches"]]
            return cls(
                model_id=str(value["model_id"]),
                seed=int(value["seed"]),
                generator=str(value["generator"]),
                patches=patches,
                camera=Camera.from_dict(value["camera"]),
                schema_version=str(value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise schema_error("invalid model payload", error=str(exc)) from exc


def clamped_uniform_knots(control_count: int, degree: int) -> list[float]:
    if degree < 1 or control_count <= degree:
        raise schema_error(
            "degree/control-count mismatch while constructing knots",
            degree=degree,
            control_count=control_count,
        )
    interior_count = control_count - degree - 1
    interior = [index / float(control_count - degree) for index in range(1, interior_count + 1)]
    return [0.0] * (degree + 1) + interior + [1.0] * (degree + 1)


@dataclass(frozen=True)
class PatchSpec:
    key: str
    degree_u: int
    degree_v: int
    control_u: int
    control_v: int
    origin: tuple[float, float]
    extent: tuple[float, float]
    base_height: float
    phase: float
    color: tuple[float, float, float]


PATCH_SPECS = (
    PatchSpec("p00", 3, 3, 6, 6, (-3.4, -2.2), (3.0, 4.2), 0.28, 0.15, (0.28, 0.62, 0.88)),
    PatchSpec("p01", 2, 3, 5, 6, (-0.1, -2.2), (3.0, 4.2), 0.48, 1.10, (0.92, 0.52, 0.24)),
    PatchSpec("p02", 2, 2, 5, 4, (-2.0, 2.5), (3.8, 2.2), 0.72, 2.25, (0.42, 0.82, 0.48)),
)


def _generate_patch(spec: PatchSpec, seed: int, index: int) -> NURBSPatch:
    stream = DeterministicStream(seed + 1009 * (index + 1), f"patch:{spec.key}")
    u = np.linspace(0.0, 1.0, spec.control_u)
    v = np.linspace(0.0, 1.0, spec.control_v)
    points = np.empty((spec.control_u, spec.control_v, 3), dtype="<f8")
    weights = np.empty((spec.control_u, spec.control_v), dtype="<f8")
    for i, uu in enumerate(u):
        for j, vv in enumerate(v):
            normalized_x = uu if spec.key != "p01" else 1.0 - uu
            x = spec.origin[0] + spec.extent[0] * uu
            y = spec.origin[1] + spec.extent[1] * vv
            radial = (uu - 0.5) ** 2 + (vv - 0.5) ** 2
            z = (
                spec.base_height
                + 0.32 * math.sin(2.0 * math.pi * normalized_x + spec.phase) * math.cos(math.pi * vv)
                + 0.18 * math.cos(3.0 * math.pi * vv - spec.phase)
                - 0.12 * radial
                + stream.signed(0.025)
            )
            taper = 0.10 * (uu * uu + vv * vv)
            points[i, j] = (x + stream.signed(0.018), y + stream.signed(0.018), z + taper)
            weights[i, j] = 0.90 + stream.uniform(0.0, 0.22)
    material = Material(
        name=f"material-{spec.key}",
        base_color=spec.color,
        ambient=0.18,
        diffuse=0.72,
        specular=0.35,
        shininess=32.0 + 4.0 * index,
        opacity=1.0,
    )
    return NURBSPatch(
        key=spec.key,
        degree_u=spec.degree_u,
        degree_v=spec.degree_v,
        knots_u=clamped_uniform_knots(spec.control_u, spec.degree_u),
        knots_v=clamped_uniform_knots(spec.control_v, spec.degree_v),
        control_points=points,
        weights=weights,
        material=material,
    )


def generate_model(seed: int = 20261119) -> NURBSModel:
    patches = [_generate_patch(spec, int(seed), index) for index, spec in enumerate(PATCH_SPECS)]
    camera = Camera(
        eye=(7.2, -8.4, 6.1),
        target=(0.0, 0.0, 0.48),
        up=(0.0, 0.0, 1.0),
        fov_y_degrees=42.0,
        near=0.1,
        far=80.0,
        viewport=(320, 240),
        background=(12, 16, 24),
        light_direction=(-0.35, 0.45, 0.82),
    )
    return NURBSModel(
        model_id=f"synthetic-cad-{int(seed)}",
        seed=int(seed),
        generator=SAMPLE_GENERATOR,
        patches=patches,
        camera=camera,
    )


def generate_model_twice(seed: int = 20261119) -> tuple[NURBSModel, bytes, bytes, str]:
    first = generate_model(seed)
    second = generate_model(seed)
    first_bytes = first.to_bytes()
    second_bytes = second.to_bytes()
    return first, first_bytes, second_bytes, first.content_hash()
