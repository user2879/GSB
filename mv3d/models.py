"""Project state, camera, and validated view models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ALLOWED_FIXED_BITS, ALLOWED_QUANT_LEVELS
from .fixed import determinant3, stable_hash


@dataclass(frozen=True)
class ImageSpec:
    width: int
    height: int
    mode: str

    def canonical(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "mode": self.mode}


@dataclass(frozen=True)
class Camera:
    intrinsic: tuple[tuple[int, int, int], ...]
    rotation: tuple[tuple[int, int, int], ...]
    rotation_scale: int
    translation: tuple[int, int, int]

    def canonical(self) -> dict[str, Any]:
        return {
            "intrinsic": [list(row) for row in self.intrinsic],
            "rotation": [list(row) for row in self.rotation],
            "rotation_scale": self.rotation_scale,
            "translation": list(self.translation),
        }

    def projection_numerator(self) -> tuple[tuple[int, ...], ...]:
        extrinsic = tuple(
            tuple(self.rotation[row][column] for column in range(3))
            + (self.translation[row] * self.rotation_scale,)
            for row in range(3)
        )
        return tuple(
            tuple(
                sum(self.intrinsic[row][k] * extrinsic[k][column] for k in range(3))
                for column in range(4)
            )
            for row in range(3)
        )


@dataclass(frozen=True)
class ReconstructionConfig:
    min_disparity: int
    voxel_size: int
    residual_threshold: int
    max_subdiv_depth: int
    max_voxels_per_parent: int
    world_bounds: tuple[int, int, int, int, int, int]
    global_point_budget: int
    view_point_quota: int
    workers: int
    bits: int
    quant_level: int
    strategy_override: str | None
    orthogonality_tolerance_ppm: int

    def canonical(self) -> dict[str, Any]:
        return {
            "min_disparity": self.min_disparity,
            "voxel_size": self.voxel_size,
            "residual_threshold": self.residual_threshold,
            "max_subdiv_depth": self.max_subdiv_depth,
            "max_voxels_per_parent": self.max_voxels_per_parent,
            "world_bounds": list(self.world_bounds),
            "global_point_budget": self.global_point_budget,
            "view_point_quota": self.view_point_quota,
            "workers": self.workers,
            "bits": self.bits,
            "quant_level": self.quant_level,
            "strategy_override": self.strategy_override,
            "orthogonality_tolerance_ppm": self.orthogonality_tolerance_ppm,
        }

    @property
    def scale(self) -> int:
        return 1 << self.bits

    @property
    def quant_step(self) -> int:
        return max(1, (self.scale // 256) * self.quant_level)


@dataclass
class ProjectState:
    data: dict[str, Any]
    base_dir: Path
    project_path: Path | None = None

    def clone(self) -> "ProjectState":
        return ProjectState(
            data=copy.deepcopy(self.data),
            base_dir=self.base_dir,
            project_path=self.project_path,
        )

    def image_spec(self) -> ImageSpec:
        raw = self.data.get("image", {})
        return ImageSpec(
            width=int(raw.get("width", 0)),
            height=int(raw.get("height", 0)),
            mode=str(raw.get("mode", "")),
        )

    def fixed_dict(self) -> dict[str, Any]:
        return self.data.get("fixed", {})

    def recon_dict(self) -> dict[str, Any]:
        return self.data.get("reconstruction", {})

    def config(self) -> ReconstructionConfig:
        fixed = self.fixed_dict()
        recon = self.recon_dict()
        bounds = tuple(int(value) for value in recon.get("world_bounds", (0, 0, 0, 0, 0, 0)))
        return ReconstructionConfig(
            min_disparity=int(recon.get("min_disparity", 0)),
            voxel_size=int(recon.get("voxel_size", 0)),
            residual_threshold=int(recon.get("residual_threshold", 1)),
            max_subdiv_depth=int(recon.get("max_subdiv_depth", 0)),
            max_voxels_per_parent=int(recon.get("max_voxels_per_parent", 8)),
            world_bounds=bounds,
            global_point_budget=int(recon.get("global_point_budget", 0)),
            view_point_quota=int(recon.get("view_point_quota", 0)),
            workers=int(recon.get("workers", 1)),
            bits=int(fixed.get("bits", 0)),
            quant_level=int(fixed.get("quant_level", 0)),
            strategy_override=recon.get("strategy_override"),
            orthogonality_tolerance_ppm=int(
                recon.get("orthogonality_tolerance_ppm", 10)
            ),
        )

    def canonical_views(self) -> list[dict[str, Any]]:
        return sorted(
            copy.deepcopy(self.data.get("views", [])),
            key=lambda item: int(item.get("id", 0)),
        )

    def canonical_pairs(self) -> list[list[int]]:
        pairs = [tuple(sorted((int(pair[0]), int(pair[1])))) for pair in self.data.get("pairs", [])]
        return [list(pair) for pair in sorted(pairs)]

    def config_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": self.data.get("schema_version"),
                "recipe_version": self.data.get("recipe_version"),
                "image": self.data.get("image"),
                "fixed": self.data.get("fixed"),
                "reconstruction": {
                    key: value
                    for key, value in self.data.get("reconstruction", {}).items()
                    if key != "workers"
                },
                "pairs": self.canonical_pairs(),
            }
        )

    def project_fingerprint(self) -> str:
        return stable_hash(
            {
                "config_hash": self.config_hash(),
                "views": self.canonical_views(),
            }
        )


@dataclass(frozen=True)
class ValidatedView:
    view_id: int
    image_path: Path
    image_bytes: bytes
    image_sha256: str
    camera: Camera
    input_key: str


def parse_camera(raw: dict[str, Any]) -> Camera:
    intrinsic = tuple(tuple(int(value) for value in row) for row in raw["intrinsic"])
    rotation = tuple(tuple(int(value) for value in row) for row in raw["rotation"])
    translation = tuple(int(value) for value in raw["translation"])
    return Camera(
        intrinsic=intrinsic,
        rotation=rotation,
        rotation_scale=int(raw["rotation_scale"]),
        translation=translation,
    )


def intrinsic_full_rank(camera: Camera, tolerance_ppm: int = 1) -> bool:
    determinant = determinant3(camera.intrinsic)
    if determinant == 0:
        return False
    scale = max(
        1,
        abs(camera.intrinsic[0][0])
        * abs(camera.intrinsic[1][1])
        * abs(camera.intrinsic[2][2]),
    )
    return abs(determinant) * 1_000_000 > scale * tolerance_ppm


def rotation_error(camera: Camera) -> int:
    rotation = camera.rotation
    scale = camera.rotation_scale
    expected = scale * scale
    maximum = 0
    for row in range(3):
        for column in range(3):
            value = sum(rotation[k][row] * rotation[k][column] for k in range(3))
            target = expected if row == column else 0
            maximum = max(maximum, abs(value - target))
    return maximum


def rotation_is_orthogonal(camera: Camera, tolerance_ppm: int) -> bool:
    expected = camera.rotation_scale * camera.rotation_scale
    tolerance = max(1, expected // 1_000_000 * tolerance_ppm)
    if tolerance_ppm == 0:
        tolerance = 0
    return rotation_error(camera) <= tolerance


def valid_config_enum(raw_bits: int, raw_quant: int) -> tuple[bool, bool]:
    return raw_bits in ALLOWED_FIXED_BITS, raw_quant in ALLOWED_QUANT_LEVELS
