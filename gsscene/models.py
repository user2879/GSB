"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    source: str
    view_index: int | None = None
    level: int | None = None
    injection_id: str | None = None
    severity: str = "ERROR"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "injection_id": self.injection_id,
            "level": self.level,
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
            "view_index": self.view_index,
        }

    def sort_key(self) -> tuple[Any, ...]:
        source_rank = {
            "injection": 0,
            "image": 1,
            "depth": 2,
            "camera": 3,
            "view": 4,
            "config": 5,
            "edit": 6,
            "scheduler": 7,
            "cache": 8,
            "snapshot": 9,
            "asset": 10,
        }.get(self.source, 20)
        return (
            source_rank,
            -1 if self.view_index is None else self.view_index,
            -1 if self.level is None else self.level,
            self.code,
            "" if self.injection_id is None else self.injection_id,
            self.message,
        )


@dataclass
class Camera:
    view_index: int
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "intrinsic": np.asarray(self.intrinsic, dtype=np.float64).tolist(),
            "rotation": np.asarray(self.rotation, dtype=np.float64).tolist(),
            "translation": np.asarray(self.translation, dtype=np.float64).tolist(),
            "view_index": int(self.view_index),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Camera":
        return cls(
            view_index=int(value["view_index"]),
            intrinsic=np.asarray(value["intrinsic"], dtype=np.float64),
            rotation=np.asarray(value["rotation"], dtype=np.float64),
            translation=np.asarray(value["translation"], dtype=np.float64),
        )

    def fingerprint(self) -> str:
        from .util import hash_value

        return hash_value(self.as_dict())


@dataclass
class ViewRecord:
    view_index: int
    image_path: Path
    depth_path: Path
    camera: Camera
    image_fingerprint: str
    depth_fingerprint: str
    image_size: tuple[int, int]
    injection_id: str | None = None

    @property
    def camera_fingerprint(self) -> str:
        return self.camera.fingerprint()

    @property
    def content_fingerprint(self) -> str:
        from .util import hash_value

        return hash_value(
            {
                "image": self.image_fingerprint,
                "depth": self.depth_fingerprint,
                "size": list(self.image_size),
            }
        )


@dataclass
class StoredView:
    view_index: int
    image_path: str
    depth_path: str
    camera: dict[str, Any]
    image_fingerprint: str
    depth_fingerprint: str
    image_size: list[int]
    injection_id: str | None = None

    @classmethod
    def from_view_record(cls, view: ViewRecord, image_path: str, depth_path: str) -> "StoredView":
        return cls(
            view_index=view.view_index,
            image_path=image_path,
            depth_path=depth_path,
            camera=view.camera.as_dict(),
            image_fingerprint=view.image_fingerprint,
            depth_fingerprint=view.depth_fingerprint,
            image_size=list(view.image_size),
            injection_id=view.injection_id,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StoredView":
        return cls(
            view_index=int(value["view_index"]),
            image_path=str(value["image_path"]),
            depth_path=str(value["depth_path"]),
            camera=dict(value["camera"]),
            image_fingerprint=str(value["image_fingerprint"]),
            depth_fingerprint=str(value["depth_fingerprint"]),
            image_size=[int(value["image_size"][0]), int(value["image_size"][1])],
            injection_id=value.get("injection_id"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "depth_fingerprint": self.depth_fingerprint,
            "depth_path": self.depth_path,
            "image_fingerprint": self.image_fingerprint,
            "image_path": self.image_path,
            "image_size": self.image_size,
            "injection_id": self.injection_id,
            "view_index": self.view_index,
        }

    def camera_model(self) -> Camera:
        return Camera.from_dict(self.camera)

    def content_fingerprint(self) -> str:
        from .util import hash_value

        return hash_value(
            {
                "image": self.image_fingerprint,
                "depth": self.depth_fingerprint,
                "size": self.image_size,
            }
        )


@dataclass
class KernelSpec:
    view_index: int
    level: int
    row: int
    col: int
    kernel_index: int
    cache_key: str


@dataclass
class KernelGroup:
    view_index: int
    level: int
    grid_size: tuple[int, int]
    cache_key: str
    kernel_keys: list[str]
    kernel_content_fingerprints: list[str]
    arrays: dict[str, np.ndarray]
    level_fingerprint: str
    view_content_fingerprint: str
    camera_fingerprint: str
    projection_matrix_fingerprint: str


@dataclass
class TileRecord:
    view_index: int
    level: int
    row: int
    col: int
    cache_key: str
    kernel_refs: list[int]
    arrays: dict[str, np.ndarray]


@dataclass
class CompositeRecord:
    view_index: int
    level: int
    cache_key: str
    tile_refs: list[tuple[int, int, int, int]]
    lower_tile_refs: list[tuple[int, int, int, int]]
    lower_composite_key: str | None
    arrays: dict[str, np.ndarray]


@dataclass
class LoadedProject:
    sample_dir: Path
    config: Any
    views: dict[int, ViewRecord]
    edits: list[dict[str, Any]]
    injections: list[dict[str, Any]]
    failures: list[Failure] = field(default_factory=list)
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    excluded_views: set[int] = field(default_factory=set)
    excluded_levels: set[tuple[int, int]] = field(default_factory=set)
    image_files: dict[int, Path] = field(default_factory=dict)
