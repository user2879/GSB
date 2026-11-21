"""Serializable data models shared by the loader, solver, and report writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import Failure
from .hashing import combine_fingerprints, sha256_json


@dataclass(frozen=True)
class Observation:
    keypoint_id: str
    view_index: int
    u: float
    v: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "keypoint_id": self.keypoint_id,
            "view_index": self.view_index,
            "u": self.u,
            "v": self.v,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Observation":
        return cls(
            keypoint_id=str(value["keypoint_id"]),
            view_index=int(value["view_index"]),
            u=float(value["u"]),
            v=float(value["v"]),
        )

    @property
    def pixel_fingerprint(self) -> str:
        return sha256_json([self.view_index, self.u, self.v])


def observations_fingerprint(observations: list[Observation]) -> str:
    ordered = sorted(observations, key=lambda obs: obs.view_index)
    return sha256_json([obs.to_dict() for obs in ordered])


@dataclass
class Camera:
    view_index: int
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_path: str
    camera_path: str
    image_sha256: str
    source_image_sha256: str

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack((self.R, self.t.reshape(3, 1)))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_index": self.view_index,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "image_path": self.image_path,
            "camera_path": self.camera_path,
            "image_sha256": self.image_sha256,
        }

    @property
    def fingerprint(self) -> str:
        # Paths are deliberately excluded: moving a sample directory must not
        # invalidate geometric caches.
        return sha256_json(
            {
                "view_index": self.view_index,
                "width": self.width,
                "height": self.height,
                "K": self.K.tolist(),
                "R": self.R.tolist(),
                "t": self.t.tolist(),
                "image_sha256": self.image_sha256,
            }
        )


@dataclass
class PointSource:
    """One observation-manifest line after load-stage validation.

    ``raw_observations`` contains the point's observations as read from JSONL,
    including malformed values, while ``observations`` contains only the usable
    subset in view-index order.
    """

    keypoint_id: str
    line_number: int
    base_name: str
    instance_suffix: str | None
    raw_observations: list[dict[str, Any]]
    observations: list[Observation]
    failures: list[Failure] = field(default_factory=list)
    injection_ids: list[str] = field(default_factory=list)
    injection_hits: dict[str, str] = field(default_factory=dict)
    load_error: dict[str, Any] | None = None

    @property
    def raw_fingerprint(self) -> str:
        return sha256_json(
            {
                "keypoint_id": self.keypoint_id,
                "raw_observations": self.raw_observations,
            }
        )

    @property
    def observations_fingerprint(self) -> str:
        return observations_fingerprint(self.observations)

    @property
    def view_indices(self) -> list[int]:
        return sorted(obs.view_index for obs in self.observations)

    def camera_fingerprint(self, cameras: dict[int, Camera]) -> str:
        return combine_fingerprints(
            cameras[view].fingerprint
            for view in self.view_indices
            if view in cameras
        )


@dataclass
class InputBundle:
    input_dir: Path
    cameras: dict[int, Camera]
    points: dict[str, PointSource]
    global_failures: list[Failure]
    input_versions: dict[str, Any]
    observations_sha256: str
    version_file_sha256: str
    config_sha256: str
    camera_hash: str
    image_resolution: tuple[int, int] | None
    injection_report: list[dict[str, Any]] = field(default_factory=list)
    grayscale_images: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def sorted_keypoints(self) -> list[str]:
        return sorted(self.points)

    @property
    def sorted_global_failures(self) -> list[dict[str, Any]]:
        def key(item: Failure) -> tuple[str, str, int, int]:
            return (
                item.keypoint_id or "",
                item.code,
                item.view_index if item.view_index is not None else -1,
                item.observation_index if item.observation_index is not None else -1,
            )

        return [item.to_dict() for item in sorted(self.global_failures, key=key)]
