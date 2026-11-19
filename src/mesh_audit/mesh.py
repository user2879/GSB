from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fixedpoint import quantize_rows
from .schema import MeshInput


@dataclass(slots=True)
class MeshState:
    """A triangle mesh with authoritative fixed-point and reference float coordinates."""

    fixed: np.ndarray
    reference: np.ndarray
    faces: np.ndarray
    scale: int
    source_vertices: tuple[int, ...] = ()

    @classmethod
    def from_input(cls, mesh: MeshInput, fixed_scale: int) -> "MeshState":
        reference = np.asarray(mesh.vertices, dtype=np.float64)
        fixed = quantize_rows(mesh.vertices, fixed_scale)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return cls(fixed=fixed, reference=reference, faces=faces, scale=fixed_scale)

    @property
    def vertex_count(self) -> int:
        return int(self.fixed.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def copy(self) -> "MeshState":
        return MeshState(
            fixed=self.fixed.copy(),
            reference=self.reference.copy(),
            faces=self.faces.copy(),
            scale=self.scale,
            source_vertices=self.source_vertices,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "fixed": self.fixed.astype(np.int64).tolist(),
            "reference": self.reference.astype(np.float64).tolist(),
            "faces": self.faces.astype(np.int64).tolist(),
            "scale": self.scale,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MeshState":
        return cls(
            fixed=np.asarray(payload["fixed"], dtype=np.int64),
            reference=np.asarray(payload["reference"], dtype=np.float64),
            faces=np.asarray(payload["faces"], dtype=np.int64),
            scale=int(payload["scale"]),
        )
