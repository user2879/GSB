"""Immutable, versioned point-cloud / reconstruction snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hashing import digest_array


@dataclass(frozen=True)
class Snapshot:
    version: int
    points: np.ndarray
    normals: np.ndarray
    params: object
    tiles: dict
    merged: dict
    tile_hashes: dict
    op_history: tuple
    edit_audit: tuple

    @property
    def points_hash(self) -> str:
        return digest_array(self.points)

    @property
    def normals_hash(self) -> str:
        return digest_array(self.normals)

    @property
    def mesh_hash(self) -> str:
        from .hashing import stable_stream_hash

        ordered = [self.tile_hashes[k] for k in sorted(self.tile_hashes)]
        return stable_stream_hash(ordered)

    def content_hash(self) -> str:
        from .hashing import stable_stream_hash

        return stable_stream_hash(
            [
                self.version,
                self.points_hash,
                self.normals_hash,
                self.params.to_dict(),
                self.mesh_hash,
                list(self.op_history),
            ]
        )
