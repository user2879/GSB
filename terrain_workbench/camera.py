"""Fixed camera track: keyframed roam trajectory, deterministic interpolation.

The track is loaded from YAML and sampled at integer frame indices. Linear
interpolation between keyframes (positions and targets) keeps everything
reproducible. Any keyframe or sampled frame leaving the valid domain raises
E_TRACK_DIVERGE with the first offending frame index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import yaml

from .errors import E_TRACK_DIVERGE, TerrainError


@dataclass(frozen=True)
class Camera:
    frame: int
    pos: tuple  # (x, y, z) world units
    target: tuple
    fov_y_deg: float
    viewport: tuple  # (w, h)

    @property
    def view_dir(self) -> np.ndarray:
        d = np.subtract(self.target, self.pos, dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([0.0, 0.0, -1.0])

    @property
    def projection_k(self) -> float:
        """Screen-space projection factor: pixels per world unit at distance 1."""
        return self.viewport[1] / (2.0 * math.tan(math.radians(self.fov_y_deg) / 2.0))


class CameraTrack:
    def __init__(self, keyframes, frames, viewport, fov_y_deg):
        if len(keyframes) < 2:
            raise TerrainError(E_TRACK_DIVERGE, "track needs at least 2 keyframes")
        self.keyframes = sorted(keyframes, key=lambda k: k["t"])
        self.frames = int(frames)
        self.viewport = tuple(int(v) for v in viewport)
        self.fov_y_deg = float(fov_y_deg)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            keyframes=cfg["keyframes"],
            frames=cfg["frames"],
            viewport=cfg["viewport"],
            fov_y_deg=cfg["fov_y_deg"],
        )

    def scaled(self, factor: float) -> "CameraTrack":
        """Return a copy with all keyframe coordinates scaled uniformly."""
        keyframes = []
        for kf in self.keyframes:
            keyframes.append({
                "t": kf["t"],
                "pos": [v * factor for v in kf["pos"]],
                "target": [v * factor for v in kf["target"]],
            })
        return CameraTrack(keyframes, self.frames, self.viewport, self.fov_y_deg)

    def sample(self, frame: int) -> Camera:
        kfs = self.keyframes
        t = frame * (kfs[-1]["t"] - kfs[0]["t"]) / max(self.frames - 1, 1) + kfs[0]["t"]
        hi = 1
        while hi < len(kfs) - 1 and kfs[hi]["t"] < t:
            hi += 1
        lo = hi - 1
        span = kfs[hi]["t"] - kfs[lo]["t"]
        alpha = 0.0 if span == 0 else (t - kfs[lo]["t"]) / span
        pos = [
            kfs[lo]["pos"][i] * (1.0 - alpha) + kfs[hi]["pos"][i] * alpha for i in range(3)
        ]
        target = [
            kfs[lo]["target"][i] * (1.0 - alpha) + kfs[hi]["target"][i] * alpha
            for i in range(3)
        ]
        return Camera(frame, tuple(pos), tuple(target), self.fov_y_deg, self.viewport)

    def validate(self, world_xy: float, z_max: float) -> None:
        """Raise E_TRACK_DIVERGE on the first frame leaving the valid domain."""
        for frame in range(self.frames):
            cam = self.sample(frame)
            for label, pt in (("pos", cam.pos), ("target", cam.target)):
                if not all(math.isfinite(v) for v in pt):
                    raise TerrainError(
                        E_TRACK_DIVERGE, f"non-finite camera {label}", frame=frame
                    )
                if not (0.0 <= pt[0] <= world_xy and 0.0 <= pt[1] <= world_xy):
                    raise TerrainError(
                        E_TRACK_DIVERGE,
                        f"camera {label} leaves terrain xy bounds",
                        frame=frame,
                        point=list(pt),
                    )
                if not (0.0 <= pt[2] <= z_max):
                    raise TerrainError(
                        E_TRACK_DIVERGE,
                        f"camera {label} leaves z bounds",
                        frame=frame,
                        point=list(pt),
                    )
