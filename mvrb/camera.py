"""相机内参/外参、投影工具与相机参数指纹。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .utils import fingerprint, read_json, write_json


@dataclass(frozen=True)
class Camera:
    view: int
    fx: float
    fy: float
    cx: float
    cy: float
    # 外参：世界 -> 相机 的旋转(3x3) 与平移
    rot: tuple
    trans: tuple

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(
            view=int(d["view"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            rot=tuple(tuple(float(v) for v in row) for row in d["rot"]),
            trans=tuple(float(v) for v in d["trans"]),
        )

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "rot": [list(row) for row in self.rot],
            "trans": list(self.trans),
        }

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, point_world) -> tuple[np.ndarray, float]:
        """世界点 -> (像素坐标 [u, v], 相机系深度 z)。"""
        r = np.array(self.rot, dtype=np.float64)
        t = np.array(self.trans, dtype=np.float64)
        p_cam = r @ np.asarray(point_world, dtype=np.float64) + t
        z = float(p_cam[2])
        p = self.matrix() @ p_cam
        return np.array([p[0] / p[2], p[1] / p[2]]), z


def camera_fingerprint(cameras: dict[int, Camera]) -> str:
    payload = {view: cam.to_dict() for view, cam in sorted(cameras.items())}
    return fingerprint(payload)


def save_cameras(path: str, cameras: dict[int, Camera]) -> None:
    write_json(path, {str(v): c.to_dict() for v, c in sorted(cameras.items())})


def load_cameras(path: str) -> dict[int, Camera]:
    raw = read_json(path)
    return {int(k): Camera.from_dict(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[tuple, tuple]:
    """构造世界->相机外参（OpenGL 风格 look-at，z 轴朝场景前方）。"""
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # 相机系：x=right, y=up, z=forward（朝场景），保证深度为正
    rot = np.vstack([right, true_up, forward])
    trans = -rot @ eye
    return tuple(tuple(float(v) for v in row) for row in rot), tuple(float(v) for v in trans)


def rotation_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
