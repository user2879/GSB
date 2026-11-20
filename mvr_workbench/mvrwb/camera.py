"""相机内外参模型、视图扫描与输入合法性校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .errors import E_VIEW_INVALID, StageError
from .util import content_fingerprint, read_json

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Camera:
    view_id: int
    intrinsics: np.ndarray  # 3x3
    rotation: np.ndarray    # 3x3 世界 -> 相机
    translation: np.ndarray  # 3, 世界 -> 相机
    width: int
    height: int

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """世界点 -> 像素坐标、深度（相机系 z）、相机系坐标。"""
        cam = self.world_to_camera(points)
        depth = cam[:, 2]
        homogeneous = cam @ self.intrinsics.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixels, depth, cam

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "intrinsics": self.intrinsics.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "width": self.width,
            "height": self.height,
        }

    def fingerprint(self) -> str:
        payload = {
            "view_id": self.view_id,
            "intrinsics": self.intrinsics.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "width": self.width,
            "height": self.height,
        }
        return content_fingerprint(payload)


@dataclass
class ViewInput:
    view_id: int
    image_path: str
    camera_path: str
    camera: Camera
    image_fingerprint: str  # 原始 PNG 字节指纹
    camera_fingerprint: str
    gray: np.ndarray | None = None  # 延迟解码，灰度 uint8


def _check_matrix(name: str, matrix: np.ndarray, shape: tuple[int, ...]) -> None:
    if matrix.shape != shape:
        raise StageError(E_VIEW_INVALID, f"{name} 形状必须为 {shape}")
    if not np.isfinite(matrix).all():
        raise StageError(E_VIEW_INVALID, f"{name} 含非有限值")


def load_camera(camera_path: str, expected_view_id: int) -> Camera:
    if not os.path.exists(camera_path):
        raise StageError(
            E_VIEW_INVALID,
            f"视图 {expected_view_id} 相机参数文件缺失: {camera_path}",
            stage="load",
            view_id=expected_view_id,
        )
    raw = read_json(camera_path)
    try:
        view_id = int(raw["view_id"])
        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float64)
        rotation = np.asarray(raw["rotation"], dtype=np.float64)
        translation = np.asarray(raw["translation"], dtype=np.float64)
        width = int(raw["width"])
        height = int(raw["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError(
            E_VIEW_INVALID,
            f"视图 {expected_view_id} 相机参数格式错误: {exc}",
            stage="load",
            view_id=expected_view_id,
        ) from exc

    if view_id != expected_view_id:
        raise StageError(
            E_VIEW_INVALID,
            f"相机文件 view_id={view_id} 与视图序号 {expected_view_id} 不一致",
            stage="load",
            view_id=expected_view_id,
        )
    _check_matrix("intrinsics", intrinsics, (3, 3))
    _check_matrix("rotation", rotation, (3, 3))
    _check_matrix("translation", translation, (3,))

    if abs(np.linalg.det(intrinsics)) < 1e-12:
        raise StageError(E_VIEW_INVALID, "内参矩阵不可逆", stage="load", view_id=view_id)
    identity = rotation @ rotation.T
    if not np.allclose(identity, np.eye(3), atol=1e-8):
        raise StageError(E_VIEW_INVALID, "外参旋转部分不满足正交性", stage="load", view_id=view_id)
    if abs(np.linalg.det(rotation) - 1.0) > 1e-8:
        raise StageError(E_VIEW_INVALID, "外参旋转行列式不为 +1", stage="load", view_id=view_id)
    if width <= 0 or height <= 0:
        raise StageError(E_VIEW_INVALID, "相机宽高必须为正数", stage="load", view_id=view_id)
    return Camera(view_id, intrinsics, rotation, translation, width, height)


def load_gray_png(image_path: str, view_id: int) -> tuple[np.ndarray, bytes]:
    if not os.path.exists(image_path):
        raise StageError(
            E_VIEW_INVALID, f"视图 {view_id} 图像缺失: {image_path}",
            stage="load", view_id=view_id,
        )
    with open(image_path, "rb") as handle:
        raw_bytes = handle.read()
    if raw_bytes[:8] != PNG_MAGIC:
        raise StageError(
            E_VIEW_INVALID, f"视图 {view_id} 不是合法 PNG 容器头",
            stage="load", view_id=view_id,
        )
    try:
        with Image.open(image_path) as image:
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:  # Pillow 解码失败同样归类为视图非法
        raise StageError(
            E_VIEW_INVALID, f"视图 {view_id} PNG 解码失败: {exc}",
            stage="load", view_id=view_id,
        ) from exc
    if gray.ndim != 2:
        raise StageError(E_VIEW_INVALID, "灰度解码结果维度错误", stage="load", view_id=view_id)
    return gray, raw_bytes


def scan_views(input_dir: str) -> list[ViewInput]:
    """扫描 views/ 目录，校验全局约束并返回按视图序号稳定排序的视图清单。"""
    views_dir = os.path.join(input_dir, "views")
    if not os.path.isdir(views_dir):
        raise StageError(E_VIEW_INVALID, f"视图目录不存在: {views_dir}", stage="load")

    view_ids: list[int] = []
    entries: dict[int, tuple[str, str]] = {}
    for name in sorted(os.listdir(views_dir)):
        path = os.path.join(views_dir, name)
        if not os.path.isfile(path) or not name.startswith("view_"):
            continue
        if name.endswith(".png"):
            view_id = int(name[len("view_"):-len(".png")])
            image_path, camera_path = path, os.path.join(views_dir, f"view_{view_id:04d}.cam.json")
        elif name.endswith(".cam.json"):
            view_id = int(name[len("view_"):-len(".cam.json")])
            image_path, camera_path = os.path.join(views_dir, f"view_{view_id:04d}.png"), path
        else:
            continue
        if view_id in entries:
            existing = entries[view_id]
            entries[view_id] = (image_path or existing[0], camera_path or existing[1])
        else:
            entries[view_id] = (image_path, camera_path)
            view_ids.append(view_id)

    if len(view_ids) != len(set(view_ids)):
        raise StageError(E_VIEW_INVALID, "视图序号不唯一", stage="load")

    views: list[ViewInput] = []
    reference_shape: tuple[int, int] | None = None
    for view_id in sorted(entries):
        image_path, camera_path = entries[view_id]
        camera = load_camera(camera_path, view_id)
        gray, raw_bytes = load_gray_png(image_path, view_id)
        shape = (camera.height, camera.width)
        if gray.shape != shape:
            raise StageError(
                E_VIEW_INVALID,
                f"视图 {view_id} 图像分辨率 {gray.shape} 与相机参数 {shape} 不一致",
                stage="load", view_id=view_id,
            )
        if reference_shape is None:
            reference_shape = shape
        elif gray.shape != reference_shape:
            raise StageError(
                E_VIEW_INVALID,
                f"视图 {view_id} 分辨率 {gray.shape} 与基准 {reference_shape} 不一致",
                stage="load", view_id=view_id,
            )
        views.append(
            ViewInput(
                view_id=view_id,
                image_path=image_path,
                camera_path=camera_path,
                camera=camera,
                image_fingerprint=content_fingerprint(raw_bytes),
                camera_fingerprint=camera.fingerprint(),
                gray=gray,
            )
        )
    if not views:
        raise StageError(E_VIEW_INVALID, "未扫描到任何视图", stage="load")
    return views
