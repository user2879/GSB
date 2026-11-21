"""视图图像与相机参数加载、校验层。"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .errors import BizCode, FailureRecord, WorkbenchError
from .utils import file_hash, read_json

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
VIEW_RE = re.compile(r"view_(\d+)\.png$")
CAMERA_RE = re.compile(r"view_(\d+)\.camera\.json$")
FOV_MIN_DEG = 1.0
FOV_MAX_DEG = 175.0


@dataclass
class Camera:
    view: int
    width: int
    height: int
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    fov_deg: float
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view": self.view,
            "width": self.width,
            "height": self.height,
            "intrinsic": np.asarray(self.intrinsic).tolist(),
            "rotation": np.asarray(self.rotation).tolist(),
            "translation": np.asarray(self.translation).tolist(),
            "fov_deg": self.fov_deg,
            "fingerprint": self.fingerprint,
        }


@dataclass
class ViewData:
    view: int
    image_path: str
    camera_path: str
    gray: np.ndarray
    color: np.ndarray
    camera: Camera
    image_fingerprint: str
    camera_fingerprint: str

    def content_fingerprint(self) -> str:
        return f"{self.image_fingerprint}:{self.camera_fingerprint}"


@dataclass
class LoadResult:
    views: List[ViewData]
    failures: List[FailureRecord] = field(default_factory=list)
    expected_resolution: Optional[Tuple[int, int]] = None


def _check_png_header(path: str) -> None:
    if not os.path.exists(path):
        raise WorkbenchError(
            f"图像文件不存在: {path}", code=BizCode.E_VIEW_INVALID
        )
    with open(path, "rb") as handle:
        header = handle.read(8)
    if header != PNG_MAGIC:
        raise WorkbenchError(
            "PNG 容器头非法", code=BizCode.E_PNG_TRUNCATED,
            detail={"header": header.hex()},
        )


def _decode_image(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    # Pillow 对截断 PNG 抛 OSError/UnidentifiedImageError，统一映射为业务码
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            color = np.asarray(rgb, dtype=np.uint8)
            gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise WorkbenchError(
            f"PNG 解码失败: {exc}", code=BizCode.E_PNG_TRUNCATED
        ) from exc
    height, width = gray.shape
    return gray, color, (width, height)


def _load_camera(path: str, view: int) -> Camera:
    if not os.path.exists(path):
        raise WorkbenchError(
            f"相机参数文件缺失: {os.path.basename(path)}",
            view=view, stage="load", code=BizCode.E_CAMERA_MISSING,
        )
    data = read_json(path)
    required = {"intrinsic", "rotation", "translation", "fov_deg", "width", "height"}
    missing = required - set(data)
    if missing:
        raise WorkbenchError(
            f"相机参数缺字段: {sorted(missing)}",
            view=view, stage="load", code=BizCode.E_CAMERA_MISSING,
        )
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise WorkbenchError(
            "内参矩阵形状非 3x3", view=view, stage="load",
            code=BizCode.E_INTRINSIC_SINGULAR,
        )
    if not np.isfinite(intrinsic).all():
        raise WorkbenchError(
            "内参矩阵含非有限值", view=view, stage="load",
            code=BizCode.E_INTRINSIC_SINGULAR,
        )
    determinant = float(np.linalg.det(intrinsic))
    if abs(determinant) < 1e-12:
        raise WorkbenchError(
            f"内参矩阵不可逆，det={determinant:.3e}", view=view,
            stage="load", code=BizCode.E_INTRINSIC_SINGULAR,
        )
    fov_deg = float(data["fov_deg"])
    if not (FOV_MIN_DEG <= fov_deg <= FOV_MAX_DEG):
        raise WorkbenchError(
            f"视场角越界: {fov_deg:.2f} 度", view=view, stage="load",
            code=BizCode.E_FOV_ILLEGAL,
            detail={"allowed": [FOV_MIN_DEG, FOV_MAX_DEG]},
        )
    rotation = np.asarray(data["rotation"], dtype=np.float64)
    translation = np.asarray(data["translation"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise WorkbenchError(
            "外参形状非法", view=view, stage="load",
            code=BizCode.E_VIEW_INVALID,
        )
    if int(data.get("view", view)) != view:
        raise WorkbenchError(
            f"视图序号不一致: 文件 {data.get('view')} vs 命名 {view}",
            view=view, stage="load", code=BizCode.E_VIEW_DUPLICATE,
        )
    fingerprint = file_hash(path)
    return Camera(
        view=view, width=int(data["width"]), height=int(data["height"]),
        intrinsic=intrinsic, rotation=rotation, translation=translation,
        fov_deg=fov_deg, fingerprint=fingerprint,
    )


def load_sample(sample_dir: str) -> LoadResult:
    """扫描样本目录，逐视图校验加载；单视图失败隔离、其余继续。"""
    image_paths = sorted(glob.glob(os.path.join(sample_dir, "view_*.png")))
    if not image_paths:
        raise WorkbenchError(
            f"目录下未发现视图 PNG: {sample_dir}",
            stage="load", code=BizCode.E_VIEW_INVALID,
        )
    result = LoadResult(views=[])
    seen_views: set[int] = set()
    expected: Optional[Tuple[int, int]] = None

    for image_path in image_paths:
        match = VIEW_RE.search(os.path.basename(image_path))
        if match is None:
            continue
        view = int(match.group(1))
        stage = "load"
        camera_path = os.path.join(
            sample_dir, f"view_{view:02d}.camera.json"
        )
        try:
            if view in seen_views:
                raise WorkbenchError(
                    f"视图序号重复: {view}", view=view, stage=stage,
                    code=BizCode.E_VIEW_DUPLICATE,
                )
            _check_png_header(image_path)
            gray, color, resolution = _decode_image(image_path)
            camera = _load_camera(camera_path, view)
            if (camera.width, camera.height) != resolution:
                raise WorkbenchError(
                    f"相机分辨率 {camera.width}x{camera.height} 与图像 "
                    f"{resolution[0]}x{resolution[1]} 不一致",
                    view=view, stage=stage,
                    code=BizCode.E_RESOLUTION_MISMATCH,
                )
            if expected is None:
                expected = resolution
            elif resolution != expected:
                raise WorkbenchError(
                    f"分辨率 {resolution} 与基准 {expected} 不一致",
                    view=view, stage=stage,
                    code=BizCode.E_RESOLUTION_MISMATCH,
                )
            seen_views.add(view)
            image_fingerprint = file_hash(image_path)
            result.views.append(
                ViewData(
                    view=view, image_path=image_path, camera_path=camera_path,
                    gray=gray, color=color, camera=camera,
                    image_fingerprint=image_fingerprint,
                    camera_fingerprint=camera.fingerprint,
                )
            )
        except WorkbenchError as exc:
            exc.view = view
            exc.stage = stage
            result.failures.append(exc.to_record())

    result.views.sort(key=lambda item: item.view)
    result.expected_resolution = expected
    return result


def input_fingerprint(views: List[ViewData]) -> str:
    """全部输入（图像字节 + 相机字节）的聚合指纹。"""
    parts = [v.content_fingerprint() for v in sorted(views, key=lambda v: v.view)]
    joined = "|".join(parts).encode("utf-8")
    import hashlib

    return hashlib.sha256(joined).hexdigest()[:16]
