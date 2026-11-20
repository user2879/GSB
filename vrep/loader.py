"""图像库与区域真值加载、校验层（业务要求 1）。

- 扫描目录下全部 PNG 与同名 ``*_truth.json``
- 校验 PNG 容器头、统一尺寸与 RGB 像素模式
- 校验真值图元矩形为整数且完全落在图内
- 图像名（去扩展名）唯一
任何单图/单真值失败都返回结构化失败记录，不中断其余图像。
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .errors import E_IMAGE_FORMAT, E_TRUTH_INVALID, Failure

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageRecord:
    name: str
    image_path: str
    truth_path: str
    width: int
    height: int
    rgb: np.ndarray
    regions: list


def _is_int(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return float(value).is_integer()
    return False


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def load_png(path: str):
    """返回 (RGB ndarray, Failure|None)。"""
    try:
        raw = _read_bytes(path)
    except OSError as exc:
        return None, Failure(
            image=os.path.splitext(os.path.basename(path))[0],
            stage="load",
            code=E_IMAGE_FORMAT,
            detail=f"无法读取文件: {exc}",
        )
    if len(raw) < 8 or raw[:8] != PNG_SIGNATURE:
        return None, Failure(
            image=os.path.splitext(os.path.basename(path))[0],
            stage="load",
            code=E_IMAGE_FORMAT,
            detail="PNG 容器头签名不正确或文件被截断",
        )
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # Pillow 抛出的各类解码异常
        return None, Failure(
            image=os.path.splitext(os.path.basename(path))[0],
            stage="load",
            code=E_IMAGE_FORMAT,
            detail=f"PNG 解码失败: {exc}",
        )
    if image.mode != "RGB":
        return None, Failure(
            image=os.path.splitext(os.path.basename(path))[0],
            stage="load",
            code=E_IMAGE_FORMAT,
            detail=f"像素模式必须为 RGB，实际为 {image.mode}",
        )
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, Failure(
            image=os.path.splitext(os.path.basename(path))[0],
            stage="load",
            code=E_IMAGE_FORMAT,
            detail="解码结果不是 HxWx3 的 RGB 数组",
        )
    return rgb, None


def validate_truth(data, width: int, height: int, image_name: str):
    """返回 Failure|None。矩形为 [x, y, w, h] 整数像素且完全落在图内。"""
    if not isinstance(data, dict) or "regions" not in data:
        return Failure(image_name, "truth", E_TRUTH_INVALID,
                       "真值 JSON 必须是包含 regions 字段的对象")
    regions = data["regions"]
    if not isinstance(regions, list):
        return Failure(image_name, "truth", E_TRUTH_INVALID,
                       "regions 必须为数组")
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            return Failure(image_name, "truth", E_TRUTH_INVALID,
                           f"第 {index} 个区域不是对象")
        category = region.get("category")
        if not isinstance(category, str) or not category:
            return Failure(image_name, "truth", E_TRUTH_INVALID,
                           f"第 {index} 个区域 category 非法")
        box = region.get("box")
        if not isinstance(box, list) or len(box) != 4:
            return Failure(image_name, "truth", E_TRUTH_INVALID,
                           f"区域 {category} 的 box 必须是长度 4 的数组")
        if not all(_is_int(value) for value in box):
            return Failure(image_name, "truth", E_TRUTH_INVALID,
                           f"区域 {category} 的矩形必须为整数像素")
        x, y, w, h = (int(value) for value in box)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            return Failure(
                image_name,
                "truth",
                E_TRUTH_INVALID,
                f"区域 {category} 矩形 [{x},{y},{w},{h}] 越界或非正",
            )
    return None


def scan_library(directory: str) -> tuple[list, list, int | None]:
    """扫描图像目录。

    返回 (ImageRecord 列表（按名称排序）, Failure 列表（稳定排序）, 统一尺寸)。
    尺寸不一致的图像进入失败清单，统一尺寸取首个有效图像。
    """
    png_paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".png")
    )
    records: list = []
    failures: list = []
    canonical_size: tuple[int, int] | None = None
    seen_names: set[str] = set()

    for image_path in png_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        if name in seen_names:
            failures.append(Failure(name, "load", E_IMAGE_FORMAT, "图像名重复"))
            continue
        seen_names.add(name)

        rgb, load_failure = load_png(image_path)
        if load_failure is not None:
            failures.append(load_failure)
            continue
        height, width = rgb.shape[0], rgb.shape[1]
        if canonical_size is None:
            canonical_size = (width, height)
        elif (width, height) != canonical_size:
            failures.append(
                Failure(
                    name,
                    "load",
                    E_IMAGE_FORMAT,
                    f"尺寸 {width}x{height} 与统一尺寸 "
                    f"{canonical_size[0]}x{canonical_size[1]} 不一致",
                )
            )
            continue

        truth_path = os.path.join(directory, f"{name}_truth.json")
        if not os.path.exists(truth_path):
            failures.append(Failure(name, "truth", E_TRUTH_INVALID,
                                    "缺少同名区域真值 JSON"))
            continue
        try:
            import json

            with open(truth_path, "r", encoding="utf-8") as handle:
                truth_data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(Failure(name, "truth", E_TRUTH_INVALID,
                                    f"真值 JSON 读取失败: {exc}"))
            continue
        truth_failure = validate_truth(truth_data, width, height, name)
        if truth_failure is not None:
            failures.append(truth_failure)
            continue

        records.append(
            ImageRecord(
                name=name,
                image_path=image_path,
                truth_path=truth_path,
                width=width,
                height=height,
                rgb=rgb,
                regions=truth_data["regions"],
            )
        )

    failures.sort(key=Failure.sort_key)
    records.sort(key=lambda record: record.name)
    size = canonical_size[0] if canonical_size else None
    return records, failures, size
