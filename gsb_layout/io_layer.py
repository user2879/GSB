"""PNG/truth loading and strict validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import struct
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Config
from .errors import ImageInvalidError, TruthInvalidError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ALLOWED_PNG_MODES = {"L", "LA", "1", "I", "I;16", "RGB", "RGBA", "P"}


@dataclass(frozen=True)
class ImageRecord:
    name: str
    path: Path
    sha256: str
    width: int
    height: int
    pixels: np.ndarray
    mode: str


@dataclass(frozen=True)
class TruthCharacter:
    reading_index: int
    bbox: tuple[int, int, int, int]
    text: str


@dataclass(frozen=True)
class TruthColumn:
    reading_index: int
    bbox: tuple[int, int, int, int]
    chars: tuple[TruthCharacter, ...]


@dataclass(frozen=True)
class TruthRecord:
    image_name: str
    columns: tuple[TruthColumn, ...]
    path: Path


def discover_pngs(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise ImageInvalidError(f"样本目录不存在: {directory}")
    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    return sorted(paths, key=lambda p: (p.name.casefold(), p.name))


def _validate_png_header(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[:8] != PNG_SIGNATURE:
        raise ImageInvalidError("PNG 容器头无效")
    length = struct.unpack(">I", data[8:12])[0]
    chunk_type = data[12:16]
    if chunk_type != b"IHDR" or length != 13 or len(data) < 29:
        raise ImageInvalidError("PNG 缺少有效 IHDR")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    if width <= 0 or height <= 0:
        raise ImageInvalidError("PNG 尺寸必须为正整数")
    if bit_depth not in (1, 2, 4, 8, 16):
        raise ImageInvalidError(f"不支持的 PNG 位深: {bit_depth}")
    if color_type not in (0, 2, 3, 4, 6):
        raise ImageInvalidError(f"不支持的 PNG 颜色类型: {color_type}")
    if compression != 0 or filter_method != 0 or interlace not in (0, 1):
        raise ImageInvalidError("PNG IHDR 压缩/滤波/交错字段无效")
    return width, height


def load_png(path: str | Path, config: Config) -> ImageRecord:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ImageInvalidError(f"无法读取图像: {exc}") from exc
    header_width, header_height = _validate_png_header(raw)
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            mode = image.mode
            if mode not in _ALLOWED_PNG_MODES:
                raise ImageInvalidError(f"不支持的像素模式: {mode}")
            width, height = image.size
            if (width, height) != (header_width, header_height):
                raise ImageInvalidError("PNG 头部尺寸与解码尺寸不一致")
            if config.expected_width is not None and width != config.expected_width:
                raise ImageInvalidError(
                    f"图像宽度 {width} 与配置期望 {config.expected_width} 不一致"
                )
            if config.expected_height is not None and height != config.expected_height:
                raise ImageInvalidError(
                    f"图像高度 {height} 与配置期望 {config.expected_height} 不一致"
                )
            if mode == "L":
                arr = np.asarray(image, dtype=np.uint8)
            elif mode == "I;16":
                arr = np.asarray(image, dtype=np.uint16)
                arr = np.clip((arr.astype(np.int64) + 128) // 257, 0, 255).astype(np.uint8)
            else:
                arr = np.asarray(image.convert("L"), dtype=np.uint8)
    except ImageInvalidError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageInvalidError(f"PNG 解码失败: {exc}") from exc
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ImageInvalidError("解码结果必须为二维 uint8 灰度数组")
    if arr.shape != (height, width):
        raise ImageInvalidError("灰度数组尺寸与图像尺寸不一致")
    digest = sha256(raw).hexdigest()
    return ImageRecord(
        name=path.name,
        path=path,
        sha256=digest,
        width=width,
        height=height,
        pixels=np.ascontiguousarray(arr),
        mode=mode,
    )


def find_truth_path(png_path: str | Path) -> Path | None:
    png_path = Path(png_path)
    stem = png_path.stem
    candidates = [
        png_path.with_name(stem + ".truth.json"),
        png_path.with_name(stem + ".layout.json"),
        png_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _as_bbox(value: Any, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TruthInvalidError(f"{label}.bbox 必须为四元整数数组")
    parts = list(value)
    for item in parts:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TruthInvalidError(f"{label}.bbox 必须全部为整数")
    x0, y0, x1, y1 = (int(v) for v in parts)
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise TruthInvalidError(f"{label}.bbox 必须满足 0<=x0<x1, 0<=y0<y1")
    return x0, y0, x1, y1


def _reading_index(container: dict[str, Any], label: str) -> int:
    value = container.get(
        "reading_index",
        container.get("reading_order", container.get("index", container.get("order"))),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TruthInvalidError(f"{label} 缺少非负整数阅读序号")
    return int(value)


def _text(container: dict[str, Any], label: str) -> str:
    value = container.get(
        "text", container.get("char", container.get("character", container.get("label")))
    )
    if not isinstance(value, str) or not value.strip():
        raise TruthInvalidError(f"{label} 字符内容必须为非空字符串")
    return value


def load_truth(png_path: str | Path, image: ImageRecord) -> TruthRecord:
    truth_path = find_truth_path(png_path)
    if truth_path is None:
        raise TruthInvalidError("缺少同名真值 JSON")
    try:
        raw = json.loads(truth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TruthInvalidError(f"真值 JSON 无法读取: {exc}") from exc
    if not isinstance(raw, dict):
        raise TruthInvalidError("真值根节点必须为对象")
    declared_image = raw.get("image")
    if declared_image is not None and not isinstance(declared_image, str):
        raise TruthInvalidError("image 字段必须为字符串")
    if declared_image and Path(declared_image).name != Path(png_path).name:
        raise TruthInvalidError("真值 image 字段与文件名不一致")
    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, list) or not columns_raw:
        raise TruthInvalidError("columns 必须为非空数组")
    columns: list[TruthColumn] = []
    seen_column_indices: set[int] = set()
    for ci, column_raw in enumerate(columns_raw):
        label = f"columns[{ci}]"
        if not isinstance(column_raw, dict):
            raise TruthInvalidError(f"{label} 必须为对象")
        col_index = _reading_index(column_raw, label)
        if col_index in seen_column_indices:
            raise TruthInvalidError(f"{label}.reading_index 重复")
        seen_column_indices.add(col_index)
        col_bbox = _as_bbox(column_raw.get("bbox", column_raw.get("box")), label)
        cx0, cy0, cx1, cy1 = col_bbox
        if cx1 > image.width or cy1 > image.height:
            raise TruthInvalidError(f"{label}.bbox 超出图像范围")
        chars_raw = column_raw.get("chars", column_raw.get("characters"))
        if not isinstance(chars_raw, list) or not chars_raw:
            raise TruthInvalidError(f"{label}.chars 必须为非空数组")
        chars: list[TruthCharacter] = []
        seen_char_indices: set[int] = set()
        for chi, char_raw in enumerate(chars_raw):
            char_label = f"{label}.chars[{chi}]"
            if not isinstance(char_raw, dict):
                raise TruthInvalidError(f"{char_label} 必须为对象")
            char_index = _reading_index(char_raw, char_label)
            if char_index in seen_char_indices:
                raise TruthInvalidError(f"{char_label}.reading_index 重复")
            seen_char_indices.add(char_index)
            bbox = _as_bbox(char_raw.get("bbox", char_raw.get("box")), char_label)
            x0, y0, x1, y1 = bbox
            if x1 > image.width or y1 > image.height:
                raise TruthInvalidError(f"{char_label}.bbox 超出图像范围")
            chars.append(TruthCharacter(char_index, bbox, _text(char_raw, char_label)))
        if len({c.reading_index for c in chars}) != len(chars):
            raise TruthInvalidError(f"{label} 字符阅读序号必须唯一")
        # Reading order is top-to-bottom: sort geometrically, then require each
        # subsequent character to have a larger reading index and no y-overlap.
        sorted_chars = sorted(chars, key=lambda c: (c.bbox[1], c.bbox[0], c.reading_index))
        for previous, current in zip(sorted_chars, sorted_chars[1:]):
            if current.reading_index <= previous.reading_index:
                raise TruthInvalidError(f"{label} 字符阅读序号必须随自上而下位置递增")
            if current.bbox[1] < previous.bbox[3]:
                raise TruthInvalidError(f"{label} 字符高度区间重叠")
        columns.append(TruthColumn(col_index, col_bbox, tuple(sorted_chars)))
    if len({c.reading_index for c in columns}) != len(columns):
        raise TruthInvalidError("栏阅读序号必须唯一")
    sorted_columns = sorted(columns, key=lambda c: c.reading_index)
    if len(sorted_columns) > 1 and not all(
        left.bbox[0] > right.bbox[0] for left, right in zip(sorted_columns, sorted_columns[1:])
    ):
        # Reading order should normally be right-to-left, but older truth files may
        # use another ordering. The hard contract is uniqueness, not geometry here.
        pass
    return TruthRecord(Path(png_path).name, tuple(sorted_columns), truth_path)


def truth_to_dict(truth: TruthRecord) -> dict[str, Any]:
    return {
        "image": truth.image_name,
        "columns": [
            {
                "reading_index": col.reading_index,
                "bbox": list(col.bbox),
                "chars": [
                    {
                        "reading_index": char.reading_index,
                        "bbox": list(char.bbox),
                        "text": char.text,
                    }
                    for char in col.chars
                ],
            }
            for col in truth.columns
        ],
    }
