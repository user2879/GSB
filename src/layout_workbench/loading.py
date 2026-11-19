from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .canonical import read_json, require_int
from .config import RecognitionConfig

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageSource:
    name: str
    png_path: Path
    truth_path: Path


@dataclass
class LoadedImage:
    name: str
    source: ImageSource
    gray: np.ndarray
    content_fingerprint: str
    width: int
    height: int
    mode: str
    truth: dict[str, Any]


class LoadError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def scan_image_sources(data_dir: Path) -> list[ImageSource]:
    sources: list[ImageSource] = []
    for path in sorted(data_dir.glob("*.png"), key=lambda p: p.name):
        if not path.is_file():
            continue
        truth = path.with_suffix(".truth.json")
        if not truth.exists():
            fallback = path.with_suffix(".json")
            truth = fallback if fallback.exists() else truth
        sources.append(ImageSource(path.stem, path, truth))
    return sources


def _inspect_png_header(raw: bytes, path: Path) -> tuple[int, int]:
    if len(raw) < 33 or raw[:8] != PNG_SIGNATURE:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: invalid PNG signature")
    length, kind = struct.unpack(">I4s", raw[8:16])
    if length != 13 or kind != b"IHDR":
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: missing IHDR chunk")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", raw[16:29]
    )
    if width <= 0 or height <= 0:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: invalid image dimensions")
    if bit_depth not in {1, 2, 4, 8, 16}:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: unsupported PNG bit depth")
    if color_type not in {0, 2, 4, 6}:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: unsupported PNG color type")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: unsupported PNG header flags")
    return width, height


def _decode_png(path: Path, config: RecognitionConfig) -> tuple[np.ndarray, str, int, int, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: cannot read file: {exc}") from exc
    header_width, header_height = _inspect_png_header(raw, path)
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG":
                raise LoadError("E_IMAGE_INVALID", f"{path.name}: decoder did not identify PNG")
            mode = opened.mode
            if mode not in config.allowed_image_modes:
                raise LoadError(
                    "E_IMAGE_INVALID",
                    f"{path.name}: mode {mode!r} not in {list(config.allowed_image_modes)}",
                )
            if opened.size != (header_width, header_height):
                raise LoadError("E_IMAGE_INVALID", f"{path.name}: IHDR/decoded size mismatch")
            if opened.size != (config.image_width, config.image_height):
                raise LoadError(
                    "E_IMAGE_INVALID",
                    f"{path.name}: expected {config.image_width}x{config.image_height}, got {opened.size[0]}x{opened.size[1]}",
                )
            gray_image = opened.convert("L")
            gray = np.asarray(gray_image, dtype=np.uint8).copy()
    except LoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: PNG decode failed: {exc}") from exc
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise LoadError("E_IMAGE_INVALID", f"{path.name}: decoded data is not uint8 grayscale")
    fingerprint = hashlib.sha256(
        b"L" + struct.pack(">II", gray.shape[1], gray.shape[0]) + gray.tobytes(order="C")
    ).hexdigest()
    return gray, fingerprint, int(gray.shape[1]), int(gray.shape[0]), mode


def _bbox(value: Any, label: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise LoadError("E_TRUTH_INVALID", f"{label} must be an integer bbox [x0,y0,x1,y1]")
    coords = [require_int(v, f"{label}[{i}]") for i, v in enumerate(value)]
    x0, y0, x1, y1 = coords
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise LoadError("E_TRUTH_INVALID", f"{label} is empty or inverted")
    return x0, y0, x1, y1


def _validate_truth(value: Any, width: int, height: int, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LoadError("E_TRUTH_INVALID", f"{name}: truth root must be an object")
    size = value.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise LoadError("E_TRUTH_INVALID", f"{name}: image_size must be [w,h]")
    tw = require_int(size[0], "image_size[0]")
    th = require_int(size[1], "image_size[1]")
    if (tw, th) != (width, height):
        raise LoadError(
            "E_TRUTH_INVALID", f"{name}: truth size {tw}x{th} != image size {width}x{height}"
        )
    lines = value.get("lines")
    if not isinstance(lines, list):
        raise LoadError("E_TRUTH_INVALID", f"{name}: lines must be a list")
    normalized_lines: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            raise LoadError("E_TRUTH_INVALID", f"{name}: line {line_index} must be an object")
        lx0, ly0, lx1, ly1 = _bbox(line.get("bbox"), f"{name}.lines[{line_index}].bbox")
        if lx1 > width or ly1 > height:
            raise LoadError("E_TRUTH_INVALID", f"{name}: line {line_index} bbox outside image")
        chars = line.get("chars")
        if not isinstance(chars, list):
            raise LoadError("E_TRUTH_INVALID", f"{name}: line {line_index} chars must be a list")
        normalized_chars: list[dict[str, Any]] = []
        previous_x0: int | None = None
        previous_x1: int | None = None
        for char_index, char in enumerate(chars):
            if not isinstance(char, dict):
                raise LoadError(
                    "E_TRUTH_INVALID", f"{name}: line {line_index} char {char_index} must be an object"
                )
            cx0, cy0, cx1, cy1 = _bbox(
                char.get("bbox"), f"{name}.lines[{line_index}].chars[{char_index}].bbox"
            )
            if cx1 > width or cy1 > height:
                raise LoadError(
                    "E_TRUTH_INVALID",
                    f"{name}: line {line_index} char {char_index} bbox outside image",
                )
            if cx0 < lx0 or cy0 < ly0 or cx1 > lx1 or cy1 > ly1:
                raise LoadError(
                    "E_TRUTH_INVALID",
                    f"{name}: line {line_index} char {char_index} bbox outside line bbox",
                )
            text = char.get("text")
            if not isinstance(text, str) or len(text) != 1:
                raise LoadError(
                    "E_TRUTH_INVALID",
                    f"{name}: line {line_index} char {char_index} text must be one character",
                )
            if previous_x0 is not None and cx0 <= previous_x0:
                raise LoadError(
                    "E_TRUTH_INVALID",
                    f"{name}: line {line_index} chars are not strictly increasing by x0",
                )
            if previous_x1 is not None and cx0 < previous_x1:
                raise LoadError(
                    "E_TRUTH_INVALID",
                    f"{name}: line {line_index} chars overlap horizontally",
                )
            normalized_chars.append(
                {"bbox": [cx0, cy0, cx1, cy1], "text": text}
            )
            previous_x0 = cx0
            previous_x1 = cx1
        normalized_lines.append({"bbox": [lx0, ly0, lx1, ly1], "chars": normalized_chars})
    return {"image_size": [width, height], "lines": normalized_lines}


def load_image_source(source: ImageSource, config: RecognitionConfig) -> LoadedImage:
    gray, fingerprint, width, height, mode = _decode_png(source.png_path, config)
    try:
        truth_raw = read_json(source.truth_path)
    except FileNotFoundError as exc:
        raise LoadError("E_TRUTH_INVALID", f"{source.name}: truth file is missing") from exc
    except (OSError, ValueError) as exc:
        raise LoadError("E_TRUTH_INVALID", f"{source.name}: truth JSON is invalid: {exc}") from exc
    try:
        truth = _validate_truth(truth_raw, width, height, source.name)
    except LoadError:
        raise
    except ValueError as exc:
        raise LoadError("E_TRUTH_INVALID", f"{source.name}: {exc}") from exc
    return LoadedImage(
        name=source.name,
        source=source,
        gray=gray,
        content_fingerprint=fingerprint,
        width=width,
        height=height,
        mode=mode,
        truth=truth,
    )
