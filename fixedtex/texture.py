"""PNG container validation and deterministic decoding."""

from __future__ import annotations

import hashlib
import io
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import BizError, issue
from .hashing import texture_key
from .models import TextureData

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_TEXTURE_BYTES = 32 * 1024 * 1024
MAX_TEXTURE_DIMENSION = 4096
ALLOWED_COLOR_TYPES = {0: "L", 2: "RGB", 6: "RGBA"}


def _raise(code: str, message: str, path: str | Path) -> None:
    raise BizError(code=code, message=message, path=str(path), stage="texture")


def inspect_png_container(path: str | Path) -> dict[str, Any]:
    """Validate PNG signature, IHDR/IEND structure, chunk CRCs and pixel mode."""

    png_path = Path(path)
    try:
        stat = png_path.stat()
    except OSError as exc:
        _raise("E_IO", f"cannot stat PNG: {exc}", png_path)
    if stat.st_size > MAX_TEXTURE_BYTES:
        _raise("E_PNG_HEADER", "PNG exceeds local sample size limit", png_path)
    try:
        data = png_path.read_bytes()
    except OSError as exc:
        _raise("E_IO", f"cannot read PNG: {exc}", png_path)
    if not data.startswith(PNG_SIGNATURE):
        _raise("E_PNG_HEADER", "invalid PNG signature", png_path)

    offset = len(PNG_SIGNATURE)
    first = True
    saw_iend = False
    ihdr: dict[str, Any] | None = None
    while offset < len(data):
        if offset + 12 > len(data):
            _raise("E_PNG_HEADER", "truncated PNG chunk", png_path)
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            _raise("E_PNG_HEADER", "truncated PNG chunk payload", png_path)
        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack(">I", data[payload_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            _raise("E_PNG_HEADER", f"CRC mismatch in {chunk_type!r}", png_path)
        if first and chunk_type != b"IHDR":
            _raise("E_PNG_HEADER", "IHDR is not the first PNG chunk", png_path)
        if chunk_type == b"IHDR":
            if not first or length != 13:
                _raise("E_PNG_HEADER", "invalid IHDR chunk", png_path)
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if width <= 0 or height <= 0:
                _raise("E_PNG_RESOLUTION", "PNG dimensions must be positive", png_path)
            if width > MAX_TEXTURE_DIMENSION or height > MAX_TEXTURE_DIMENSION:
                _raise("E_PNG_RESOLUTION", "PNG dimensions exceed local limit", png_path)
            if bit_depth != 8:
                _raise("E_PNG_MODE", "only 8-bit PNG pixels are supported", png_path)
            if color_type not in ALLOWED_COLOR_TYPES:
                _raise("E_PNG_MODE", f"unsupported PNG color type {color_type}", png_path)
            if compression != 0 or filter_method != 0 or interlace != 0:
                _raise("E_PNG_HEADER", "unsupported PNG compression/filter/interlace", png_path)
            ihdr = {
                "width": width,
                "height": height,
                "bit_depth": bit_depth,
                "color_type": color_type,
                "interlace": interlace,
            }
        if chunk_type == b"IEND":
            if length != 0:
                _raise("E_PNG_HEADER", "invalid IEND chunk", png_path)
            saw_iend = True
            offset = crc_end
            break
        first = False
        offset = crc_end
    if ihdr is None:
        _raise("E_PNG_HEADER", "missing IHDR", png_path)
    if not saw_iend:
        _raise("E_PNG_HEADER", "missing IEND", png_path)
    ihdr["file_sha256"] = hashlib.sha256(data).hexdigest()
    return ihdr


def load_texture(
    path: str | Path,
    expected_resolution: int | tuple[int, int] | None = None,
) -> tuple[TextureData | None, list[dict[str, Any]]]:
    """Validate and decode a PNG into an RGB uint8 array.

    Returns ``(texture, issues)``. Rendering is allowed to continue when this
    texture fails so that other valid inputs can still be inspected.
    """

    png_path = Path(path)
    issues: list[dict[str, Any]] = []
    try:
        header = inspect_png_container(png_path)
    except BizError as exc:
        return None, [exc.to_dict()]

    if expected_resolution is not None:
        if isinstance(expected_resolution, int):
            expected = (expected_resolution, expected_resolution)
        else:
            expected = (int(expected_resolution[0]), int(expected_resolution[1]))
        if (header["width"], header["height"]) != expected:
            issues.append(
                issue(
                    "E_PNG_RESOLUTION",
                    f"PNG resolution {header['width']}x{header['height']} does not match {expected[0]}x{expected[1]}",
                    path=str(png_path),
                    stage="texture",
                    expected=list(expected),
                    actual=[header["width"], header["height"]],
                )
            )
            return None, issues

    expected_mode = ALLOWED_COLOR_TYPES[header["color_type"]]
    try:
        with Image.open(io.BytesIO(png_path.read_bytes())) as image:
            image.load()
            if image.width != header["width"] or image.height != header["height"]:
                issues.append(
                    issue(
                        "E_PNG_HEADER",
                        "PIL dimensions disagree with IHDR",
                        path=str(png_path),
                        stage="texture",
                    )
                )
                return None, issues
            if image.mode != expected_mode:
                issues.append(
                    issue(
                        "E_PNG_MODE",
                        f"PIL decoded mode {image.mode} does not match PNG color type {header['color_type']}",
                        path=str(png_path),
                        stage="texture",
                    )
                )
                return None, issues
            if image.mode == "RGB":
                rgb = np.asarray(image, dtype=np.uint8).copy()
            elif image.mode == "RGBA":
                rgb = np.asarray(image, dtype=np.uint8)[:, :, :3].copy()
            elif image.mode == "L":
                gray = np.asarray(image, dtype=np.uint8)
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
            else:
                issues.append(
                    issue(
                        "E_PNG_MODE",
                        f"unsupported decoded pixel mode {image.mode}",
                        path=str(png_path),
                        stage="texture",
                    )
                )
                return None, issues
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        issues.append(
            issue(
                "E_PNG_DECODE",
                f"PNG decode failed: {exc}",
                path=str(png_path),
                stage="texture",
            )
        )
        return None, issues

    provisional = TextureData(
        path=str(png_path.resolve()),
        width=header["width"],
        height=header["height"],
        mode=expected_mode,
        png_color_type=header["color_type"],
        png_bit_depth=header["bit_depth"],
        rgb=rgb,
        content_key="",
        file_sha256=header["file_sha256"],
    )
    provisional.content_key = texture_key(provisional)
    return provisional, issues


def scan_png_directory(
    directory: str | Path,
    expected_resolution: int | tuple[int, int] | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    report: dict[str, Any] = {"directory": str(root), "files": [], "errors": []}
    if not root.exists() or not root.is_dir():
        report["errors"].append(
            issue("E_IO", "PNG scan directory does not exist", path=str(root), stage="texture")
        )
        return report
    for path in sorted(root.rglob("*.png")):
        texture, issues = load_texture(path, expected_resolution=expected_resolution)
        entry = {
            "path": str(path),
            "ok": texture is not None,
            "issues": issues,
        }
        if texture is not None:
            entry["texture"] = texture.summary()
        report["files"].append(entry)
        report["errors"].extend(issues)
    return report
