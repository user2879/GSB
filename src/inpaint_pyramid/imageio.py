"""PNG validation and deterministic image conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .fixed import from_fixed_u8
from .util import array_fingerprint


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def validate_png(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing PNG file: {path}")
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) < 33 or header[:8] != PNG_SIGNATURE:
        raise ValueError(f"invalid PNG container header: {path.name}")
    if header[12:16] != b"IHDR":
        raise ValueError(f"PNG missing IHDR chunk: {path.name}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {path.name}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Pillow did not identify PNG: {path.name}")
            if image.width != width or image.height != height:
                raise ValueError(f"PNG IHDR/Pillow dimension mismatch: {path.name}")
    except Exception as exc:
        raise ValueError(f"PNG verification failed for {path.name}: {exc}") from exc
    return {"name": path.name, "width": width, "height": height, "bytes": path.stat().st_size}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_fingerprint(path: str | Path) -> str:
    return array_fingerprint(load_rgb(path))


def save_fixed_png(path: str | Path, fixed_rgb: np.ndarray, fraction_bits: int) -> None:
    rgb = from_fixed_u8(fixed_rgb, fraction_bits)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", optimize=False)


def inspect_png_directory(directory: str | Path) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"image directory does not exist: {directory}")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.png")):
        try:
            records.append(validate_png(path))
        except ValueError as exc:
            records.append({"name": path.name, "error": str(exc)})
    return records

