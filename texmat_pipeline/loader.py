"""Sample and configuration loading layer (business requirement 1).

Scans every ``*.png`` under the input directory, validates the PNG container
signature, image size and pixel mode, then decodes to uint8 arrays.  Files that
cannot be decoded are recorded with ``E_TEX_DECODE`` while the remaining
samples keep processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import E_TEX_DECODE, Failure

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ALLOWED_MODES = ("L", "RGB", "RGBA")


@dataclass(frozen=True)
class BaseTexture:
    texture_id: str
    path: Path
    width: int
    height: int
    mode: str
    gray: np.ndarray  # uint8 (H, W), row-major

    def to_dict(self) -> dict:
        return {
            "texture_id": self.texture_id,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
        }


def _to_grayscale(img: Image.Image) -> np.ndarray:
    if img.mode == "L":
        return np.asarray(img, dtype=np.uint8).copy()
    arr = np.asarray(img, dtype=np.uint8)
    if img.mode == "RGB":
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    else:  # RGBA: composite over black then luma
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
        scale = a.astype(np.uint16)
        r = (r.astype(np.uint16) * scale // 255).astype(np.uint8)
        g = (g.astype(np.uint16) * scale // 255).astype(np.uint8)
        b = (b.astype(np.uint16) * scale // 255).astype(np.uint8)
    luma = (r.astype(np.uint16) * 77 + g.astype(np.uint16) * 150 + b.astype(np.uint16) * 29) // 256
    return luma.astype(np.uint8)


def load_png(path: Path, expected_size: int | None) -> tuple[BaseTexture | None, Failure | None, bool]:
    """Returns (texture, failure, skipped).  ``skipped`` marks a valid PNG that
    belongs to another resolution tier and must not pollute the failure list.
    """
    try:
        header = path.read_bytes()[:8]
        if header != PNG_SIGNATURE:
            return None, Failure(E_TEX_DECODE, str(path), "missing or invalid PNG container signature"), False
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.mode not in ALLOWED_MODES:
                return None, Failure(E_TEX_DECODE, str(path), f"unsupported pixel mode {img.mode!r}"), False
            width, height = img.size
            if width != height:
                return None, Failure(E_TEX_DECODE, str(path), f"texture must be square, got {width}x{height}"), False
            if expected_size is not None and width != expected_size:
                return None, None, True
            gray = _to_grayscale(img)
            mode = img.mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, Failure(E_TEX_DECODE, str(path), f"cannot decode PNG: {exc}"), False
    texture_id = path.stem
    return BaseTexture(texture_id, path, width, height, mode, gray), None, False


def scan_textures(input_dir: str | Path, resolution: int) -> tuple[dict[str, BaseTexture], list[Failure]]:
    """Load every PNG matching the requested resolution tier.

    PNGs that decode correctly but belong to another resolution tier are
    skipped silently; genuinely undecodable files produce ``E_TEX_DECODE`` and
    the remaining samples continue processing.
    """
    root = Path(input_dir)
    textures: dict[str, BaseTexture] = {}
    failures: list[Failure] = []
    for path in sorted(root.glob("*.png")):
        texture, failure, skipped = load_png(path, resolution)
        if skipped:
            continue
        if failure is not None:
            failures.append(failure)
            continue
        if texture.texture_id in textures:
            failures.append(Failure(E_TEX_DECODE, str(path), "duplicate texture id"))
            continue
        textures[texture.texture_id] = texture
    return textures, failures
