"""Texture scanning, container validation and decoding with failure isolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import Failure, TextureDecodeError
from .hashing import array_fingerprint, sha256_hex

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class DecodedTexture:
    name: str
    path: Path
    width: int
    height: int
    mode: str
    pixels: np.ndarray
    content_fingerprint: int
    file_hash: str


def luminance(pixels: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an RGB/RGBA/L uint8 array, returned as uint8 [0,255]."""
    if pixels.ndim == 2:
        return pixels.copy()
    rgb = pixels[:, :, :3].astype(np.uint16)
    luma = (rgb[:, :, 0] * 299 + rgb[:, :, 1] * 587 + rgb[:, :, 2] * 114) // 1000
    return luma.astype(np.uint8)


def _check_png_header(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(8)
    if header != PNG_SIGNATURE:
        raise TextureDecodeError(f"{path.name}: invalid PNG container signature")


def decode_texture(path: Path, expected_size: int, allowed_modes: tuple[str, ...]) -> DecodedTexture:
    _check_png_header(path)
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # Pillow raises a variety of exceptions on corruption
        raise TextureDecodeError(f"{path.name}: PNG verification failed: {exc}") from exc
    try:
        with Image.open(path) as image:
            mode = image.mode
            if mode not in allowed_modes:
                raise TextureDecodeError(
                    f"{path.name}: pixel mode {mode} not allowed, expected one of {allowed_modes}")
            if image.size != (expected_size, expected_size):
                raise TextureDecodeError(
                    f"{path.name}: size {image.size} does not match expected {(expected_size, expected_size)}")
            pixels = np.asarray(image, dtype=np.uint8).copy()
    except TextureDecodeError:
        raise
    except Exception as exc:
        raise TextureDecodeError(f"{path.name}: decode failed: {exc}") from exc
    file_hash = sha256_hex(path.read_bytes())
    return DecodedTexture(
        name=path.stem, path=path, width=expected_size, height=expected_size, mode=mode,
        pixels=pixels, content_fingerprint=array_fingerprint(pixels), file_hash=file_hash,
    )


def scan_textures(sample_dir: str | Path, expected_size: int,
                  allowed_modes: tuple[str, ...] = ("RGB", "RGBA", "L")
                  ) -> tuple[dict[str, DecodedTexture], list[Failure]]:
    """Decode every ``*.png`` in sorted filename order; corrupt files only add
    failures and never stop the remaining files from being processed."""
    sample_dir = Path(sample_dir)
    textures: dict[str, DecodedTexture] = {}
    failures: list[Failure] = []
    for path in sorted(sample_dir.glob("*.png"), key=lambda p: p.name):
        try:
            texture = decode_texture(path, expected_size, allowed_modes)
        except TextureDecodeError as exc:
            failures.append(Failure(
                code=exc.code, target=path.name, message=exc.message,
                order_key=f"texture:{path.name}", details=exc.details,
            ))
            continue
        if texture.name in textures:
            failures.append(Failure(
                code="E_TEXTURE_DECODE", target=texture.name,
                message="duplicate texture stem name", order_key=f"texture:{texture.name}"))
            continue
        textures[texture.name] = texture
    return textures, failures
