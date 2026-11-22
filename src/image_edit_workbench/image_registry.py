from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .errors import E_IMAGE_IO, WorkbenchError
from .models import ImageInfo
from .util import fingerprint, sha256_json


def load_images(image_dir: Path, *, expected_size: tuple[int, int] | None = None) -> dict[str, ImageInfo]:
    if not image_dir.is_dir():
        raise WorkbenchError(E_IMAGE_IO, f"image directory not found: {image_dir}", stage="images")
    images: dict[str, ImageInfo] = {}
    for path in sorted(image_dir.glob("*.png")):
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                pixels = rgb.tobytes()
        except OSError as exc:
            raise WorkbenchError(E_IMAGE_IO, f"failed to read {path}: {exc}", stage="images") from exc
        if expected_size is not None and (width, height) != expected_size:
            raise WorkbenchError(
                E_IMAGE_IO,
                f"{path.name} is {width}x{height}, expected {expected_size[0]}x{expected_size[1]}",
                stage="images",
            )
        source_fp = fingerprint("png-rgb-pixels", path.name, width, height, pixels)
        images[path.name] = ImageInfo(path.name, str(path), width, height, source_fp)
    if not images:
        raise WorkbenchError(E_IMAGE_IO, f"no PNG images found in {image_dir}", stage="images")
    return images


def image_collection_fingerprint(images: Mapping[str, ImageInfo]) -> str:
    return sha256_json({name: info.source_fingerprint for name, info in sorted(images.items())})


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

