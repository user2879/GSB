from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .errors import E_MASK_FORMAT_INVALID, PipelineError


def encode_panorama_png(
    instance_ids: np.ndarray,
    class_ids: np.ndarray,
    path: str | Path,
) -> None:
    instance_ids = np.asarray(instance_ids, dtype=np.uint32)
    class_ids = np.asarray(class_ids, dtype=np.uint32)
    if instance_ids.shape != class_ids.shape:
        raise ValueError("instance_ids and class_ids must have equal shapes")
    if np.any(instance_ids > 0xFFFF) or np.any(class_ids > 0xFFFF):
        raise ValueError("instance and class IDs must fit in 16 bits")
    packed = (instance_ids << 16) | class_ids
    rgba = np.stack(
        (
            (packed >> 24) & 0xFF,
            (packed >> 16) & 0xFF,
            (packed >> 8) & 0xFF,
            packed & 0xFF,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def decode_panorama_png(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint32)
    except (OSError, ValueError) as exc:
        raise PipelineError(
            E_MASK_FORMAT_INVALID,
            f"cannot decode PNG panorama {path}: {exc}",
        ) from exc
    packed = (
        (rgba[:, :, 0] << 24)
        | (rgba[:, :, 1] << 16)
        | (rgba[:, :, 2] << 8)
        | rgba[:, :, 3]
    )
    instance_ids = (packed >> 16) & 0xFFFF
    class_ids = packed & 0xFFFF
    return instance_ids, class_ids
