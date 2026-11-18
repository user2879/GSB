"""Local image loading, deterministic projection and PNG-ready decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ResolutionTier


LATENT_CHANNELS = 4


# Fixed matrices are intentionally constants.  No model weights or random
# initialisation are involved, which keeps every projection reproducible.
LATENT_TO_RGB = np.asarray(
    [
        [0.92, 0.18, 0.08],
        [0.11, 0.90, 0.17],
        [0.16, 0.10, 0.91],
        [0.31, 0.31, 0.31],
    ],
    dtype=np.float64,
)

RGB_TO_LATENT = np.asarray(
    [
        [0.88, 0.10, 0.18, 0.20],
        [0.15, 0.86, 0.12, 0.20],
        [0.10, 0.17, 0.90, 0.20],
    ],
    dtype=np.float64,
)

DECODE_SCALE = 55.0
PIXEL_CENTER = 127.5


def _read_ppm(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    position = 0
    tokens: list[bytes] = []
    while len(tokens) < 4:
        while position < len(payload) and payload[position : position + 1].isspace():
            position += 1
        if position < len(payload) and payload[position : position + 1] == b"#":
            while position < len(payload) and payload[position : position + 1] not in (b"\n", b"\r"):
                position += 1
            continue
        start = position
        while position < len(payload) and not payload[position : position + 1].isspace():
            position += 1
        if start == position:
            raise ValueError("truncated PPM header")
        tokens.append(payload[start:position])
    magic, width_raw, height_raw, maxval_raw = tokens
    if magic == b"P6":
        if payload[position : position + 2] == b"\r\n":
            position += 2
        elif position < len(payload) and payload[position] in (9, 10, 11, 12, 13, 32):
            position += 1

    if magic not in (b"P6", b"P3"):
        raise ValueError("only P6/P3 PPM images are supported")
    width, height, maxval = int(width_raw), int(height_raw), int(maxval_raw)
    if width <= 0 or height <= 0 or maxval != 255:
        raise ValueError("PPM must have positive dimensions and maxval 255")
    if magic == b"P6":
        expected = width * height * 3
        body = payload[position:]
        if len(body) < expected:
            raise ValueError("truncated P6 image body")
        return np.frombuffer(body[:expected], dtype=np.uint8).reshape(height, width, 3).copy()

    values = payload[position:].split()
    if len(values) < width * height * 3:
        raise ValueError("truncated P3 image body")
    return np.asarray([int(value) for value in values[: width * height * 3]], dtype=np.uint8).reshape(
        height, width, 3
    )


def load_initial_image(path: str | Path) -> np.ndarray:
    return _read_ppm(Path(path))


def resize_nearest(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    source = np.asarray(pixels, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("pixels must be HxWx3")
    source_height, source_width = source.shape[:2]
    if (source_width, source_height) == (width, height):
        return source.copy()
    row_indices = np.minimum((np.arange(height, dtype=np.int64) * source_height) // height, source_height - 1)
    column_indices = np.minimum((np.arange(width, dtype=np.int64) * source_width) // width, source_width - 1)
    return np.ascontiguousarray(source[row_indices[:, None], column_indices[None, :], :])


def encode_image_latent(pixels: np.ndarray, tier: ResolutionTier) -> np.ndarray:
    resized = resize_nearest(pixels, tier.width, tier.height)
    normalized = resized.astype(np.float64) / 255.0 * 2.0 - 1.0
    latent = normalized @ RGB_TO_LATENT
    latent = np.tanh(latent * 0.85)
    return np.ascontiguousarray(latent, dtype=np.float64)


def decode_latent(latent: np.ndarray) -> np.ndarray:
    array = np.asarray(latent, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != LATENT_CHANNELS:
        raise ValueError("latent must have shape HxWx4")
    projected = np.tensordot(array, LATENT_TO_RGB, axes=([2], [0]))
    rgb = PIXEL_CENTER + DECODE_SCALE * projected
    # The operation order is fixed: add bias, clip, then round to nearest.
    clipped = np.clip(rgb, 0.0, 255.0)
    rounded = np.floor(clipped + 0.5)
    return np.ascontiguousarray(rounded.astype(np.uint8))
