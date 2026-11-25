from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ALLOWED_COLORMAPS
from .errors import ErrorCode, make_issue
from .hashing import sha256_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLORMAP_CODE_COUNT = 1001
COLORMAP_DISTANCE_LIMIT_SQ = 9000


def _make_injective(table: np.ndarray) -> np.ndarray:
    used: set[tuple[int, int, int]] = set()
    for index in range(COLORMAP_CODE_COUNT):
        r, g, b = (int(v) for v in table[index])
        chosen: tuple[int, int, int] | None = None
        candidates: list[tuple[int, int, int]] = [(r, g, b)]
        for delta in range(1, 256):
            candidates.extend(
                [
                    (r + delta, g, b),
                    (r - delta, g, b),
                    (r, g + delta, b),
                    (r, g - delta, b),
                    (r, g, b + delta),
                    (r, g, b - delta),
                ]
            )
        for candidate in candidates:
            rr, gg, bb = candidate
            if not (0 <= rr <= 255 and 0 <= gg <= 255 and 0 <= bb <= 255):
                continue
            if candidate not in used:
                chosen = candidate
                break
        if chosen is None:
            raise RuntimeError("failed to allocate injective pseudo-color table")
        table[index] = chosen
        used.add(chosen)
    return table


def palette(colormap_name: str) -> np.ndarray:
    if colormap_name not in ALLOWED_COLORMAPS:
        raise ValueError(f"unknown colormap {colormap_name}")
    heat = np.arange(COLORMAP_CODE_COUNT, dtype=np.float64) / 1000.0
    table = np.zeros((COLORMAP_CODE_COUNT, 3), dtype=np.uint8)

    if colormap_name == "GRAY_V1":
        value = np.rint(heat * 255.0).astype(np.uint8)
        table[:, 0] = value
        table[:, 1] = value
        table[:, 2] = value
        return _make_injective(table)

    red = np.clip(np.rint(np.where(heat < 0.4, heat / 0.4 * 190.0 + 25.0, 215.0 + (heat - 0.4) / 0.6 * 40.0)), 0, 255)
    green = np.clip(
        np.rint(
            np.where(
                heat < 0.42,
                heat / 0.42 * 35.0,
                np.where(heat < 0.78, 35.0 + (heat - 0.42) / 0.36 * 180.0, 215.0 + (heat - 0.78) / 0.22 * 40.0),
            )
        ),
        0,
        255,
    )
    blue = np.clip(
        np.rint(
            np.where(
                heat < 0.22,
                55.0 + heat / 0.22 * 55.0,
                np.where(heat < 0.52, 110.0 * (1.0 - (heat - 0.22) / 0.30), 0.0),
            )
        ),
        0,
        255,
    )
    table[:, 0] = red.astype(np.uint8)
    table[:, 1] = green.astype(np.uint8)
    table[:, 2] = blue.astype(np.uint8)
    return _make_injective(table)


def heat_to_rgb(heat: np.ndarray, colormap_name: str) -> np.ndarray:
    table = palette(colormap_name)
    clipped = np.clip(np.rint(heat).astype(np.int64), 0, COLORMAP_CODE_COUNT - 1)
    return table[clipped]


def write_png(path: str | Path, heat: np.ndarray, colormap_name: str) -> Path:
    rgb = heat_to_rgb(heat, colormap_name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)
    return path


def scan_pngs(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted((path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".png"), key=lambda p: p.name)


def _decode_unique_colors(rgb: np.ndarray, colormap_name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    table = palette(colormap_name).astype(np.int32)
    pixels = rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(pixels, axis=0, return_inverse=True)
    exact = {(int(r), int(g), int(b)): h for h, (r, g, b) in enumerate(palette(colormap_name))}
    unique_heat = np.zeros(len(unique_colors), dtype=np.int32)

    for index, color in enumerate(unique_colors):
        key = (int(color[0]), int(color[1]), int(color[2]))
        if key in exact:
            unique_heat[index] = exact[key]
            continue
        distances = np.sum((table - color.astype(np.int32)) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        nearest_distance = int(distances[nearest])
        if nearest_distance > COLORMAP_DISTANCE_LIMIT_SQ:
            issues.append(
                make_issue(
                    ErrorCode.IMAGE_FORMAT,
                    "pixel color code is outside the supported pseudo-color table",
                    reason="COLORMAP_CODE_OUT_OF_RANGE",
                    rgb=list(key),
                    nearest_heat_code=nearest,
                    distance_sq=nearest_distance,
                )
            )
        unique_heat[index] = nearest
    return unique_heat[inverse].reshape(rgb.shape[0], rgb.shape[1]), issues


def load_and_decode_png(
    path: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = Path(path)
    issues: list[dict[str, Any]] = []
    try:
        data = path.read_bytes()
        if len(data) < 24 or data[:8] != PNG_SIGNATURE:
            raise ValueError("missing PNG signature or truncated IHDR")
        ihdr_length = struct.unpack(">I", data[8:12])[0]
        ihdr_type = data[12:16]
        if ihdr_length != 13 or ihdr_type != b"IHDR":
            raise ValueError("invalid PNG IHDR container")
        header_width, header_height = struct.unpack(">II", data[16:24])
        expected_width = int(config["fixed_width"])
        expected_height = int(config["fixed_height"])
        if (header_width, header_height) != (expected_width, expected_height):
            raise ValueError(f"fixed resolution mismatch: {header_width}x{header_height}")

        with Image.open(path) as image:
            image.load()
            mode = image.mode
            size = image.size
            if mode != config["pixel_mode"]:
                raise ValueError(f"pixel mode mismatch: {mode}")
            if size != (expected_width, expected_height):
                raise ValueError(f"decoded resolution mismatch: {size[0]}x{size[1]}")
            rgb = np.asarray(image.convert("RGB" if mode != "RGB" else None), dtype=np.uint8)

        heat, color_issues = _decode_unique_colors(rgb, str(config["colormap_name"]))
        issues.extend(color_issues)
        if color_issues:
            return None, issues
        return {
            "path": str(path),
            "image": path.name,
            "image_sha256": sha256_bytes(data),
            "width": expected_width,
            "height": expected_height,
            "mode": mode,
            "heat": heat.astype(np.int32),
        }, issues
    except Exception as exc:  # Pillow raises varied OSError/SyntaxError types.
        issues.append(
            make_issue(
                ErrorCode.IMAGE_FORMAT,
                "PNG container could not be validated or decoded",
                path=str(path),
                image=path.name,
                reason=type(exc).__name__,
                detail=str(exc),
            )
        )
        return None, issues
