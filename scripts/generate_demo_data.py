"""Generate the deterministic PNG/JSONL/truth fixture set.

The script uses only NumPy and the Python standard library. Running it twice
with the same output directory produces byte-identical files.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np


WIDTH = 64
HEIGHT = 64


def _chunk(name: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(name + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", crc)


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("RGB array required")
    raw = bytearray()
    for row in rgb.astype(np.uint8):
        raw.append(0)
        raw.extend(row.tobytes())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
    payload += _chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    payload += _chunk(b"IEND", b"")
    path.write_bytes(payload)


def _seed(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _draw_rectangle(rgb: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    rgb[y : y + height, x : x + width] = np.asarray(color, dtype=np.uint8)


def _draw_circle(rgb: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    rgb[mask] = np.asarray(color, dtype=np.uint8)


def _draw_stripe(rgb: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    for offset in range(height):
        rgb[y + offset, min(WIDTH - 1, x + offset) : min(WIDTH, x + offset + width)] = np.asarray(color, dtype=np.uint8)


SPECS = [
    {
        "sample_id": "sample-001",
        "background": (30, 48, 64),
        "shapes": [("rectangle", 8, 10, 20, 16, (210, 50, 45)), ("circle", 45, 20, 9, (240, 190, 40))],
        "noise": 4,
        "assertions": [
            {"entity": "shape", "attribute": "color", "value": "red"},
            {"entity": "shape", "attribute": "position", "value": "left"},
            {"entity": "scene", "attribute": "light", "value": "dim"},
        ],
    },
    {
        "sample_id": "sample-002",
        "background": (72, 60, 42),
        "shapes": [("circle", 22, 42, 13, (40, 180, 220)), ("rectangle", 42, 8, 14, 22, (230, 230, 225))],
        "noise": 5,
        "assertions": [
            {"entity": "shape", "attribute": "color", "value": "cyan"},
            {"entity": "shape", "attribute": "shape", "value": "round"},
            {"entity": "scene", "attribute": "light", "value": "bright"},
            {"entity": "shape", "attribute": "position", "value": "center"},
        ],
    },
    {
        "sample_id": "sample-003",
        "background": (28, 70, 52),
        "shapes": [("rectangle", 6, 34, 18, 20, (245, 140, 30)), ("stripe", 28, 10, 7, 40, (180, 80, 200))],
        "noise": 3,
        "assertions": [
            {"entity": "shape", "attribute": "color", "value": "orange"},
            {"entity": "shape", "attribute": "position", "value": "bottom"},
            {"entity": "pattern", "attribute": "orientation", "value": "vertical"},
        ],
    },
    {
        "sample_id": "sample-004",
        "background": (56, 38, 68),
        "shapes": [("circle", 32, 31, 17, (100, 220, 100)), ("rectangle", 8, 8, 10, 10, (250, 250, 250))],
        "noise": 7,
        "assertions": [
            {"entity": "shape", "attribute": "color", "value": "green"},
            {"entity": "shape", "attribute": "position", "value": "center"},
            {"entity": "scene", "attribute": "noise", "value": "high"},
        ],
    },
    {
        "sample_id": "sample-005",
        "background": (18, 18, 24),
        "shapes": [("stripe", 50, 4, 6, 56, (220, 40, 100)), ("circle", 18, 18, 6, (250, 220, 80))],
        "noise": 2,
        "assertions": [],
    },
    {
        "sample_id": "sample-006",
        "background": (84, 84, 70),
        "shapes": [("rectangle", 12, 14, 24, 12, (30, 30, 160)), ("rectangle", 42, 38, 14, 14, (240, 90, 60))],
        "noise": 6,
        "assertions": [
            {"entity": "shape", "attribute": "color", "value": "blue"},
            {"entity": "shape", "attribute": "color", "value": "red"},
            {"entity": "shape", "attribute": "position", "value": "split"},
            {"entity": "scene", "attribute": "light", "value": "flat"},
        ],
    },
]


def main(output_dir: str | Path = "examples/input") -> dict[str, int]:
    root = Path(output_dir)
    images = root / "images"
    truth = root / "truth"
    images.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for spec in SPECS:
        sample_id = str(spec["sample_id"])
        rng = np.random.default_rng(_seed(sample_id))
        background = np.asarray(spec["background"], dtype=np.int16)
        noise = rng.integers(-int(spec["noise"]), int(spec["noise"]) + 1, size=(HEIGHT, WIDTH, 3), dtype=np.int16)
        rgb = np.clip(background[None, None, :] + noise, 0, 255).astype(np.uint8)
        regions: list[dict[str, object]] = []
        for shape in spec["shapes"]:
            kind = shape[0]
            if kind == "rectangle":
                _, x, y, width, height, color = shape
                _draw_rectangle(rgb, int(x), int(y), int(width), int(height), tuple(color))
                regions.append({"label": "rectangle", "bbox": [int(x), int(y), int(width), int(height)]})
            elif kind == "circle":
                _, x, y, radius, color = shape
                _draw_circle(rgb, int(x), int(y), int(radius), tuple(color))
                regions.append({"label": "circle", "bbox": [int(x) - int(radius), int(y) - int(radius), 2 * int(radius) + 1, 2 * int(radius) + 1]})
            elif kind == "stripe":
                _, x, y, width, height, color = shape
                _draw_stripe(rgb, int(x), int(y), int(width), int(height), tuple(color))
                regions.append({"label": "stripe", "bbox": [int(x), int(y), int(width), int(height)]})
        write_png(images / f"{sample_id}.png", rgb)
        truth_doc = {"sample_id": sample_id, "width": WIDTH, "height": HEIGHT, "regions": regions}
        (truth / f"{sample_id}.truth.json").write_text(
            json.dumps(truth_doc, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        lines.append(json.dumps({"sample_id": sample_id, "assertions": spec["assertions"]}, sort_keys=True))
    (root / "texts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"images": len(SPECS), "descriptions": len(SPECS), "truth_files": len(SPECS)}


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
