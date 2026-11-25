from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import is_int, load_json
from .errors import WorkbenchError
from .hashing import canonical_hash


@dataclass(frozen=True)
class Rect:
    id: str
    x: int
    y: int
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


def parse_rect(raw: Any, image_width: int, image_height: int, *, require_id: bool = True) -> Rect:
    if not isinstance(raw, dict):
        raise WorkbenchError("E_MASK_OUT_OF_RANGE", "mask rectangle must be an object", {"value": raw})
    rect_id = raw.get("id")
    if require_id:
        if not isinstance(rect_id, str) or not rect_id:
            raise WorkbenchError("E_MASK_OUT_OF_RANGE", "mask rectangle id must be a non-empty string", {"id": rect_id})
    elif rect_id is None:
        rect_id = ""

    coords: dict[str, int] = {}
    for key in ("x", "y", "width", "height"):
        value = raw.get(key)
        if not is_int(value):
            raise WorkbenchError(
                "E_MASK_OUT_OF_RANGE",
                f"mask rectangle {key} must be an integer",
                {"rectangle_id": rect_id, "field": key, "value": value},
            )
        coords[key] = int(value)

    x, y, width, height = coords["x"], coords["y"], coords["width"], coords["height"]
    if width <= 0 or height <= 0:
        raise WorkbenchError(
            "E_MASK_OUT_OF_RANGE",
            "mask rectangle width and height must be positive",
            {"rectangle_id": rect_id, "x": x, "y": y, "width": width, "height": height},
        )
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise WorkbenchError(
            "E_MASK_OUT_OF_RANGE",
            "mask rectangle is outside the fixed image bounds",
            {
                "rectangle_id": rect_id,
                "image_width": image_width,
                "image_height": image_height,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },
        )
    return Rect(str(rect_id), x, y, width, height)


def load_mask_document(path: str | Path, image_width: int, image_height: int) -> dict[str, Rect]:
    data = load_json(path, "E_MASK_OUT_OF_RANGE")
    if not isinstance(data, dict):
        raise WorkbenchError("E_MASK_OUT_OF_RANGE", "mask document must be a JSON object")
    if data.get("width") != image_width or data.get("height") != image_height:
        raise WorkbenchError(
            "E_MASK_OUT_OF_RANGE",
            "mask document dimensions do not match the fixed image",
            {"mask_width": data.get("width"), "mask_height": data.get("height"),
             "image_width": image_width, "image_height": image_height},
        )
    raw_rects = data.get("rectangles")
    if not isinstance(raw_rects, list):
        raise WorkbenchError("E_MASK_OUT_OF_RANGE", "mask document must contain a rectangles list")
    rects: dict[str, Rect] = {}
    for index, raw in enumerate(raw_rects):
        rect = parse_rect(raw, image_width, image_height)
        if rect.id in rects:
            raise WorkbenchError("E_MASK_OUT_OF_RANGE", "duplicate mask rectangle id", {"id": rect.id, "index": index})
        rects[rect.id] = rect
    return rects


def mask_hash(rectangles: dict[str, Rect]) -> str:
    return canonical_hash([rectangles[key].to_json() for key in sorted(rectangles)])


def rasterize_rectangles(rectangles: list[Rect] | dict[str, Rect], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    values = list(rectangles.values()) if isinstance(rectangles, dict) else list(rectangles)
    for rect in values:
        mask[rect.y:rect.y2, rect.x:rect.x2] = True
    return mask


def downsample_mask_any(mask: np.ndarray, factor: int) -> np.ndarray:
    height, width = mask.shape
    if height % factor or width % factor:
        raise WorkbenchError(
            "E_TILE_GEOMETRY",
            "mask level cannot be downsampled because factor does not divide dimensions",
            {"shape": list(mask.shape), "factor": factor},
        )
    blocks = mask.reshape(height // factor, factor, width // factor, factor)
    return np.any(blocks, axis=(1, 3))
