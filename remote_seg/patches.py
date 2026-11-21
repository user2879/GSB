"""Deterministic local multi-band PNG patch application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .errors import E_PATCH_BOUNDS, E_TILE_SHAPE_MISMATCH, SegmentationError
from .pngio import encode_png, read_png
from .util import array_content_fingerprint, atomic_write_bytes, read_json


@dataclass(frozen=True, slots=True)
class PatchRecord:
    batch: int
    tile_id: str
    row: int
    col: int
    rect: tuple[int, int, int, int]
    before_fingerprint: str
    after_fingerprint: str


def _synthetic_patch_values(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = np.array(
        [
            [72, 126, 186, 102],
            [22, 46, 19, 17],
            [151, 143, 149, 180],
            [113, 119, 122, 58],
        ],
        dtype=np.float64,
    )
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    classes = ((x // 3) + (y // 3) + seed) % 4
    values = palette[classes].copy()
    values += rng.normal(0.0, 2.5, size=values.shape)
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def apply_patch_batch(index_path: Path, plan_path: Path, batch_number: int) -> list[PatchRecord]:
    try:
        index = read_json(index_path)
        plan = read_json(plan_path)
    except (OSError, ValueError) as exc:
        raise SegmentationError(
            E_PATCH_BOUNDS, f"cannot read patch inputs: {exc}", stage="patch"
        ) from exc
    batches = plan.get("batches") if isinstance(plan, dict) else None
    if not isinstance(batches, list):
        raise SegmentationError(E_PATCH_BOUNDS, "patch plan has no batches", stage="patch")
    batch = next((item for item in batches if item.get("batch") == batch_number), None)
    if batch is None:
        raise SegmentationError(
            E_PATCH_BOUNDS,
            f"patch batch {batch_number} is not declared",
            stage="patch",
            details={"batch": batch_number},
        )
    tiles = index.get("tiles") if isinstance(index, dict) else None
    if not isinstance(tiles, list):
        raise SegmentationError(E_PATCH_BOUNDS, "index has no tiles", stage="patch")
    tile_size = index.get("tile_size")
    if not isinstance(tile_size, list) or len(tile_size) != 2:
        raise SegmentationError(E_PATCH_BOUNDS, "index tile_size is invalid", stage="patch")
    tile_width, tile_height = int(tile_size[0]), int(tile_size[1])
    by_cell = {(item.get("row"), item.get("col")): item for item in tiles if isinstance(item, dict)}
    records: list[PatchRecord] = []
    for operation in batch.get("operations", []):
        if not isinstance(operation, dict):
            raise SegmentationError(E_PATCH_BOUNDS, "patch operation must be an object", stage="patch")
        row = operation.get("row")
        col = operation.get("col")
        tile = by_cell.get((row, col))
        tile_id = operation.get("tile_id")
        if tile is None or (tile_id is not None and tile.get("tile_id") != tile_id):
            raise SegmentationError(
                E_PATCH_BOUNDS,
                f"patch target row={row} col={col} tile={tile_id!r} is not indexed",
                stage="patch",
                row=row if isinstance(row, int) else None,
                col=col if isinstance(col, int) else None,
                tile_id=tile_id if isinstance(tile_id, str) else None,
            )
        rect = operation.get("rect")
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in rect)
        ):
            raise SegmentationError(
                E_PATCH_BOUNDS,
                "patch rect must be [x0,y0,x1,y1]",
                stage="patch",
                row=row,
                col=col,
                tile_id=tile.get("tile_id"),
            )
        x0, y0, x1, y1 = rect
        within = 0 <= x0 < x1 <= tile_width and 0 <= y0 < y1 <= tile_height
        within = within and (tile["bounds"][2] - tile["bounds"][0]) == tile_width
        within = within and (tile["bounds"][3] - tile["bounds"][1]) == tile_height
        if not within:
            raise SegmentationError(
                E_PATCH_BOUNDS,
                f"patch rect {rect} is outside {tile_width}x{tile_height} tile",
                stage="patch",
                row=row,
                col=col,
                tile_id=tile.get("tile_id"),
                details={"rect": rect, "tile_size": [tile_width, tile_height]},
            )
        tile_path = index_path.parent / tile["path"]
        decoded = read_png(tile_path)
        if decoded.channels != 4:
            raise SegmentationError(
                E_TILE_SHAPE_MISMATCH,
                f"cannot patch {decoded.channels}-band tile",
                stage="patch",
                row=row,
                col=col,
                tile_id=tile.get("tile_id"),
            )
        if (decoded.width, decoded.height) != (tile_width, tile_height):
            raise SegmentationError(
                E_TILE_SHAPE_MISMATCH,
                f"decoded tile size {decoded.width}x{decoded.height} does not match "
                f"{tile_width}x{tile_height}",
                stage="patch",
                row=row,
                col=col,
                tile_id=tile.get("tile_id"),
            )
        before = array_content_fingerprint(decoded.array)
        seed = operation.get("seed", 1)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SegmentationError(E_PATCH_BOUNDS, "patch seed must be an integer", stage="patch")
        values = _synthetic_patch_values(y1 - y0, x1 - x0, seed)
        updated = decoded.array.copy()
        updated[y0:y1, x0:x1, :] = values
        atomic_write_bytes(tile_path, encode_png(updated))
        after = array_content_fingerprint(updated)
        records.append(
            PatchRecord(
                batch=batch_number,
                tile_id=str(tile.get("tile_id")),
                row=int(row),
                col=int(col),
                rect=(x0, y0, x1, y1),
                before_fingerprint=before,
                after_fingerprint=after,
            )
        )
    return records
