"""Deterministic procedural terrain data source and quadtree block manifest.

The heightfield, material partition and quadtree blocks are generated purely
from fixed integer seeds. All floating point work is float64. Block records are
built from grid coordinates rather than dictionary/set iteration order, so the
manifest is byte-identical across runs and independent of traversal order.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from .config import (
    HEIGHT_SEED, MATERIAL_SEED, WORLD_EXTENT, MAX_ELEVATION, UINT16_MAX,
    LEAF_SAMPLES, LEAF_OVERLAP,
)
from .errors import HeightRangeError, BlockMissingError

SUPPORTED_SIZES = (512, 2048)
MATERIAL_COUNT = 5


def _value_noise_grid(rng: np.random.Generator, grid: int) -> np.ndarray:
    return rng.random((grid, grid), dtype=np.float64)


def _bilinear_sample(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a periodic value-noise grid at float grid coordinates."""
    n = grid.shape[0]
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    x0 %= n
    y0 %= n
    c00 = grid[y0, x0]
    c10 = grid[y0, x1]
    c01 = grid[y1, x0]
    c11 = grid[y1, x1]
    top = c00 + (c10 - c00) * fx
    bottom = c01 + (c11 - c01) * fx
    return top + (bottom - top) * fy


def fbm_heightfield(size: int, seed: int = HEIGHT_SEED) -> np.ndarray:
    """Generate a deterministic uint16 heightfield of ``size`` x ``size``."""
    if size not in SUPPORTED_SIZES:
        raise HeightRangeError(
            f"unsupported heightfield size {size}; expected one of {SUPPORTED_SIZES}",
            context={"size": size},
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    coords = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(coords, coords)
    field = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    weight_sum = 0.0
    # Octave 0 covers the whole map in 8 cells; each octave doubles frequency.
    for octave in range(6):
        grid_n = 8 * (2 ** octave)
        lattice = _value_noise_grid(rng, grid_n)
        scale = grid_n / size
        sample = _bilinear_sample(lattice, gx * scale, gy * scale)
        field += amplitude * sample
        weight_sum += amplitude
        amplitude *= 0.5
    field /= weight_sum
    # Smooth ridging keeps large coherent hills and valleys.
    field = field ** 1.35
    heights = np.rint(field * UINT16_MAX).astype(np.uint16)
    return heights


def material_partition(size: int, seed: int = MATERIAL_SEED) -> np.ndarray:
    """Deterministic low-frequency material zones plus a small stochastic dither."""
    rng = np.random.Generator(np.random.PCG64(seed))
    zone_grid = _value_noise_grid(rng, 16)
    dither = rng.random((size, size), dtype=np.float64)
    coords = np.arange(size, dtype=np.float64)
    gx, gy = np.meshgrid(coords, coords)
    zone = _bilinear_sample(zone_grid, gx * (16.0 / size), gy * (16.0 / size))
    score = zone * 0.85 + dither * 0.15
    materials = np.minimum((score * MATERIAL_COUNT).astype(np.uint8), MATERIAL_COUNT - 1)
    return materials.astype(np.uint8)


def validate_heightfield(heights: np.ndarray) -> None:
    if heights.dtype != np.uint16:
        raise HeightRangeError(
            "heightfield must be uint16", context={"dtype": str(heights.dtype)}
        )
    if heights.ndim != 2 or heights.shape[0] != heights.shape[1]:
        raise HeightRangeError(
            "heightfield must be a square 2-D array", context={"shape": list(heights.shape)}
        )
    if heights.shape[0] not in SUPPORTED_SIZES:
        raise HeightRangeError(
            f"heightfield size {heights.shape[0]} not in {SUPPORTED_SIZES}",
            context={"shape": list(heights.shape)},
        )
    if int(heights.min()) < 0 or int(heights.max()) > UINT16_MAX:
        raise HeightRangeError(
            "height codes outside uint16 range",
            context={"min": int(heights.min()), "max": int(heights.max())},
        )


def height_to_world(code: np.ndarray | int) -> np.ndarray | float:
    return np.asarray(code, dtype=np.float64) * (MAX_ELEVATION / UINT16_MAX)


@dataclass(frozen=True)
class Block:
    """One quadtree block. Coordinates are grid cells in height-sample units."""

    block_id: str
    level: int
    bx: int
    by: int
    size_samples: int
    x0: int
    y0: int
    x1: int
    y1: int
    bbox: tuple          # (minx, miny, minz, maxx, maxy, maxz) world-space
    height_min: int
    height_max: int
    height_range: int
    payload_bytes: int
    vram_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _make_block(level: int, bx: int, by: int, size_samples: int,
                heights: np.ndarray, materials: np.ndarray) -> Block:
    size = heights.shape[0]
    x0 = bx * size_samples
    y0 = by * size_samples
    x1 = min(x0 + size_samples + LEAF_OVERLAP, size)
    y1 = min(y0 + size_samples + LEAF_OVERLAP, size)
    region_h = heights[y0:y1, x0:x1]
    h_min = int(region_h.min())
    h_max = int(region_h.max())
    cell = WORLD_EXTENT / size
    minx = x0 * cell
    maxx = x1 * cell
    minz = y0 * cell
    maxz = y1 * cell
    miny = float(height_to_world(h_min))
    maxy = float(height_to_world(h_max))
    samples = region_h.size
    material_cells = materials[y0:y1, x0:x1].size
    # Height payload (uint16) + material payload (uint8) resident in system memory.
    payload_bytes = samples * 2 + material_cells
    # VRAM reserves vertex/index/mip bookkeeping: position(3f)+normal(3f)+uv(2f).
    vram_bytes = samples * 8 * 4
    block_id = f"L{level:02d}_X{bx:04d}_Y{by:04d}"
    return Block(
        block_id=block_id, level=level, bx=bx, by=by,
        size_samples=size_samples, x0=x0, y0=y0, x1=x1, y1=y1,
        bbox=(minx, miny, minz, maxx, maxy, maxz),
        height_min=h_min, height_max=h_max, height_range=h_max - h_min,
        payload_bytes=int(payload_bytes), vram_bytes=int(vram_bytes),
    )


def build_quadtree(heights: np.ndarray, materials: np.ndarray) -> list[Block]:
    """Build all quadtree levels.

    Leaves cover the map with ``LEAF_SAMPLES`` cells. Each parent level halves
    the grid resolution. Blocks are constructed by explicit nested coordinate
    loops and then sorted by ``block_id``; nothing depends on hash iteration.
    """
    validate_heightfield(heights)
    size = heights.shape[0]
    if materials.shape != heights.shape:
        raise HeightRangeError(
            "material map shape must match heightfield",
            context={"height_shape": list(heights.shape),
                     "material_shape": list(materials.shape)},
        )
    blocks: list[Block] = []
    leaf_count = size // LEAF_SAMPLES
    level = 0
    current = leaf_count
    block_size = LEAF_SAMPLES
    while current >= 1:
        for by in range(current):
            for bx in range(current):
                blocks.append(_make_block(level, bx, by, block_size,
                                          heights, materials))
        if current == 1:
            break
        current //= 2
        block_size *= 2
        level += 1
    blocks.sort(key=lambda b: b.block_id)
    return blocks


class TerrainDataset:
    """In-memory deterministic terrain source with persistence helpers."""

    def __init__(self, size: int, heights: np.ndarray, materials: np.ndarray,
                 blocks: list[Block]):
        self.size = size
        self.heights = heights
        self.materials = materials
        self.blocks = blocks
        self._index = {b.block_id: b for b in blocks}
        self.height_sha256 = hashlib.sha256(heights.tobytes()).hexdigest()
        self.material_sha256 = hashlib.sha256(materials.tobytes()).hexdigest()

    @classmethod
    def generate(cls, size: int) -> "TerrainDataset":
        heights = fbm_heightfield(size)
        materials = material_partition(size)
        blocks = build_quadtree(heights, materials)
        return cls(size, heights, materials, blocks)

    def get_block(self, block_id: str) -> Block:
        try:
            return self._index[block_id]
        except KeyError:
            raise BlockMissingError(
                f"block {block_id} not present in manifest",
                context={"block_id": block_id},
            )

    def leaf_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.level == 0]

    def manifest_dict(self) -> dict:
        return {
            "size": self.size,
            "world_extent": WORLD_EXTENT,
            "leaf_samples": LEAF_SAMPLES,
            "height_sha256": self.height_sha256,
            "material_sha256": self.material_sha256,
            "block_count": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    def save(self, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        # Raw little-endian arrays carry no timestamps -> byte reproducible.
        h_path = os.path.join(out_dir, f"height_{self.size}.u16.bin")
        m_path = os.path.join(out_dir, f"material_{self.size}.u8.bin")
        with open(h_path, "wb") as fh:
            fh.write(self.heights.astype("<u2", copy=False).tobytes())
        with open(m_path, "wb") as fh:
            fh.write(self.materials.astype(np.uint8, copy=False).tobytes())
        manifest = self.manifest_dict()
        manifest_path = os.path.join(out_dir, f"manifest_{self.size}.json")
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        with open(manifest_path, "wb") as fh:
            fh.write(payload)
        manifest_sha = hashlib.sha256(payload).hexdigest()
        # Basenames only: absolute output paths must not enter content hashes,
        # otherwise two identical runs in different directories diverge.
        return {
            "height_path": os.path.basename(h_path),
            "material_path": os.path.basename(m_path),
            "manifest_path": os.path.basename(manifest_path),
            "height_sha256": self.height_sha256,
            "material_sha256": self.material_sha256,
            "manifest_sha256": manifest_sha,
        }
