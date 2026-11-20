from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import numpy_sha256, sha256_bytes, write_json
from .errors import ErrorCode, fail


SUPPORTED_SIZES = (257, 1025)
DEFAULT_SEED = 0x5EED1234
HEIGHT_SCALE = 0.01
OCTAVE_LATTICES = (4, 8, 16, 32, 64)
OCTAVE_WEIGHTS = (10, 7, 5, 3, 2)
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class TerrainConfig:
    size: int = 1025
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.size not in SUPPORTED_SIZES:
            raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {self.size}")
        if not 0 <= int(self.seed) <= MASK64:
            raise ValueError("seed must fit in uint64")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.uint64, copy=True)
    result += np.uint64(0x9E3779B97F4A7C15)
    result ^= result >> np.uint64(30)
    result *= np.uint64(0xBF58476D1CE4E5B9)
    result ^= result >> np.uint64(27)
    result *= np.uint64(0x94D049BB133111EB)
    result ^= result >> np.uint64(31)
    return result


def _lattice(seed: int, octave: int, lattice_size: int) -> np.ndarray:
    values = np.empty((lattice_size + 1, lattice_size + 1), dtype=np.uint32)
    domain = (seed ^ (0xA24BAED4963EE407 * (octave + 1))) & MASK64
    for row in range(lattice_size + 1):
        for column in range(lattice_size + 1):
            cell = (domain + row * 0x9E3779B185EBCA87 + column * 0xC2B2AE3D27D4EB4F) & MASK64
            values[row, column] = np.uint32(_splitmix64(cell) >> 32)
    return values


def _value_noise(size: int, lattice: np.ndarray, octave: int) -> np.ndarray:
    lattice_size = lattice.shape[0] - 1
    coordinates = np.arange(size, dtype=np.float64) * (lattice_size / float(size - 1))
    base = np.floor(coordinates).astype(np.int64)
    fraction = np.clip(((coordinates - base) * 65536.0).astype(np.int64), 0, 65536)
    next_index = np.minimum(base + 1, lattice_size)
    base = np.minimum(base, lattice_size)
    fx = fraction[:, None]
    fz = fraction[None, :]

    values = lattice.astype(np.int64) >> 16
    v00 = values[np.ix_(base, base)]
    v10 = values[np.ix_(next_index, base)]
    v01 = values[np.ix_(base, next_index)]
    v11 = values[np.ix_(next_index, next_index)]
    top = v00 * (65536 - fx) + v10 * fx
    bottom = v01 * (65536 - fx) + v11 * fx
    interpolated = (top * (65536 - fz) + bottom * fz) >> 32
    row_indices = np.arange(size, dtype=np.uint64)
    column_indices = np.arange(size, dtype=np.uint64)
    row_seed = row_indices ^ np.uint64(((octave + 1) << 32) ^ 0xD1B54A32D192ED03)
    row_hashes = _splitmix64_array(row_seed)
    detail = _splitmix64_array(
        row_hashes[:, None] ^ (column_indices[None, :] * np.uint64(0x94D049BB133111EB))
    ) & np.uint64(0x1F)
    return np.clip(interpolated + detail.astype(np.int64) - 15, 0, 65535).astype(np.uint16)


@dataclass(frozen=True)
class GeneratedTerrain:
    config: TerrainConfig
    heights: np.ndarray
    layer_metadata: tuple[dict[str, Any], ...]

    @property
    def content_sha256(self) -> str:
        return numpy_sha256(self.heights, "<u2")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "size": self.config.size,
            "seed": int(self.config.seed),
            "shape": [self.config.size, self.config.size],
            "dtype": "uint16",
            "byte_order": "little",
            "height_scale": HEIGHT_SCALE,
            "min_raw": int(self.heights.min()),
            "max_raw": int(self.heights.max()),
            "content_sha256": self.content_sha256,
            "noise_layers": list(self.layer_metadata),
        }


def validate_height_range(values: Iterable[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        fail(ErrorCode.E_HEIGHT_RANGE, "height field is empty")
    if array.dtype.kind not in "iu":
        fail(ErrorCode.E_HEIGHT_RANGE, "height field must use an integer dtype", dtype=str(array.dtype))
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum > 65535:
        fail(
            ErrorCode.E_HEIGHT_RANGE,
            "height sample lies outside uint16 range",
            min_value=minimum,
            max_value=maximum,
            allowed_min=0,
            allowed_max=65535,
        )
    return array


def generate_heightfield(config: TerrainConfig) -> GeneratedTerrain:
    accumulator = np.zeros((config.size, config.size), dtype=np.int64)
    layer_metadata: list[dict[str, Any]] = []
    total_weight = sum(OCTAVE_WEIGHTS)
    for octave, (lattice_size, weight) in enumerate(zip(OCTAVE_LATTICES, OCTAVE_WEIGHTS)):
        lattice = _lattice(config.seed, octave, lattice_size)
        layer_hash = sha256_bytes(lattice.astype("<u4", copy=False).tobytes(order="C"))
        layer_metadata.append(
            {
                "octave": octave,
                "lattice_size": lattice_size,
                "weight": weight,
                "seed": int(config.seed),
                "lattice_sha256": layer_hash,
            }
        )
        accumulator += _value_noise(config.size, lattice, octave).astype(np.int64) * weight

    heights = np.clip(accumulator // total_weight, 0, 65535).astype("<u2")
    validate_height_range(heights)
    return GeneratedTerrain(config=config, heights=heights, layer_metadata=tuple(layer_metadata))


def terrain_paths(workspace_root: str | Path, config: TerrainConfig) -> tuple[Path, Path]:
    root = Path(workspace_root)
    stem = f"terrain_{config.size}_{int(config.seed):016x}"
    return root / "terrain" / f"{stem}.u16", root / "manifests" / f"{stem}.terrain.json"


def save_generated_terrain(
    workspace_root: str | Path,
    generated: GeneratedTerrain,
) -> tuple[Path, Path, str, str]:
    raw_path, meta_path = terrain_paths(workspace_root, generated.config)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = generated.heights.astype("<u2", copy=False).tobytes(order="C")
    raw_path.write_bytes(raw_bytes)
    metadata = generated.metadata()
    metadata["raw_file"] = raw_path.name
    meta_hash = write_json(meta_path, metadata)
    return raw_path, meta_path, sha256_bytes(raw_bytes), meta_hash


def load_terrain(
    workspace_root: str | Path,
    config: TerrainConfig,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path, meta_path = terrain_paths(workspace_root, config)
    if not raw_path.is_file() or not meta_path.is_file():
        fail(
            ErrorCode.E_BLOCK_MISSING,
            "terrain source is missing",
            size=config.size,
            seed=int(config.seed),
        )
    import json

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = raw_path.read_bytes()
    expected_bytes = config.size * config.size * 2
    if len(raw) != expected_bytes:
        fail(
            ErrorCode.E_HEIGHT_RANGE,
            "terrain byte length does not match configured size",
            expected_bytes=expected_bytes,
            actual_bytes=len(raw),
        )
    digest = hashlib.sha256(raw).hexdigest()
    if metadata.get("content_sha256") != digest:
        fail(ErrorCode.E_HEIGHT_RANGE, "terrain content hash does not match metadata")
    if expected_sha256 is not None and digest != expected_sha256:
        fail(ErrorCode.E_HEIGHT_RANGE, "terrain content hash does not match caller expectation")
    heights = np.frombuffer(raw, dtype="<u2").reshape(config.size, config.size).copy()
    validate_height_range(heights)
    return heights, metadata


def sample_bilinear(heights: np.ndarray, x: float, z: float) -> float:
    size = heights.shape[0]
    x = min(max(float(x), 0.0), float(size - 1))
    z = min(max(float(z), 0.0), float(size - 1))
    x0 = int(np.floor(x))
    z0 = int(np.floor(z))
    x1 = min(x0 + 1, size - 1)
    z1 = min(z0 + 1, size - 1)
    fx = x - x0
    fz = z - z0
    top = float(heights[z0, x0]) * (1.0 - fx) + float(heights[z0, x1]) * fx
    bottom = float(heights[z1, x0]) * (1.0 - fx) + float(heights[z1, x1]) * fx
    return (top * (1.0 - fz) + bottom * fz) * HEIGHT_SCALE
