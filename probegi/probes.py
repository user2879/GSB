"""Deterministic probe-grid generation, sampling and binary persistence.

Binary layout (little-endian, fixed order, no padding, no timestamps) so a
given (grid_n, seed, n_directions) triple always produces identical bytes::

    magic      8s   b"PGIPROB1"
    grid_n     <u4
    n_dirs     <u4
    seed       <u8
    bbox_min   3 x <f8
    bbox_max   3 x <f8
    positions  grid_n^3 x 3 x <f8   (ijk order, x fastest)
    directions n_dirs  x 3 x <f8
    radiance   grid_n^3 x n_dirs x 3 x <f4
"""
import json
import os
import struct

import numpy as np

from .environment import LightingEnvironment
from .errors import ProbeGridError
from .hashing import canonical_json, sha256_bytes
from .sampling import fibonacci_sphere

MAGIC = b"PGIPROB1"
GRID_SIZES = (8, 16, 32)
DEFAULT_BBOX = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
DEFAULT_DIRECTIONS = 256

_HEADER = struct.Struct("<8sIIQ6d")


def check_grid(grid_n: int) -> int:
    grid_n = int(grid_n)
    if grid_n not in GRID_SIZES:
        raise ProbeGridError("illegal probe grid", grid_n=grid_n,
                             allowed=list(GRID_SIZES))
    return grid_n


def grid_positions(grid_n: int, bbox=DEFAULT_BBOX) -> np.ndarray:
    """(P, 3) probe positions, ijk order with x fastest."""
    check_grid(grid_n)
    lo, hi = (np.array(b, dtype=np.float64) for b in bbox)
    axes = [np.linspace(lo[a], hi[a], grid_n) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def generate(grid_n: int, seed: int, n_directions: int = DEFAULT_DIRECTIONS):
    """Generate positions, directions and sampled radiance (all deterministic)."""
    check_grid(grid_n)
    positions = grid_positions(grid_n)
    directions = fibonacci_sphere(int(n_directions))
    env = LightingEnvironment(seed)
    radiance = env.radiance_at(positions, directions)
    return positions, directions, radiance.astype("<f4"), env


def encode(grid_n: int, seed: int, n_directions: int, positions, directions,
           radiance, bbox=DEFAULT_BBOX) -> bytes:
    header = _HEADER.pack(MAGIC, grid_n, n_directions, seed,
                          *bbox[0], *bbox[1])
    return (header
            + np.ascontiguousarray(positions, dtype="<f8").tobytes()
            + np.ascontiguousarray(directions, dtype="<f8").tobytes()
            + np.ascontiguousarray(radiance, dtype="<f4").tobytes())


def decode(data: bytes):
    (magic, grid_n, n_dirs, seed, *bbox) = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ProbeGridError("bad probe file magic", magic=magic.hex())
    check_grid(grid_n)
    off = [_HEADER.size]

    def take(dtype, shape):
        arr = np.frombuffer(data, dtype=dtype,
                            count=int(np.prod(shape)), offset=off[0])
        off[0] += arr.nbytes
        return arr.reshape(shape)

    positions = take("<f8", (grid_n ** 3, 3))
    directions = take("<f8", (n_dirs, 3))
    radiance = take("<f4", (grid_n ** 3, n_dirs, 3))
    bbox = ((bbox[0], bbox[1], bbox[2]), (bbox[3], bbox[4], bbox[5]))
    return {"grid_n": grid_n, "n_directions": n_dirs, "seed": seed,
            "bbox": bbox, "positions": positions, "directions": directions,
            "radiance": radiance}


def energy_stats(radiance: np.ndarray) -> dict:
    lum = radiance.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    return {
        "mean_luminance": float(lum.mean()),
        "max_luminance": float(lum.max()),
        "total_energy": float(lum.sum()),
    }


def build_manifest(grid_n: int, seed: int, n_directions: int, blob: bytes,
                   radiance: np.ndarray, bbox=DEFAULT_BBOX) -> dict:
    return {
        "format": "PGIPROB1",
        "grid_n": grid_n,
        "seed": seed,
        "probe_count": grid_n ** 3,
        "bbox": [list(bbox[0]), list(bbox[1])],
        "n_directions": n_directions,
        "sha256": sha256_bytes(blob),
        "energy": energy_stats(radiance),
    }


def generate_to_file(path: str, grid_n: int, seed: int,
                     n_directions: int = DEFAULT_DIRECTIONS) -> dict:
    """Generate a probe file atomically; returns the manifest."""
    positions, directions, radiance, _env = generate(grid_n, seed, n_directions)
    blob = encode(grid_n, seed, n_directions, positions, directions, radiance)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    manifest = build_manifest(grid_n, seed, n_directions, blob, radiance)
    with open(path + ".manifest.json", "w", encoding="utf-8") as fh:
        fh.write(canonical_json(manifest))
    return manifest


def load_probe_file(path: str) -> dict:
    with open(path, "rb") as fh:
        return decode(fh.read())
