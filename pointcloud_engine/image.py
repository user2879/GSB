"""Row-order tile assembly, downsampling and atomic image output.

Outputs (all written through temp-file + os.replace):

* ``gray.pgm``  - 16-bit big-endian binary PGM (P5) grayscale image.
* ``depth.pfm`` - 32-bit little-endian portable float map depth image.
* ``gray.npy`` / ``depth.npy`` - exact float32 arrays for programmatic checks.
* ``outputs.json`` - dimensions, digest and the engine version id.
"""

from __future__ import annotations

import os

import numpy as np

from .errors import EngineError
from .io_utils import atomic_write_bytes, atomic_write_json, hash_array, read_json
from .scene import Scene
from .state import TileRecord

OUTPUTS_META = "outputs.json"


def assemble_grids(scene: Scene, records: dict[int, TileRecord]):
    grid_gray = np.zeros((scene.grid_height, scene.grid_width), dtype=np.float32)
    grid_depth = np.zeros((scene.grid_height, scene.grid_width), dtype=np.float32)
    for tile_index, record in sorted(records.items()):
        ty = tile_index // scene.tiles_x
        tx = tile_index % scene.tiles_x
        y0 = ty * scene.tile_height
        x0 = tx * scene.tile_width
        grid_gray[y0:y0 + scene.tile_height, x0:x0 + scene.tile_width] = record.gray
        grid_depth[y0:y0 + scene.tile_height, x0:x0 + scene.tile_width] = record.depth
    return grid_gray, grid_depth


def write_pgm(path: str, gray: np.ndarray) -> None:
    clipped = np.clip(gray, 0.0, 1.0)
    sixteen = np.rint(clipped * 65535.0).astype(">u2")
    header = f"P5\n{gray.shape[1]} {gray.shape[0]}\n65535\n".encode("ascii")
    atomic_write_bytes(path, header + np.ascontiguousarray(sixteen).tobytes(order="C"))


def write_pfm(path: str, depth: np.ndarray) -> None:
    h, w = depth.shape
    # Negative scale means little-endian for the portable float map format.
    header = f"PF\n{w} {h}\n-1.0\n".encode("ascii")
    flat = np.ascontiguousarray(depth.astype("<f4"), dtype="<f4")
    atomic_write_bytes(path, header + flat.tobytes(order="C"))


def read_outputs_meta(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, OUTPUTS_META)
    if not os.path.isfile(path):
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def write_outputs(scene: Scene, gray: np.ndarray, depth: np.ndarray, version_id: str) -> dict:
    os.makedirs(scene.output_dir, exist_ok=True)
    existing = read_outputs_meta(scene.output_dir)
    if existing is not None and existing.get("version_id") == version_id:
        # Same version: idempotent run, do not churn identical files.
        return {"replaced": False, "reason": "same version already present", **existing}

    dimensions = {"height": int(gray.shape[0]), "width": int(gray.shape[1])}
    if existing is not None and existing.get("dimensions") != dimensions:
        # Version differs and geometry differs - still replace atomically.
        pass

    gray_pgm = os.path.join(scene.output_dir, "gray.pgm")
    depth_pfm = os.path.join(scene.output_dir, "depth.pfm")
    gray_npy_tmp = os.path.join(scene.output_dir, "gray.npy")
    depth_npy_tmp = os.path.join(scene.output_dir, "depth.npy")

    write_pgm(gray_pgm, gray)
    write_pfm(depth_pfm, depth)
    _atomic_write_npy(gray_npy_tmp, gray)
    _atomic_write_npy(depth_npy_tmp, depth)

    meta = {
        "version_id": version_id,
        "dimensions": dimensions,
        "gray_digest": hash_array(gray),
        "depth_digest": hash_array(depth),
        "files": ["gray.pgm", "depth.pfm", "gray.npy", "depth.npy"],
    }
    atomic_write_json(os.path.join(scene.output_dir, OUTPUTS_META), meta)
    return {"replaced": True, **meta}


def _atomic_write_npy(path: str, array: np.ndarray) -> None:
    from .io_utils import atomic_write_bytes

    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    atomic_write_bytes(path, buffer.getvalue())


def verify_pair_same_version(output_dir: str) -> None:
    meta = read_outputs_meta(output_dir)
    if meta is None:
        raise EngineError("E_IO", "outputs metadata missing after write", output_dir=output_dir)
    required = ["gray.pgm", "depth.pfm", "gray.npy", "depth.npy"]
    for name in required:
        if not os.path.isfile(os.path.join(output_dir, name)):
            raise EngineError("E_IO", f"expected output file missing: {name}", file=name)
