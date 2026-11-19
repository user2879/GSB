from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .canonical import array_digest, canonical_json_bytes, sha256_bytes
from .errors import PipelineError
from .spec import Edit, VolumeSpec


@dataclass(frozen=True)
class VolumeData:
    spec: VolumeSpec
    array: np.ndarray
    base_digest: str

    @property
    def content_digest(self) -> str:
        return volume_content_digest(self.array, self.spec)


def _read_npy(path: Path) -> np.ndarray:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PipelineError("E_VOLUME", "volume.path", f"cannot load volume data: {exc}")
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise PipelineError("E_VOLUME", "volume.path", "volume file must contain one NumPy array")
    return loaded


def load_volume(spec: VolumeSpec) -> VolumeData:
    raw = _read_npy(spec.path)
    expected_shape = (spec.dims[2], spec.dims[1], spec.dims[0])
    if raw.shape != expected_shape:
        raise PipelineError(
            "E_DIMENSIONS",
            "volume.dims",
            "volume array shape does not match dims [nx, ny, nz]",
            details={
                "expected_shape_nz_ny_nx": list(expected_shape),
                "actual_shape": list(raw.shape),
            },
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise PipelineError("E_VOLUME", "volume.path", "volume array must be numeric")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise PipelineError("E_VOLUME", "volume.path", "volume array contains non-finite values")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < spec.scalar_min or maximum > spec.scalar_max:
        raise PipelineError(
            "E_VOLUME",
            "volume.scalar_domain",
            "volume scalar values are outside scalar_domain",
            details={
                "observed_min": minimum,
                "observed_max": maximum,
                "scalar_min": spec.scalar_min,
                "scalar_max": spec.scalar_max,
            },
        )
    array.flags.writeable = False
    return VolumeData(spec=spec, array=array, base_digest=volume_content_digest(array, spec))


def volume_content_digest(array: np.ndarray, spec: VolumeSpec) -> str:
    meta = canonical_json_bytes(
        {
            "dims": list(spec.dims),
            "spacing_mm": list(spec.spacing_mm),
            "scalar_domain": [spec.scalar_min, spec.scalar_max],
            "array_shape_nz_ny_nx": list(array.shape),
            "dtype": "<f8",
        }
    )
    return sha256_bytes(meta, array_digest(array).encode("ascii"))


def apply_edits(base: np.ndarray, edits: Sequence[Edit]) -> np.ndarray:
    """Apply edits front-to-back in declaration order without mutating the base."""

    if not edits:
        result = np.array(base, dtype=np.float64, order="C", copy=True)
    else:
        result = np.array(base, dtype=np.float64, order="C", copy=True)
        for edit in edits:
            x0, y0, z0 = edit.start
            sx, sy, sz = edit.size
            result[z0 : z0 + sz, y0 : y0 + sy, x0 : x0 + sx] = edit.value
    result.flags.writeable = False
    return result


def apply_normalized_edits(base: np.ndarray, edits: Sequence[dict[str, object]]) -> np.ndarray:
    parsed: list[Edit] = []
    for raw in edits:
        center_raw = raw["center"]
        size_raw = raw["size"]
        if not isinstance(center_raw, list) or not isinstance(size_raw, list):
            raise ValueError("invalid normalized edit")
        parsed.append(
            Edit(
                center=(int(center_raw[0]), int(center_raw[1]), int(center_raw[2])),
                size=(int(size_raw[0]), int(size_raw[1]), int(size_raw[2])),
                value=float(raw["value"]),
            )
        )
    return apply_edits(base, parsed)

