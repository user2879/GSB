"""Immutable, versioned diagnostic snapshots and their render gate."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .errors import ErrorCode, WorkbenchError
from .io_utils import atomic_directory, canonical_json, read_array, read_json, sha256_array, write_array, write_json
from .pipeline import PipelineResult


_ARRAY_FIELDS = {
    "band_positions": "<f8",
    "concentration_curves": "<f4",
    "error_low": "<f4",
    "error_high": "<f4",
    "ill_conditioned_bands": "u1",
    "ill_conditioned_mask": "u1",
    "pseudocolor": "u1",
}


@dataclass(frozen=True)
class DiagnosticSnapshot:
    version: int
    content_hash: str
    source_run_hash: str
    band_positions: np.ndarray
    concentration_curves: np.ndarray
    error_low: np.ndarray
    error_high: np.ndarray
    ill_conditioned_bands: np.ndarray
    ill_conditioned_mask: np.ndarray
    pseudocolor: np.ndarray
    fluorophore_names: tuple[str, ...]
    mean_concentration_curves: np.ndarray
    summary_json: str

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "content_hash": self.content_hash,
            "source_run_hash": self.source_run_hash,
            "fluorophore_names": list(self.fluorophore_names),
            "summary_json": self.summary_json,
        }


def _readonly(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array).copy()
    contiguous.setflags(write=False)
    return contiguous


def snapshot_content_hash(snapshot: DiagnosticSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "version": int(snapshot.version),
                "source_run_hash": snapshot.source_run_hash,
                "fluorophore_names": list(snapshot.fluorophore_names),
                "summary_json": snapshot.summary_json,
            }
        ).encode("utf-8")
    )
    digest.update(b"\n")
    for name in _ARRAY_FIELDS:
        array = np.asarray(getattr(snapshot, name))
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(sha256_array(array).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_snapshot(result: PipelineResult, version: int) -> DiagnosticSnapshot:
    if version < 1:
        raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "snapshot version must be a positive integer")
    band_count = result.report["cube"]["shape"][0]
    order, height, width = result.post_unmix.concentrations.shape
    positions = np.linspace(0.0, 1.0, band_count, dtype=np.float64)
    curves = np.zeros((order, band_count), dtype=np.float32)
    low = np.zeros_like(curves)
    high = np.zeros_like(curves)
    ill_bands = np.zeros((band_count,), dtype=np.uint8)
    for band, metrics in enumerate(result.post_frame_metrics):
        for k, record in enumerate(metrics.per_fluorophore):
            mean = record["concentration_mean"]
            uncertainty = record["uncertainty_mean"]
            if mean is None:
                mean = 0.0
            if uncertainty is None:
                uncertainty = 0.0
            curves[k, band] = float(mean)
            low[k, band] = float(mean) - float(uncertainty)
            high[k, band] = float(mean) + float(uncertainty)
        ill_bands[band] = np.uint8(result.post_unmix.ill_conditioned_pixels > 0)

    pseudocolor = _pseudocolor(result.post_unmix.concentrations, result.post_unmix.invalid_mask)
    summary = {
        "condition_number": result.model.condition_number,
        "condition_ill_conditioned": result.model.ill_conditioned,
        "compensation_enabled": result.config.drift.compensate,
        "compensation_evaluation": result.report["metrics"]["compensation_evaluation"],
        "warnings": result.report["warnings"],
        "run_hash": result.report["run_hash"],
    }
    snapshot = DiagnosticSnapshot(
        version=int(version),
        content_hash="",
        source_run_hash=str(result.report["run_hash"]),
        band_positions=_readonly(positions),
        concentration_curves=_readonly(curves),
        error_low=_readonly(low),
        error_high=_readonly(high),
        ill_conditioned_bands=_readonly(ill_bands),
        ill_conditioned_mask=_readonly(result.post_unmix.ill_conditioned.astype(np.uint8)),
        pseudocolor=_readonly(pseudocolor),
        fluorophore_names=tuple(result.model.names),
        mean_concentration_curves=_readonly(curves.copy()),
        summary_json=canonical_json(summary),
    )
    content_hash = snapshot_content_hash(snapshot)
    final = DiagnosticSnapshot(
        version=snapshot.version,
        content_hash=content_hash,
        source_run_hash=snapshot.source_run_hash,
        band_positions=snapshot.band_positions,
        concentration_curves=snapshot.concentration_curves,
        error_low=snapshot.error_low,
        error_high=snapshot.error_high,
        ill_conditioned_bands=snapshot.ill_conditioned_bands,
        ill_conditioned_mask=snapshot.ill_conditioned_mask,
        pseudocolor=snapshot.pseudocolor,
        fluorophore_names=snapshot.fluorophore_names,
        mean_concentration_curves=snapshot.mean_concentration_curves,
        summary_json=snapshot.summary_json,
    )
    if snapshot_content_hash(final) != content_hash:
        raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "snapshot hash is not stable")
    return final


def _pseudocolor(concentrations: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [1.00, 0.16, 0.12],
            [0.10, 0.92, 0.28],
            [0.12, 0.38, 1.00],
            [1.00, 0.78, 0.08],
            [0.82, 0.18, 0.92],
            [0.10, 0.88, 0.88],
            [1.00, 0.48, 0.08],
            [0.78, 0.78, 0.82],
        ],
        dtype=np.float64,
    )
    order, height, width = concentrations.shape
    weights = np.zeros_like(concentrations, dtype=np.float64)
    for k in range(order):
        percentile = float(np.percentile(concentrations[k], 99.0))
        scale = max(percentile, 1.0e-8)
        weights[k] = np.maximum(concentrations[k].astype(np.float64), 0.0) / scale
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    for k in range(order):
        rgb += weights[k, :, :, None] * palette[k % palette.shape[0]][None, None, :]
    peak = float(np.max(rgb)) if rgb.size else 0.0
    if peak > 0.0:
        rgb = rgb / peak
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))
    rgb[invalid_mask.astype(bool)] = 0.0
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def save_snapshot(snapshot: DiagnosticSnapshot, directory: str | Path) -> Path:
    target = Path(directory)
    with atomic_directory(target) as temporary:
        array_meta = {}
        for name, dtype in _ARRAY_FIELDS.items():
            array = np.asarray(getattr(snapshot, name))
            filename = f"{name}.bin"
            digest = write_array(temporary / filename, array, dtype)
            array_meta[name] = {
                "file": filename,
                "shape": [int(value) for value in array.shape],
                "dtype": dtype,
                "sha256": digest,
            }
        write_json(
            temporary / "snapshot.json",
            {**snapshot.metadata_dict(), "arrays": array_meta},
        )
    return target


def load_snapshot(directory: str | Path) -> DiagnosticSnapshot:
    root = Path(directory)
    metadata_path = root / "snapshot.json"
    if not metadata_path.is_file():
        raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "snapshot metadata is missing", {"path": str(root)})
    metadata = read_json(metadata_path)
    arrays: Dict[str, np.ndarray] = {}
    try:
        for name, dtype in _ARRAY_FIELDS.items():
            item = metadata["arrays"][name]
            shape = tuple(int(value) for value in item["shape"])
            arrays[name] = read_array(root / item["file"], dtype, shape)
    except (KeyError, OSError, ValueError) as exc:
        raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "snapshot array is missing or malformed", {"error": str(exc)}) from exc
    snapshot = DiagnosticSnapshot(
        version=int(metadata["version"]),
        content_hash=str(metadata["content_hash"]),
        source_run_hash=str(metadata["source_run_hash"]),
        band_positions=_readonly(arrays["band_positions"]),
        concentration_curves=_readonly(arrays["concentration_curves"]),
        error_low=_readonly(arrays["error_low"]),
        error_high=_readonly(arrays["error_high"]),
        ill_conditioned_bands=_readonly(arrays["ill_conditioned_bands"]),
        ill_conditioned_mask=_readonly(arrays["ill_conditioned_mask"]),
        pseudocolor=_readonly(arrays["pseudocolor"]),
        fluorophore_names=tuple(str(value) for value in metadata["fluorophore_names"]),
        mean_concentration_curves=_readonly(arrays["concentration_curves"].copy()),
        summary_json=str(metadata["summary_json"]),
    )
    if snapshot_content_hash(snapshot) != snapshot.content_hash:
        raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "snapshot content hash mismatch", {"path": str(root)})
    return snapshot


class SnapshotStore:
    """Single-current-snapshot store; readers must match version and content hash."""

    def __init__(self) -> None:
        self._current: Optional[DiagnosticSnapshot] = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Optional[DiagnosticSnapshot]:
        with self._lock:
            return self._current

    def publish(self, snapshot: DiagnosticSnapshot) -> DiagnosticSnapshot:
        with self._lock:
            if snapshot_content_hash(snapshot) != snapshot.content_hash:
                raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "refusing to publish snapshot with invalid hash")
            if self._current is not None and snapshot.version <= self._current.version:
                raise WorkbenchError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "snapshot version must increase on replacement",
                    {"current_version": self._current.version, "incoming_version": snapshot.version},
                )
            self._current = snapshot
            return snapshot

    def validate_render(self, snapshot: DiagnosticSnapshot) -> None:
        with self._lock:
            if self._current is None:
                raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "no current snapshot is published")
            if snapshot_content_hash(snapshot) != snapshot.content_hash:
                raise WorkbenchError(ErrorCode.E_SNAPSHOT_STALE, "render candidate hash is invalid", {"version": snapshot.version})
            if (
                snapshot.version != self._current.version
                or snapshot.content_hash != self._current.content_hash
            ):
                raise WorkbenchError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "render candidate is not the current snapshot",
                    {
                        "candidate_version": snapshot.version,
                        "current_version": self._current.version,
                        "candidate_hash": snapshot.content_hash,
                        "current_hash": self._current.content_hash,
                    },
                )

    def publish_directory(self, directory: str | Path) -> DiagnosticSnapshot:
        return self.publish(load_snapshot(directory))
