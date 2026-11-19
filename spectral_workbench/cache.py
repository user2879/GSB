"""Content-addressed per-stage frame cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .calibration import CalibrationResult
from .io_utils import atomic_directory, read_array, read_json, sha256_array, write_array, write_json
from .metrics import FrameMetrics
from .unmix import UnmixResult


def stage_dir(cache_root: Path, stage: str, key: str) -> Path:
    return cache_root / stage / key


@dataclass(frozen=True)
class CacheLookup:
    source: str
    version: str
    path: Path


def save_calibration(cache_root: Path, key: str, result: CalibrationResult) -> Path:
    target = stage_dir(cache_root, "calibration", key)
    with atomic_directory(target) as temporary:
        write_array(temporary / "corrected.f32", result.corrected, "<f4")
        write_array(temporary / "invalid.u1", result.invalid_mask, "u1")
        write_json(temporary / "record.json", result.record)
    return target


def load_calibration(cache_root: Path, key: str) -> Optional[Tuple[CalibrationResult, CacheLookup]]:
    target = stage_dir(cache_root, "calibration", key)
    record_path = target / "record.json"
    corrected_path = target / "corrected.f32"
    invalid_path = target / "invalid.u1"
    if not (record_path.is_file() and corrected_path.is_file() and invalid_path.is_file()):
        return None
    record = read_json(record_path)
    shape = tuple(int(value) for value in record["shape"])
    corrected = read_array(corrected_path, "<f4", shape)
    invalid = read_array(invalid_path, "u1", shape)
    if sha256_array(corrected) != record["corrected_sha256"] or sha256_array(invalid) != record["invalid_mask_sha256"]:
        return None
    result = CalibrationResult(
        corrected=corrected,
        invalid_mask=invalid,
        saturated_pixels=int(record["saturated_pixels"]),
        zero_flat_pixels=int(record["zero_flat_pixels"]),
        invalid_pixels=int(record["invalid_pixels"]),
        quantization_upper_bound=float(record["quantization_upper_bound_dn"]),
        quantization_mean_bound=float(record["quantization_mean_bound_dn"]),
        gain_min=float(record["gain_min"]),
        gain_max=float(record["gain_max"]),
        corrected_sha256=str(record["corrected_sha256"]),
        invalid_sha256=str(record["invalid_mask_sha256"]),
        record=record,
    )
    return result, CacheLookup(source="reused", version=key, path=target)


def save_unmix(cache_root: Path, key: str, result: UnmixResult) -> Path:
    target = stage_dir(cache_root, "unmix", key)
    with atomic_directory(target) as temporary:
        write_array(temporary / "concentrations.f32", result.concentrations, "<f4")
        write_array(temporary / "residual.f32", result.residual, "<f4")
        write_array(temporary / "active.u1", result.active_subset, "u1")
        write_array(temporary / "ill_conditioned.u1", result.ill_conditioned, "u1")
        write_array(temporary / "invalid.u1", result.invalid_mask, "u1")
        write_array(temporary / "signal_norm.f32", result.signal_norm, "<f4")
        write_json(temporary / "record.json", result.record)
    return target


def load_unmix(cache_root: Path, key: str) -> Optional[Tuple[UnmixResult, CacheLookup]]:
    target = stage_dir(cache_root, "unmix", key)
    record_path = target / "record.json"
    if not record_path.is_file():
        return None
    record = read_json(record_path)
    order, height, width = (int(value) for value in record["shape"])
    expected_files = (
        "concentrations.f32",
        "residual.f32",
        "active.u1",
        "ill_conditioned.u1",
        "invalid.u1",
        "signal_norm.f32",
    )
    if any(not (target / name).is_file() for name in expected_files):
        return None
    concentrations = read_array(target / "concentrations.f32", "<f4", (order, height, width))
    residual = read_array(target / "residual.f32", "<f4", (height, width))
    active = read_array(target / "active.u1", "u1", (height, width))
    ill = read_array(target / "ill_conditioned.u1", "u1", (height, width))
    invalid = read_array(target / "invalid.u1", "u1", (height, width))
    signal_norm = read_array(target / "signal_norm.f32", "<f4", (height, width)).astype(np.float64)
    checks = {
        "concentrations_sha256": sha256_array(np.ascontiguousarray(concentrations, dtype="<f4")),
        "residual_sha256": sha256_array(np.ascontiguousarray(residual, dtype="<f4")),
        "active_subset_sha256": sha256_array(np.ascontiguousarray(active, dtype="u1")),
        "ill_conditioned_sha256": sha256_array(np.ascontiguousarray(ill, dtype="u1")),
        "signal_norm_sha256": sha256_array(np.ascontiguousarray(signal_norm.astype(np.float32), dtype="<f4")),
    }
    if any(record[key_name] != value for key_name, value in checks.items()):
        return None
    result = UnmixResult(
        concentrations=concentrations,
        residual=residual,
        active_subset=active,
        ill_conditioned=ill,
        invalid_mask=invalid,
        signal_norm=signal_norm,
        condition_number=float(record["condition_number"]),
        ill_conditioned_pixels=int(record["ill_conditioned_pixels"]),
        concentrations_sha256=record["concentrations_sha256"],
        residual_sha256=record["residual_sha256"],
        active_sha256=record["active_subset_sha256"],
        ill_conditioned_sha256=record["ill_conditioned_sha256"],
        record=record,
    )
    return result, CacheLookup(source="reused", version=key, path=target)


def save_metrics(cache_root: Path, key: str, metrics: FrameMetrics) -> Path:
    target = stage_dir(cache_root, "metrics", key)
    with atomic_directory(target) as temporary:
        for k, values in enumerate(metrics.absolute_errors):
            write_array(temporary / f"absolute_error_{k}.f32", values, "<f4")
        for k, values in enumerate(metrics.uncertainty_bounds):
            write_array(temporary / f"uncertainty_{k}.f32", values, "<f4")
        write_json(
            temporary / "record.json",
            {
                "per_fluorophore": list(metrics.per_fluorophore),
                "first_violation": metrics.first_violation,
                "evaluated_pixels": metrics.evaluated_pixels,
            },
        )
    return target


def load_metrics(cache_root: Path, key: str) -> Optional[Tuple[FrameMetrics, CacheLookup]]:
    target = stage_dir(cache_root, "metrics", key)
    record_path = target / "record.json"
    if not record_path.is_file():
        return None
    record = read_json(record_path)
    count = len(record["per_fluorophore"])
    errors = []
    bounds = []
    try:
        for k in range(count):
            errors.append(_read_vector(target / f"absolute_error_{k}.f32"))
            bounds.append(_read_vector(target / f"uncertainty_{k}.f32"))
    except (OSError, ValueError):
        return None
    metrics = FrameMetrics(
        per_fluorophore=tuple(record["per_fluorophore"]),
        absolute_errors=tuple(errors),
        uncertainty_bounds=tuple(bounds),
        first_violation=record["first_violation"],
        evaluated_pixels=int(record["evaluated_pixels"]),
    )
    return metrics, CacheLookup(source="reused", version=key, path=target)


def _read_vector(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"invalid float32 vector size: {path}")
    return np.frombuffer(raw, dtype="<f4").copy()


def prune_stage(cache_root: Path, stage: str, active_keys: set[str]) -> Dict[str, Any]:
    root = cache_root / stage
    removed = 0
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name not in active_keys:
                import shutil

                shutil.rmtree(child)
                removed += 1
    return {"stage": stage, "active_versions": len(active_keys), "removed_versions": removed}
