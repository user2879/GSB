"""Field-level and array-level equivalence checks used by acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .util import array_fingerprint, read_npz


def comparable_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project a run report onto state content independent of cache history."""

    diagnostics = []
    for item in report.get("diagnostics", []):
        normalized = dict(item)
        normalized.pop("detail_npz", None)
        diagnostics.append(normalized)
    artifacts = {}
    for kind in ("kernels", "tiles", "composites"):
        entries = []
        for item in report.get("artifacts", {}).get(kind, []):
            normalized = dict(item)
            normalized.pop("artifact_id", None)
            entries.append(normalized)
        artifacts[kind] = entries
    return {
        "active_views": report.get("active_views", []),
        "artifacts": artifacts,
        "camera_hash": report.get("camera_hash"),
        "config_fingerprint": report.get("config_fingerprint"),
        "diagnostics": diagnostics,
        "failures": report.get("failures", []),
        "key_manifest": report.get("key_manifest", {}),
    }


def compare_reports(left: dict[str, Any], right: dict[str, Any], *, compare_diagnostics: bool = True) -> dict[str, Any]:
    left_view = comparable_report(left)
    right_view = comparable_report(right)
    checks: dict[str, bool] = {}
    for key in ("active_views", "artifacts", "camera_hash", "config_fingerprint", "failures", "key_manifest"):
        checks[key] = left_view[key] == right_view[key]
    if compare_diagnostics:
        checks["diagnostics"] = left_view["diagnostics"] == right_view["diagnostics"]
    return {
        "checks": checks,
        "equal": all(checks.values()),
        "mismatches": [key for key, value in checks.items() if not value],
    }


def compare_cache_arrays(work_left: Path, work_right: Path) -> dict[str, Any]:
    left_root = Path(work_left) / "cache"
    right_root = Path(work_right) / "cache"
    checks: dict[str, bool] = {}
    mismatches: list[str] = []
    for section in ("kernels", "tiles", "composites"):
        left_files = {
            path.relative_to(left_root): path
            for path in sorted((left_root / section).glob("*.npz"), key=lambda item: item.name)
        }
        right_files = {
            path.relative_to(right_root): path
            for path in sorted((right_root / section).glob("*.npz"), key=lambda item: item.name)
        }
        section_ok = set(left_files) == set(right_files)
        if not section_ok:
            mismatches.append(f"{section}: file set differs")
        for relative in sorted(set(left_files) & set(right_files), key=str):
            left_arrays = read_npz(left_files[relative])
            right_arrays = read_npz(right_files[relative])
            if set(left_arrays) != set(right_arrays):
                section_ok = False
                mismatches.append(f"{relative}: array names differ")
                continue
            for name in sorted(left_arrays):
                if not np.array_equal(left_arrays[name], right_arrays[name]):
                    section_ok = False
                    mismatches.append(f"{relative}:{name}")
                    break
        checks[section] = section_ok
    return {"checks": checks, "equal": all(checks.values()), "mismatches": mismatches[:50]}


def cache_digest(work_dir: Path) -> str:
    from .util import hash_value

    root = Path(work_dir) / "cache"
    payload: dict[str, Any] = {}
    for section in ("kernels", "tiles", "composites"):
        entries = {}
        for path in sorted((root / section).glob("*.npz"), key=lambda item: item.name):
            arrays = read_npz(path)
            entries[path.stem] = {name: array_fingerprint(value) for name, value in sorted(arrays.items())}
        payload[section] = entries
    return hash_value(payload)
