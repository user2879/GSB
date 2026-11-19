"""High-level entry points used by the CLI and the test harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import SegmentationConfig, load_config
from .engine import IncrementalEngine
from .errors import BusinessError, FailureRecord
from .injection import apply_index_injection, load_injections
from .runner_types import RunInputs
from .snapshot import SnapshotManager
from .tiles import _parse_index, load_index


def build_inputs(sample_dir: str | Path, work_dir: str | Path,
                 *, patches_dir: str | Path | None = None,
                 injections_path: str | Path | None = None,
                 config_override: dict[str, Any] | None = None,
                 index_override: dict | None = None) -> RunInputs:
    sample_dir = Path(sample_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(sample_dir / "segmentation_config.json")
    if config_override:
        merged = _deep_merge(config.to_dict(), config_override)
        config = SegmentationConfig.from_dict(merged)

    injections = load_injections(injections_path) if injections_path else None
    index_path = sample_dir / "tile_index.json"
    if index_override is not None:
        index = _parse_index(index_override, injection_id=None, base_dir=sample_dir)
    elif injections is not None and injections.for_index() is not None:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        injected = apply_index_injection(raw, injections.for_index())
        index = _parse_index(injected, injection_id=injections.for_index().injection_id,
                             base_dir=sample_dir)
    else:
        index = load_index(index_path)
    return RunInputs(
        sample_dir=sample_dir,
        work_dir=work_dir,
        config=config,
        index=index,
        index_path=index_path,
        patches_dir=Path(patches_dir) if patches_dir else None,
        injections=injections,
    )


def run_incremental(sample_dir: str | Path, work_dir: str | Path, **kwargs) -> tuple[RunInputs, Any]:
    inputs = build_inputs(sample_dir, work_dir, **kwargs)
    engine = IncrementalEngine(inputs)
    report = engine.run()
    return inputs, report


def run_full_rebuild(sample_dir: str | Path, work_dir: str | Path, **kwargs) -> tuple[RunInputs, Any]:
    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)
    inputs = build_inputs(sample_dir, work_dir, **kwargs)
    engine = IncrementalEngine(inputs)
    report = engine.run(force_full=True)
    return inputs, report


def load_latest_grid(work_dir: str | Path) -> np.ndarray | None:
    info = SnapshotManager(Path(work_dir)).latest()
    if info is None:
        return None
    return np.load(info.path / "global_grid.npy")


def _deep_merge(base: dict, override: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_index_or_report(sample_dir: str | Path, injections_path: str | Path | None) -> FailureRecord | None:
    """Standalone index validation used by injection tests."""

    injections = load_injections(injections_path) if injections_path else None
    index_path = Path(sample_dir) / "tile_index.json"
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    injection = injections.for_index() if injections else None
    if injection is not None:
        raw = apply_index_injection(raw, injection)
    try:
        _parse_index(raw, injection_id=injection.injection_id if injection else None,
                     base_dir=Path(sample_dir))
    except BusinessError as exc:
        return FailureRecord(
            code=exc.code,
            target="index",
            message=exc.message,
            sort_key=(-1, -1),
            injection_id=exc.injection_id,
            detail=exc.detail,
        )
    return None
