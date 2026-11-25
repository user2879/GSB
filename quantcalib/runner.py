"""Incremental command-stream runner and final-state full rebuild."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from quantcalib.data.config import QuantConfig, load_config, write_config
from quantcalib.engine import EngineContext, QuantCalibEngine
from quantcalib.schedule.commands import apply_command, load_command_stream


def _load_cfg(workdir: Path) -> QuantConfig:
    cfg, issues = load_config(workdir / "config" / "quant.yaml")
    if issues:
        raise ValueError(issues)
    return cfg


def run_incremental(workdir: str | Path,
                    command_stream: str | Path) -> list[dict[str, Any]]:
    workdir = Path(workdir)
    commands = load_command_stream(command_stream)
    results: list[dict[str, Any]] = []

    # Version 0: baseline full run.
    cfg = _load_cfg(workdir)
    context = EngineContext(workdir=workdir)
    engine = QuantCalibEngine(context, cfg)
    results.append(engine.run(global_version=0))
    write_config(workdir / "config" / "quant.yaml", cfg)

    for command in commands:
        cfg = _load_cfg(workdir)
        apply_command(command, workdir, cfg)
        write_config(workdir / "config" / "quant.yaml", cfg)
        engine = QuantCalibEngine(EngineContext(workdir=workdir), cfg)
        result = engine.run(global_version=command.seq + 1)
        result["command"] = {"name": command.name, "payload": command.payload}
        results.append(result)
    return results


def run_full_rebuild(workdir: str | Path,
                     isolated_snapshot: str | Path | None = None) -> dict[str, Any]:
    """One-shot rebuild from an empty snapshot at the final on-disk state.

    When ``isolated_snapshot`` is given, the live snapshot is copied aside and
    the rebuild writes into an isolated directory so the incremental workspace
    is not touched.
    """
    workdir = Path(workdir)
    cfg = _load_cfg(workdir)
    context = EngineContext(workdir=workdir)
    if isolated_snapshot is not None:
        context.snapshot_dir = str(Path(isolated_snapshot))
    else:
        snapshot = workdir / "out" / "snapshot"
        if snapshot.exists():
            import shutil
            shutil.rmtree(snapshot)
    engine = QuantCalibEngine(context, cfg)
    return engine.run(global_version=0)


def comparable_view(result: dict[str, Any]) -> dict[str, Any]:
    """Strip scheduling/thread metadata for field-level equality checks."""
    return {
        "layers": [
            {
                "layer": layer["layer"],
                "bits": layer["bits"],
                "group_granularity": layer["group_granularity"],
                "groups": [
                    {
                        "group_index": group["group_index"],
                        "calibration": group["calibration"],
                        "quantization": {
                            key: group["quantization"][key]
                            for key in (
                                "integer_codes", "dequantized", "clipped",
                                "abs_errors", "rel_errors", "over_tolerance",
                                "error_sources", "source_evidence",
                                "saturation_count",
                            )
                        },
                    }
                    for group in layer["groups"]
                ],
            }
            for layer in result["layers"]
        ],
        "distillation": result["distillation"],
        "ledger": result["ledger"],
    }
