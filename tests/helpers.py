from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voicecal.engine import EnginePaths
from voicecal.util import read_json, write_json

SAMPLES = ROOT / "samples"


def write_json_file(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return path


def sample_config(tmp: Path, *, budget: int | None = None, concurrency: int | None = None,
                  min_segment_ms: float | None = None) -> Path:
    config = copy.deepcopy(read_json(SAMPLES / "calibration.json"))
    if budget is not None:
        config["batch_budget"] = budget
    if concurrency is not None:
        config["concurrency"] = concurrency
    if min_segment_ms is not None:
        config["attribution"]["min_segment_ms"] = min_segment_ms
    return write_json_file(tmp / "calibration.json", config)


def engine_paths(
    tmp: Path,
    *,
    config: Path | None = None,
    edits: Path | None = None,
    injections: Path | None = None,
    name: str = "run",
) -> EnginePaths:
    return EnginePaths(
        sample_dir=SAMPLES,
        dialogue_path=SAMPLES / "dialogue.json",
        config_path=config or SAMPLES / "calibration.json",
        edits_path=edits if edits is not None else SAMPLES / "edits.json",
        injections_path=injections,
        snapshot_dir=tmp / name / "snapshot",
        output_dir=tmp / name / "output",
    )


def canonical_files() -> tuple[str, ...]:
    return (
        "initial_evidence.json",
        "calibration.json",
        "attribution.json",
        "calibration_ledger.json",
        "failures.json",
    )
