"""End-to-end deterministic demo: cold start, fault injection, crash recovery."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .engine import PATH_FULL, PATH_INCREMENTAL
from .injection import (
    FAULT_BAD_RATIO,
    FAULT_CORRUPT_SCENE,
    FAULT_MISSING_SNAPSHOT_VERSION,
    StructuredLogger,
    exercise_fault,
    strip_snapshot_version,
)
from .ledger import Ledger, write_snapshot
from .loader import load_scene
from .samplegen import write_samples
from .session import assert_paths_identical, crash_and_recover, run_and_persist


def run_demo(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace)
    sample_dir = root / "samples"
    snapshot_dir = root / "snapshots"
    data_dir = root / "data"
    for directory in (sample_dir, snapshot_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "structured.log.jsonl"
    logger = StructuredLogger(log_path)

    write_samples(sample_dir)
    scene_path = sample_dir / "demo_scene.json"
    scene = load_scene(scene_path)

    # Phase 1: cold-start full rebuild plus cross-path byte comparison.
    consistency = assert_paths_identical(scene)
    cold_db = data_dir / "cold_session.sqlite3"
    cold = run_and_persist(
        scene,
        cold_db,
        path_name=PATH_FULL,
        snapshot_dir=snapshot_dir,
    )
    logger.log("INFO", "OK", "cold start full rebuild committed", rounds=cold["committed_rounds"])

    # Phase 2: deterministic fault injections and incremental advance.
    scene_dict = json.loads(scene_path.read_text(encoding="utf-8"))
    fault_results = {}
    for kind, context in (
        (FAULT_CORRUPT_SCENE, {"scene_dict": scene_dict}),
        (FAULT_BAD_RATIO, {"ratio": 1.25}),
        (FAULT_MISSING_SNAPSHOT_VERSION, None),
    ):
        if kind == FAULT_MISSING_SNAPSHOT_VERSION:
            temp_db = data_dir / "snapshot_fault.sqlite3"
            ledger = Ledger(temp_db)
            try:
                ledger.initialize_session(scene, PATH_INCREMENTAL)
                snapshot_file = write_snapshot(snapshot_dir, scene, ledger)
            finally:
                ledger.close()
            strip_snapshot_version(snapshot_file)
            context = {
                "snapshot_dir": snapshot_dir,
                "scene_id": scene.scene_id,
            }
        try:
            result = exercise_fault(kind, context, logger)
            fault_results[kind] = result.to_dict()
        except Exception as exc:  # logged inside exercise_fault
            fault_results[kind] = {
                "ok": False,
                "error": exc.to_dict() if hasattr(exc, "to_dict") else {"code": type(exc).__name__},
            }

    incremental_db = data_dir / "incremental_session.sqlite3"
    incremental = run_and_persist(scene, incremental_db, path_name=PATH_INCREMENTAL)
    logger.log(
        "INFO",
        "OK",
        "incremental migration committed after fault injections",
        rounds=incremental["committed_rounds"],
    )

    # Phase 3: forced termination mid-commit then ledger-based recovery.
    crash_db = data_dir / "crash_session.sqlite3"
    if crash_db.exists():
        crash_db.unlink()
    crash = crash_and_recover(
        scene,
        crash_db,
        crash_round=scene.rounds[3],
        log_path=log_path,
        snapshot_dir=snapshot_dir,
        path_name=PATH_INCREMENTAL,
    )

    replay_detail = crash["replay"]

    summary = {
        "scene": {
            "scene_id": scene.scene_id,
            "scene_version": scene.scene_version,
            "scale": scene.scale(),
        },
        "cold_start": {
            "path": PATH_FULL,
            "planner": consistency["planner"],
            "canonical_bytes_per_round": consistency["canonical_bytes_per_round"],
            "ledger": cold,
        },
        "fault_injection": fault_results,
        "incremental": incremental,
        "crash_recovery": crash,
        "replay": replay_detail,
        "log_file": str(log_path),
    }
    report_path = data_dir / "demo_summary.json"
    report_path.write_text(canonical_text(summary) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    import sys

    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    summary = run_demo(workspace)
    print(canonical_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
