"""Standalone fault injection driver producing structured JSONL logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dr_bench.injection import (
    FAULT_BAD_RATIO,
    FAULT_CORRUPT_SCENE,
    FAULT_MISSING_SNAPSHOT_VERSION,
    StructuredLogger,
    exercise_fault,
    strip_snapshot_version,
)
from dr_bench.ledger import Ledger, write_snapshot
from dr_bench.loader import load_scene
from dr_bench.engine import PATH_INCREMENTAL


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject the three supported fault classes")
    parser.add_argument("scene")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--ratio", type=float, default=1.5)
    args = parser.parse_args()
    workspace = Path(args.workspace)
    log_path = workspace / "data" / "injection.log.jsonl"
    logger = StructuredLogger(log_path)
    scene_path = Path(args.scene)
    scene_dict = json.loads(scene_path.read_text(encoding="utf-8"))
    results = {}
    for kind, context in (
        (FAULT_CORRUPT_SCENE, {"scene_dict": scene_dict}),
        (FAULT_BAD_RATIO, {"ratio": args.ratio}),
        (FAULT_MISSING_SNAPSHOT_VERSION, None),
    ):
        if kind == FAULT_MISSING_SNAPSHOT_VERSION:
            scene = load_scene(scene_path)
            db = workspace / "data" / "injection_snapshot.sqlite3"
            snapshot_dir = workspace / "snapshots"
            ledger = Ledger(db)
            try:
                ledger.initialize_session(scene, PATH_INCREMENTAL)
                snapshot_file = write_snapshot(snapshot_dir, scene, ledger)
            finally:
                ledger.close()
            strip_snapshot_version(snapshot_file)
            context = {"snapshot_dir": snapshot_dir, "scene_id": scene.scene_id}
        try:
            exercise_fault(kind, context, logger)
            results[kind] = "OK"
        except Exception as exc:
            results[kind] = getattr(exc, "code", type(exc).__name__)
    print(json.dumps(results, sort_keys=True, indent=2))
    print(f"structured log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
