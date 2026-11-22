from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from driftwatch.engine import DriftEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--snapshot", type=int, required=True)
    args = parser.parse_args()
    engine = DriftEngine.from_files(args.snapshots, args.agents, args.criteria, args.db)
    marker = Path(args.marker)

    def hook(event: str, payload: dict) -> None:
        if event == "batch_begun" and payload["snapshot_seq"] == args.snapshot:
            marker.write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(120)

    try:
        engine.run(lifecycle_hook=hook)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
