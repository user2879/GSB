from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .service import WorkbenchService
from .util import canonical_json


def _sample(index: int) -> list[float]:
    return [((index + offset * 3) % 19) / 19.0 for offset in range(8)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crash-recovery demo child")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args(argv)

    service = WorkbenchService(args.spec, args.db)
    for index in range(24):
        service.ingest(
            f"worker-client-{index % 7}",
            "alpha:worker" if index % 2 == 0 else "beta:worker",
            _sample(index),
        )
    committed_snapshot = service.rebuild_now()
    rebuild_id = service.start_rebuild(pause_ms=30_000)
    marker = {
        "pid": __import__("os").getpid(),
        "initial_snapshot_id": committed_snapshot.snapshot_id,
        "initial_snapshot_digest": committed_snapshot.content_digest,
        "initial_source_version": committed_snapshot.source_ledger_version,
        "running_rebuild_id": rebuild_id,
    }
    Path(args.marker).write_text(canonical_json(marker) + "\n", encoding="utf-8")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())

