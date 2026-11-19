from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panorama_pipeline.pipeline import build_session
from panorama_pipeline.snapshot import GracefulShutdown, SnapshotStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--snapshots", required=True)
    args = parser.parse_args()
    shutdown = GracefulShutdown()
    shutdown.install()
    session = build_session(args.sample)
    store = SnapshotStore(args.snapshots)
    store.commit(session, report_payload={"phase": "ready"})
    print("READY", flush=True)
    while not shutdown.requested:
        time.sleep(0.01)
    store.commit(session, report_payload={"phase": "graceful_shutdown"})
    print("COMMITTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
