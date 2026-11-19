from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import panorama_pipeline.snapshot as snapshot_module
from panorama_pipeline.pipeline import build_session
from panorama_pipeline.snapshot import SnapshotStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--snapshots", required=True)
    args = parser.parse_args()
    session = build_session(args.sample)
    store = SnapshotStore(args.snapshots)
    real_replace = snapshot_module.os.replace

    def crash_before_pointer_publish(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == store.current_link:
            os.kill(os.getpid(), signal.SIGKILL)
        real_replace(source, destination)

    snapshot_module.os.replace = crash_before_pointer_publish
    store.commit(session, report_payload={"phase": "about_to_publish"})
    print("UNREACHABLE", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
