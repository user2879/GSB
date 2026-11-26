"""Child process used by the forced-termination recovery test.

It installs the signal handler, commits one snapshot, then waits. The test
sends SIGTERM; the handler flushes a diagnostic and the parent verifies that
recovery relies solely on the committed snapshot.
"""

import sys
import time
from pathlib import Path

from fxrender.app import build_engine
from fxrender.pipeline import render_all
from fxrender.samplegen import generate_sample_bundle
from fxrender.snapshot import Snapshot, SnapshotStore, TerminationHandler


def main() -> None:
    work = Path(sys.argv[1])
    bundle = generate_sample_bundle(work / "samples", frame_count=2)
    engine, _ = build_engine(bundle, work / "cache", max_workers=1)
    result = render_all(engine)
    store = SnapshotStore(work / "snap")

    def flusher():
        return Snapshot(
            global_version=engine.state.version,
            config_hash=engine.state.config_hash(),
            state_fp=engine.state.state_fp(),
            scheduler_state=getattr(engine, "_last_scheduler_state", {}),
            completed_frames=result.completed_frames,
            completed_tiles={k: list(v)
                             for k, v in result.completed_tiles.items()},
            consumed_budget=result.consumed_budget,
            pending_items=result.pending_items,
            failures=result.failures,
            cache_root=str(work / "cache"))

    snapshot = flusher()
    store.write_staged(snapshot, snapshot.global_version,
                       snapshot.config_hash)
    handler = TerminationHandler(store, flusher)
    handler.install()
    print("ready", flush=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(0.05)


if __name__ == "__main__":
    main()
